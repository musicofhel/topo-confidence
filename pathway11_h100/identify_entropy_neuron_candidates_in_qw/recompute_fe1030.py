"""P11-FE1030 — Entropy-neuron candidates in Qwen-2.5-1.5B and their correlation with DoM.

Entropy neurons (Stolfo et al.; Gurnee et al.) are MLP neurons that write a
large-norm vector into the residual stream which lands (mostly) in the null
space of the unembedding: high ||w_out|| but low var(W_U . w_out). Instead of
moving logits directly, they rescale the residual norm and thereby modulate the
post-LayerNorm logit *temperature* — i.e. they regulate next-token entropy.

This script:
  1. Loads the L19 MLP down-projection weights (w_out, one row per neuron) and
     the unembedding W_U for Qwen-2.5-1.5B from a cached NPZ.
  2. Scores every neuron by norm_i = ||w_out[i]|| and logit_var_i =
     var_vocab(w_out[i] @ W_U), computed in vocab chunks to stay memory-safe.
  3. Ranks entropy-neuron candidates by norm_i / (logit_var_i + eps).
  4. If cached per-neuron activations on the 500 MATH-500 prefills are present,
     correlates each candidate's activation with the DoM score and measures its
     standalone AUROC against correctness — testing whether the first documented
     uncertainty-quantification mechanism grounds F-2 / F-8.

Weights/activations are not loadable CPU-only via transformers here, so they
must be pre-extracted into the entropy-neuron NPZ; the script degrades
gracefully when inputs are missing.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
# Pre-extracted L19 MLP down-proj (w_out), unembedding (W_U), and optional
# per-neuron activations on the 500 prefills. Keys (any orientation tolerated):
#   w_out:        (d_mlp, d_model)      MLP down-projection, one row per neuron
#   W_U:          (d_model, vocab)      unembedding (lm_head) weight
#   neuron_acts:  (500, d_mlp)          optional: L19 MLP neuron pre-down acts
ENTROPY_NPZ = ROOT / "pathway11_h100/entropy_neurons/cache/qwen15b_l19_mlp.npz"
OUT_JSON = ROOT / "pathway11_h100/entropy_neurons/results.json"

D_MODEL = 1536
TOP_K = 20
VOCAB_CHUNK = 8192
EPS = 1e-12


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < EPS:
        return float("nan")
    return float((a * b).sum() / denom)


def _orient_w_out(w_out: np.ndarray) -> np.ndarray:
    # want (d_mlp, d_model)
    if w_out.shape[1] == D_MODEL:
        return w_out
    if w_out.shape[0] == D_MODEL:
        return w_out.T
    raise ValueError(f"w_out shape {w_out.shape} has no axis == d_model {D_MODEL}")


def _orient_w_u(W_U: np.ndarray) -> np.ndarray:
    # want (d_model, vocab)
    if W_U.shape[0] == D_MODEL:
        return W_U
    if W_U.shape[1] == D_MODEL:
        return W_U.T
    raise ValueError(f"W_U shape {W_U.shape} has no axis == d_model {D_MODEL}")


def logit_var_per_neuron(w_out: np.ndarray, W_U: np.ndarray) -> np.ndarray:
    """var over vocab of (w_out @ W_U), computed in vocab chunks.

    w_out: (d_mlp, d_model) float32, W_U: (d_model, vocab) float32.
    Returns (d_mlp,) float64 of per-neuron logit variance.
    """
    d_mlp = w_out.shape[0]
    vocab = W_U.shape[1]
    s1 = np.zeros(d_mlp, dtype=np.float64)
    s2 = np.zeros(d_mlp, dtype=np.float64)
    for start in range(0, vocab, VOCAB_CHUNK):
        stop = min(start + VOCAB_CHUNK, vocab)
        L = w_out @ W_U[:, start:stop]          # (d_mlp, chunk)
        L = L.astype(np.float64, copy=False)
        s1 += L.sum(axis=1)
        s2 += (L * L).sum(axis=1)
    mean = s1 / vocab
    var = s2 / vocab - mean * mean
    return np.maximum(var, 0.0)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2
    if not ENTROPY_NPZ.exists():
        print("MISSING_REGEN_INPUT", ENTROPY_NPZ, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    correct = cache["correct"].astype(bool)
    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert correct.shape == (500,) and dom_score.shape == (500,)

    blob = np.load(ENTROPY_NPZ)
    if "w_out" not in blob or "W_U" not in blob:
        print("MISSING_REGEN_INPUT", "w_out/W_U keys in", ENTROPY_NPZ, file=sys.stderr)
        return 2

    w_out = _orient_w_out(blob["w_out"].astype(np.float32))
    W_U = _orient_w_u(blob["W_U"].astype(np.float32))
    d_mlp = w_out.shape[0]
    vocab = W_U.shape[1]

    # Per-neuron weight statistics.
    norm = np.linalg.norm(w_out.astype(np.float64), axis=1)          # ||w_out_i||
    logit_var = logit_var_per_neuron(w_out, W_U)                      # var(W_U . w_out_i)

    # Entropy-neuron signature: large residual-write norm, small logit effect.
    # Rank by norm relative to logit variance; standardize for a clean ratio.
    norm_z = (norm - norm.mean()) / (norm.std() + EPS)
    lv_z = (logit_var - logit_var.mean()) / (logit_var.std() + EPS)
    cand_score = norm / (np.sqrt(logit_var) + EPS)
    order = np.argsort(cand_score)[::-1]
    candidates = order[:TOP_K]

    out = {
        "experiment": "P11-FE1030",
        "d_mlp": int(d_mlp),
        "d_model": int(D_MODEL),
        "vocab": int(vocab),
        "top_k": int(TOP_K),
        "norm_mean": float(norm.mean()),
        "norm_max": float(norm.max()),
        "logit_var_mean": float(logit_var.mean()),
        "logit_var_min": float(logit_var.min()),
        "candidate_neurons": [int(i) for i in candidates],
        "candidate_norm": [float(norm[i]) for i in candidates],
        "candidate_logit_var": [float(logit_var[i]) for i in candidates],
        "candidate_norm_z": [float(norm_z[i]) for i in candidates],
        "candidate_logit_var_z": [float(lv_z[i]) for i in candidates],
        "auroc_dom_baseline": float(auroc(dom_score, correct)),
    }

    # Activation-level correlation, when per-neuron activations are cached.
    if "neuron_acts" in blob:
        acts = blob["neuron_acts"].astype(np.float64)
        if acts.shape == (d_mlp, 500):
            acts = acts.T
        if acts.shape != (500, d_mlp):
            print("MISSING_REGEN_INPUT", "neuron_acts shape", acts.shape, file=sys.stderr)
            return 2
        corr_dom = []
        auroc_act = []
        for i in candidates:
            a = acts[:, i]
            corr_dom.append(pearson(a, dom_score))
            ac = auroc(a, correct)
            # report the better-oriented standalone AUROC
            auroc_act.append(float(max(ac, 1.0 - ac)) if np.isfinite(ac) else float("nan"))
        abs_corr = [abs(c) for c in corr_dom if np.isfinite(c)]
        out.update({
            "has_activations": True,
            "candidate_corr_with_dom": corr_dom,
            "candidate_standalone_auroc": auroc_act,
            "max_abs_corr_with_dom": float(max(abs_corr)) if abs_corr else float("nan"),
            "mean_abs_corr_with_dom": float(np.mean(abs_corr)) if abs_corr else float("nan"),
            "best_candidate_auroc": float(np.nanmax(auroc_act)) if auroc_act else float("nan"),
        })
        # Aggregate entropy-neuron signal: mean of standardized candidate acts.
        agg = acts[:, candidates].mean(axis=1)
        agg_ac = auroc(agg, correct)
        out["aggregate_candidate_auroc"] = float(max(agg_ac, 1.0 - agg_ac)) if np.isfinite(agg_ac) else float("nan")
        out["aggregate_candidate_corr_with_dom"] = pearson(agg, dom_score)
    else:
        out["has_activations"] = False

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())