"""P11-FE383 — SLA-style logit-lens window probe on Qwen-2.5-1.5B.

VISTA's Self-Logits-Augmentation (SLA) averages the LM head applied to a window
of late-layer hidden states. Here, for each of 500 MATH-500 final-token
activations we apply the Qwen LM head to the L24-L27 hidden states, average to
get o_aug = (1/4)·Σ_l H(h^l), and ensemble with the L28 logits as
õ = (1-γ)·o_L28 + γ·o_aug for γ ∈ {0.1, 0.2, 0.3, 0.4}. We then train a linear
correctness probe on the rank of the gold answer's first token in õ and compare
the OOF AUROC against 0.7186 (final-token L19 single-layer DoM baseline) and
0.7731 (prefill L19 baseline).

Hypothesis (re F-9 redundancy claim): if a SLA-style window probe beats the
single-layer baselines by more than F-9's 0.014 redundancy margin, then
multi-layer ensembling carries non-redundant correctness signal and F-9 is wrong.

This recompute reads cached final-token multi-layer hidden states, the Qwen LM
head + final-norm weights, and the gold first-token ids — none of which live in
the L19-only prefill cache, so they are loaded from dedicated Stage-2 NPZs and
the script degrades gracefully (MISSING_REGEN_INPUT, exit 2) if absent.
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
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
# L19-only prefill cache — used here only for the ground-truth correctness labels.
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
# Stage-2 final-token multi-layer hidden states: keys h24,h25,h26,h27,h28 each (500,1536).
LAYER_NPZ = ROOT / "pathway11_h100/logit_lens/cache/m15b_finaltok_layers.npz"
# Qwen LM head + final RMSNorm weights: lm_head (V,1536), [final_norm_weight (1536,)], [lm_head_bias (V,)].
HEAD_NPZ = ROOT / "pathway11_h100/logit_lens/cache/qwen15b_lm_head.npz"
# Gold answer first-token ids: gold_first_token (500,) int.
GOLD_NPZ = ROOT / "pathway11_h100/logit_lens/cache/m15b_gold_first_token.npz"

OUT_JSON = ROOT / "pathway11_h100/sla_logit_lens/results.json"

GAMMAS = [0.1, 0.2, 0.3, 0.4]
AUG_LAYERS = ["h24", "h25", "h26", "h27"]
BASE_LAYER = "h28"
RMS_EPS = 1e-6
N_FOLDS = 5
SEED = 9999

BASELINE_FINAL_L19 = 0.7186
BASELINE_PREFILL_L19 = 0.7731
F9_REDUNDANCY_MARGIN = 0.014


def auroc(scores, labels):
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def rms_norm(H, weight, eps=RMS_EPS):
    """Qwen-style RMSNorm applied row-wise before the LM head (logit-lens convention)."""
    ms = np.mean(H * H, axis=1, keepdims=True)
    Hn = H / np.sqrt(ms + eps)
    if weight is not None:
        Hn = Hn * weight[None, :]
    return Hn


def logit_lens(H, W, norm_w, bias):
    """Project hidden states to vocab logits: logits = RMSNorm(H) @ W.T (+ bias)."""
    Hn = rms_norm(H.astype(np.float64), norm_w)
    logits = Hn @ W.T
    if bias is not None:
        logits = logits + bias[None, :]
    return logits


def gold_rank(logits, gold_ids):
    """Rank (0 = top) of each row's gold token among all vocab logits, descending."""
    n = logits.shape[0]
    ranks = np.empty(n, dtype=np.float64)
    for i in range(n):
        # rankdata on -logits: smallest rank = highest logit; subtract 1 for 0-based.
        ranks[i] = rankdata(-logits[i], method="ordinal")[gold_ids[i]] - 1.0
    return ranks


def oof_probe_auroc(feature, y):
    """OOF 5-fold logistic-regression probe on a single rank feature."""
    feature = feature.reshape(-1, 1).astype(np.float64)
    oof = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(feature, y):
        mu = feature[tr].mean()
        sd = feature[tr].std() + 1e-12
        Xtr = (feature[tr] - mu) / sd
        Xte = (feature[te] - mu) / sd
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
    return auroc(oof, y)


def main() -> int:
    for p in (CACHE, LAYER_NPZ, HEAD_NPZ, GOLD_NPZ):
        if not p.exists():
            print("MISSING_REGEN_INPUT", p, file=sys.stderr)
            return 2

    correct = np.load(CACHE)["correct"].astype(bool)
    assert correct.shape == (500,)

    layers = np.load(LAYER_NPZ)
    for k in AUG_LAYERS + [BASE_LAYER]:
        if k not in layers:
            print("MISSING_REGEN_INPUT", f"{LAYER_NPZ}:{k}", file=sys.stderr)
            return 2

    head = np.load(HEAD_NPZ)
    if "lm_head" not in head:
        print("MISSING_REGEN_INPUT", f"{HEAD_NPZ}:lm_head", file=sys.stderr)
        return 2
    W = head["lm_head"].astype(np.float64)
    norm_w = head["final_norm_weight"].astype(np.float64) if "final_norm_weight" in head else None
    bias = head["lm_head_bias"].astype(np.float64) if "lm_head_bias" in head else None

    gold = np.load(GOLD_NPZ)
    if "gold_first_token" not in gold:
        print("MISSING_REGEN_INPUT", f"{GOLD_NPZ}:gold_first_token", file=sys.stderr)
        return 2
    gold_ids = gold["gold_first_token"].astype(np.int64)
    assert gold_ids.shape == (500,)
    vocab = W.shape[0]
    if gold_ids.min() < 0 or gold_ids.max() >= vocab:
        print("MISSING_REGEN_INPUT", "gold_first_token out of vocab range", file=sys.stderr)
        return 2

    # Base L28 logits and the averaged L24-L27 augmentation, computed layer-by-layer
    # to keep peak memory near two vocab-width buffers rather than five.
    o_base = logit_lens(layers[BASE_LAYER].astype(np.float64), W, norm_w, bias)
    o_aug = np.zeros_like(o_base)
    for k in AUG_LAYERS:
        o_aug += logit_lens(layers[k].astype(np.float64), W, norm_w, bias)
    o_aug /= float(len(AUG_LAYERS))

    # Single-layer L28 logit-lens reference (γ = 0): rank-of-gold probe.
    rank_base = gold_rank(o_base, gold_ids)
    auroc_l28 = oof_probe_auroc(-rank_base, correct)

    per_gamma = {}
    best_gamma = None
    best_auroc = -1.0
    for g in GAMMAS:
        o_tilde = (1.0 - g) * o_base + g * o_aug
        ranks = gold_rank(o_tilde, gold_ids)
        # Lower rank (gold near the top) should indicate correctness, hence the sign flip.
        a = oof_probe_auroc(-ranks, correct)
        per_gamma[f"{g:.1f}"] = float(a)
        if a > best_auroc:
            best_auroc = a
            best_gamma = g

    delta_vs_final = best_auroc - BASELINE_FINAL_L19
    delta_vs_prefill = best_auroc - BASELINE_PREFILL_L19
    f9_overturned = bool(delta_vs_final > F9_REDUNDANCY_MARGIN)

    out = {
        "experiment": "P11-FE383",
        "description": "SLA-style logit-lens window probe (L24-L27 aug + L28 base) on Qwen-2.5-1.5B final-token activations",
        "n": int(len(correct)),
        "vocab": int(vocab),
        "aug_layers": AUG_LAYERS,
        "base_layer": BASE_LAYER,
        "gammas": GAMMAS,
        "auroc_l28_single_layer": float(auroc_l28),
        "auroc_per_gamma": per_gamma,
        "best_gamma": float(best_gamma),
        "best_auroc": float(best_auroc),
        "baseline_final_l19_dom": BASELINE_FINAL_L19,
        "baseline_prefill_l19_dom": BASELINE_PREFILL_L19,
        "f9_redundancy_margin": F9_REDUNDANCY_MARGIN,
        "delta_vs_final_l19": float(delta_vs_final),
        "delta_vs_prefill_l19": float(delta_vs_prefill),
        "f9_overturned": f9_overturned,
        "verdict": (
            "MULTI_LAYER_NON_REDUNDANT" if f9_overturned else "REDUNDANT_WITH_SINGLE_LAYER"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())