"""P11-FE419 — Layerwise logit-lens trajectory on Qwen-2.5-1.5B (MATH-500).

At each transformer layer, project the prefill residual stream through the
(RMS-normed) unembedding and measure a correctness-correlated logit margin
(answer token vs. best distractor). We then take the per-problem layer margins
as scores and compute AUROC against ground-truth correctness at every layer,
asking *where* the math-correctness signal localizes.

Motivation: F-2 treats L19 as "THE" probe site. If the signal peaks well before
L19 — matching the intent-first staging reported for TinySQL in 2503.12730 §6
(intent → grounding → assembly) — then "L19 is THE site" weakens to "L19 is one
of several mid-stack layers", and the right cross-architecture abstraction is a
*depth fraction* (L_peak / n_layers) rather than an absolute index (F-1).

This is a CPU-only logit-lens recompute. It requires three cached Stage-2
artifacts that are NOT part of the single-layer m15b_prefill.npz bundle:

  1. a per-layer prefill residual cache  (500, n_layers, 1536),
  2. the model unembedding matrix W_U    (1536, vocab)  [+ optional RMSNorm γ],
  3. an answer/distractor token spec      per problem.

If any are absent the script prints MISSING_REGEN_INPUT and returns 2 — it never
fabricates a trajectory.
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
LAYERS_NPZ = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_layerwise.npz"
UNEMBED_NPZ = ROOT / "pathway11_h100/data/unembed/qwen15b_unembed.npz"
TOKENS_NPZ = ROOT / "pathway11_h100/data/unembed/answer_tokens.npz"
OUT_JSON = ROOT / "pathway11_h100/logit_lens_trajectory/results.json"

RMS_EPS = 1e-6


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def rmsnorm(x: np.ndarray, gamma: np.ndarray | None, eps: float = RMS_EPS) -> np.ndarray:
    """Qwen2.5 final-norm logit-lens convention: RMSNorm then unembed."""
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    xn = x / rms
    if gamma is not None:
        xn = xn * gamma
    return xn


def main() -> int:
    for path in (CACHE, LAYERS_NPZ, UNEMBED_NPZ, TOKENS_NPZ):
        if not path.exists():
            print("MISSING_REGEN_INPUT", path, file=sys.stderr)
            return 2

    # --- labels -------------------------------------------------------------
    base = np.load(CACHE)
    correct = base["correct"].astype(bool)
    n = correct.shape[0]
    assert n == 500, f"expected 500 problems, got {n}"

    # --- per-layer residuals ------------------------------------------------
    lblob = np.load(LAYERS_NPZ)
    key = next((k for k in ("prefill_layers", "layers", "hidden_states") if k in lblob), None)
    if key is None:
        print("MISSING_REGEN_INPUT", LAYERS_NPZ, "(no layerwise key)", file=sys.stderr)
        return 2
    layers = lblob[key].astype(np.float32)  # (500, n_layers, 1536)
    if layers.ndim != 3 or layers.shape[0] != n or layers.shape[2] != 1536:
        print("MISSING_REGEN_INPUT", LAYERS_NPZ, f"bad shape {layers.shape}", file=sys.stderr)
        return 2
    n_layers = layers.shape[1]

    # --- unembedding --------------------------------------------------------
    ublob = np.load(UNEMBED_NPZ)
    if "W_U" not in ublob:
        print("MISSING_REGEN_INPUT", UNEMBED_NPZ, "(no W_U)", file=sys.stderr)
        return 2
    W_U = ublob["W_U"].astype(np.float32)  # (1536, vocab)
    if W_U.shape[0] != 1536:
        if W_U.shape[1] == 1536:  # tolerate (vocab, 1536) orientation
            W_U = W_U.T
        else:
            print("MISSING_REGEN_INPUT", UNEMBED_NPZ, f"bad W_U {W_U.shape}", file=sys.stderr)
            return 2
    vocab = W_U.shape[1]
    gamma = ublob["ln_f_weight"].astype(np.float32) if "ln_f_weight" in ublob else None

    # --- token spec ---------------------------------------------------------
    tblob = np.load(TOKENS_NPZ)
    if "answer_token" not in tblob or "distractor_tokens" not in tblob:
        print("MISSING_REGEN_INPUT", TOKENS_NPZ, "(no token spec)", file=sys.stderr)
        return 2
    answer_tok = tblob["answer_token"].astype(np.int64)        # (500,)
    distractor_tok = tblob["distractor_tokens"].astype(np.int64)  # (500, D)
    if answer_tok.shape[0] != n or distractor_tok.shape[0] != n:
        print("MISSING_REGEN_INPUT", TOKENS_NPZ, "bad token-spec shape", file=sys.stderr)
        return 2
    n_distract = distractor_tok.shape[1]

    # Slice W_U to only the columns we actually need (answer + distractors).
    all_tok = np.concatenate([answer_tok[:, None], distractor_tok], axis=1)  # (500, 1+D)
    if all_tok.min() < 0 or all_tok.max() >= vocab:
        print("MISSING_REGEN_INPUT", TOKENS_NPZ, "token id out of range", file=sys.stderr)
        return 2
    uniq, inv = np.unique(all_tok.reshape(-1), return_inverse=True)
    inv = inv.reshape(all_tok.shape)  # (500, 1+D) -> column index into W_slice
    W_slice = W_U[:, uniq]  # (1536, U)

    rows = np.arange(n)

    # --- layerwise logit-lens margin trajectory -----------------------------
    per_layer = []
    margins_all = np.zeros((n_layers, n), dtype=np.float64)
    for li in range(n_layers):
        xn = rmsnorm(layers[:, li, :].astype(np.float32), gamma)  # (500, 1536)
        logits = xn @ W_slice                                     # (500, U)
        gathered = logits[rows[:, None], inv]                     # (500, 1+D)
        ans_logit = gathered[:, 0]
        dist_max = gathered[:, 1:].max(axis=1)
        margin = (ans_logit - dist_max).astype(np.float64)        # (500,)
        margins_all[li] = margin
        a = auroc(margin, correct)
        per_layer.append({
            "layer": li,
            "depth_fraction": round(li / max(n_layers - 1, 1), 4),
            "auroc": a,
            "mean_margin": float(margin.mean()),
            "mean_margin_correct": float(margin[correct].mean()) if correct.any() else float("nan"),
            "mean_margin_incorrect": float(margin[~correct].mean()) if (~correct).any() else float("nan"),
        })

    aurocs = np.array([d["auroc"] for d in per_layer], dtype=np.float64)
    finite = np.isfinite(aurocs)
    peak_layer = int(np.nanargmax(np.where(finite, aurocs, -np.inf)))
    peak_auroc = float(aurocs[peak_layer])

    # Is L19 the peak, or merely "one of several"? Flag layers within 0.01 of it.
    l19_auroc = float(aurocs[19]) if n_layers > 19 else float("nan")
    near_peak = [int(li) for li in range(n_layers)
                 if finite[li] and (peak_auroc - aurocs[li]) <= 0.01]
    localizes_before_l19 = bool(peak_layer < 19)

    out = {
        "experiment": "P11-FE419",
        "n_problems": int(n),
        "n_layers": int(n_layers),
        "n_distractors": int(n_distract),
        "rms_eps": RMS_EPS,
        "used_ln_f_weight": gamma is not None,
        "per_layer": per_layer,
        "peak_layer": peak_layer,
        "peak_depth_fraction": round(peak_layer / max(n_layers - 1, 1), 4),
        "peak_auroc": peak_auroc,
        "l19_auroc": l19_auroc,
        "near_peak_layers": near_peak,
        "n_near_peak_layers": len(near_peak),
        "localizes_before_l19": localizes_before_l19,
        # Interpretation guard for the F-2 narrative (see module docstring):
        "verdict": (
            "SIGNAL_EARLIER_THAN_L19" if localizes_before_l19
            else ("L19_IS_PEAK" if peak_layer == 19 else "PEAK_AFTER_L19")
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())