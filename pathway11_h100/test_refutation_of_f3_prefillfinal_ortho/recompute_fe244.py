"""P11-FE244 — Is F-3 prefill⊥final orthogonality a structural NC3 violation?

Tests whether F-3's prefill/final DoM orthogonality is reasoning-specific or a
generic classifier-vs-class-mean artifact (Wu & Papyan §4.3, NC3 violation in
CLMs over imbalanced vocabularies). Final-token L19 activations are projected
through W_unembed; prefill activations are not. We compute cosine similarity
between each DoM direction (final-token L19 DoM, prefill L19 DoM) and the
unembedding rows W_unembed[t] for the top-50 most-frequent MATH-500 answer
tokens. If final_DoM aligns with unembed rows (|cos| > 0.3) but prefill_DoM does
not, the orthogonality is the structural projection effect, not a finding about
reasoning geometry.
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
FINAL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz"
WUNEMBED = ROOT / "pathway11_h100/data/w_unembed.npz"
OUT_JSON = ROOT / "pathway11_h100/nc3_unembed_alignment/results.json"

ALIGN_THRESH = 0.3
TOP_K_TOKENS = 50


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def dom_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Class-mean difference direction (correct minus incorrect)."""
    return X[y].mean(axis=0) - X[~y].mean(axis=0)


def cos_stats(dom_unit: np.ndarray, W_rows_unit: np.ndarray) -> dict:
    """|cos| alignment of a DoM direction against a set of unit unembed rows.

    DoM sign is arbitrary (correct minus incorrect), so alignment uses |cos|.
    """
    cos = W_rows_unit @ dom_unit
    abs_cos = np.abs(cos)
    return {
        "n_tokens": int(W_rows_unit.shape[0]),
        "mean_abs_cos": float(abs_cos.mean()),
        "max_abs_cos": float(abs_cos.max()),
        "median_abs_cos": float(np.median(abs_cos)),
        "frac_aligned": float((abs_cos > ALIGN_THRESH).mean()),
        "n_aligned": int((abs_cos > ALIGN_THRESH).sum()),
        "any_aligned": bool((abs_cos > ALIGN_THRESH).any()),
    }


def main() -> int:
    for p in (CACHE, FINAL_CACHE, WUNEMBED):
        if not p.exists():
            print("MISSING_REGEN_INPUT", p, file=sys.stderr)
            return 2

    pre_blob = np.load(CACHE)
    X_prefill = pre_blob["prefill"].astype(np.float64)
    y = pre_blob["correct"].astype(bool)
    assert X_prefill.shape == (500, 1536) and y.shape == (500,)

    fin_blob = np.load(FINAL_CACHE)
    final_key = "final" if "final" in fin_blob.files else fin_blob.files[0]
    X_final = fin_blob[final_key].astype(np.float64)
    if X_final.shape != (500, 1536):
        print("MISSING_REGEN_INPUT", FINAL_CACHE, "bad shape", X_final.shape, file=sys.stderr)
        return 2

    wblob = np.load(WUNEMBED)
    W = wblob["W_unembed"].astype(np.float64)            # (V, 1536)
    if W.ndim != 2 or W.shape[1] != 1536:
        print("MISSING_REGEN_INPUT", WUNEMBED, "bad W shape", W.shape, file=sys.stderr)
        return 2
    # All MATH-500 answer-token ids (with repetition) to rank by frequency.
    answer_tokens = wblob["answer_tokens"].astype(np.int64).ravel()
    ids, counts = np.unique(answer_tokens, return_counts=True)
    order = np.argsort(-counts)
    top_ids = ids[order][:TOP_K_TOKENS]
    top_ids = top_ids[(top_ids >= 0) & (top_ids < W.shape[0])]

    # Unit-normalize the top-K unembed rows once.
    W_rows = W[top_ids]
    W_rows_unit = W_rows / np.clip(np.linalg.norm(W_rows, axis=1, keepdims=True), 1e-12, None)

    d_prefill = dom_direction(X_prefill, y)
    d_final = dom_direction(X_final, y)
    u_prefill = unit(d_prefill)
    u_final = unit(d_final)

    final_stats = cos_stats(u_final, W_rows_unit)
    prefill_stats = cos_stats(u_prefill, W_rows_unit)

    cos_prefill_final = float(np.abs(u_prefill @ u_final))

    # Refutation verdict: F-3 framing is over-interpreted (structural NC3 effect)
    # iff final DoM aligns with unembed rows but prefill DoM does not.
    refutes_f3 = bool(final_stats["any_aligned"] and not prefill_stats["any_aligned"])

    out = {
        "experiment": "P11-FE244",
        "align_threshold": ALIGN_THRESH,
        "top_k_tokens": int(len(top_ids)),
        "final_dom_vs_unembed": final_stats,
        "prefill_dom_vs_unembed": prefill_stats,
        "cos_prefill_final_abs": cos_prefill_final,
        "auroc_prefill_dom": auroc(X_prefill @ u_prefill, y),
        "auroc_final_dom": auroc(X_final @ u_final, y),
        "refutes_f3_as_structural_nc3": refutes_f3,
        "interpretation": (
            "REFUTES_F3: final DoM aligns with W_unembed rows but prefill does "
            "not — orthogonality is a generic classifier-vs-class-mean (NC3) "
            "projection effect, not reasoning-specific."
            if refutes_f3 else
            "DOES_NOT_REFUTE: alignment pattern does not match the structural "
            "NC3 prediction; F-3 orthogonality survives this control."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())