"""P11-FE263 — Sparsity proxy for F-2's L19 prefill DoM direction (W_V vocab-projection shadow).

The full FE263 protocol projects the ~8960 columns of the Qwen2.5-1.5B L19
second MLP matrix (W_V) onto the top-200 vocab tokens via the unembedding and
GPT-4-scores each token set for math-correctness / answer-formatting /
numeric-reasoning coherence, then checks whether the candidate correctness-coded
columns' activations m_j^L19 reproduce the prefill DoM scalar. That parametric
vocab-projection plus the LLM scoring require the model weights and network
access — neither is available in the CPU-only / cached-NPZ recompute
environment (HARD CONSTRAINTS: no GPU, no transformers, no network). This script
therefore runs the *activation-side shadow* of the identical falsifiable
question that IS computable from cached residual-stream states:

    Does the L19 prefill DoM AUROC (0.7731, F-2) reduce to a SPARSE subset of
    coordinates, or is it an irreducibly DENSE direction?

For each OOF fold we fit the DoM direction d_vec on train, rank the 1536
coordinates by a logit-mean-analog importance (|d_vec[j]| * std_j), drop the
bottom 30% (mirroring FE263's "drop bottom 30%" step), then sweep top-K
retention. We report how many coordinates recover 95% of the full-dim OOF AUROC
and correlate the sparse-projected score with the canonical prefill DoM scalar.
Small K (sparse) supports re-framing F-2 as a sparse parametric trace; large K
(dense) keeps F-2 as a dense direction with no clean vocab-side reduction.
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
OUT_JSON = ROOT / "pathway11_h100/wv_vocab_sparsity/results.json"

N_FOLDS = 5
SEED = 9999
DROP_FRAC = 0.30
TOPK_GRID = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
RECOVER_FRAC = 0.95


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    if X.shape != (500, 1536) or y.shape != (500,):
        print("UNEXPECTED_SHAPE", X.shape, y.shape, file=sys.stderr); return 3

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    if dom_score.shape != (500,):
        print("UNEXPECTED_SHAPE", dom_score.shape, file=sys.stderr); return 3

    n, d = X.shape
    folds = stratified_kfold(y, N_FOLDS, SEED)
    n_keep_after_drop = int(round((1.0 - DROP_FRAC) * d))

    # OOF dense (all-coordinate) DoM, and OOF sparse scores per top-K.
    oof_dense = np.zeros(n, dtype=np.float64)
    oof_sparse = {k: np.zeros(n, dtype=np.float64) for k in TOPK_GRID}

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]

        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        std_tr = Xtr.std(axis=0) + 1e-12
        importance = np.abs(d_vec) * std_tr
        order = np.argsort(importance)[::-1]
        allowed = order[:n_keep_after_drop]  # surviving 70% of coordinates

        oof_dense[test_idx] = Xte @ d_vec
        for k in TOPK_GRID:
            sel = allowed[:k] if k <= n_keep_after_drop else order[:k]
            oof_sparse[k][test_idx] = Xte[:, sel] @ d_vec[sel]

    auroc_dense = auroc(oof_dense, y)
    auroc_dom_scalar = auroc(dom_score, y)
    target = RECOVER_FRAC * auroc_dense

    sparse_curve = {}
    min_k_for_95 = None
    for k in TOPK_GRID:
        a = auroc(oof_sparse[k], y)
        # correlation of sparse-projected score with canonical prefill DoM scalar
        if np.std(oof_sparse[k]) > 0 and np.std(dom_score) > 0:
            corr = float(np.corrcoef(oof_sparse[k], dom_score)[0, 1])
        else:
            corr = float("nan")
        sparse_curve[str(k)] = {"auroc_oof": a, "corr_with_dom_scalar": corr}
        if min_k_for_95 is None and a >= target:
            min_k_for_95 = k

    corr_dense_scalar = float(np.corrcoef(oof_dense, dom_score)[0, 1])

    verdict = (
        "SPARSE" if (min_k_for_95 is not None and min_k_for_95 <= 64)
        else "DENSE" if min_k_for_95 is not None
        else "NO_RECOVERY"
    )

    out = {
        "experiment": "P11-FE263",
        "note": (
            "Activation-side sparsity shadow of the W_V vocab-projection sweep; "
            "the parametric vocab projection + GPT-4 token-set scoring are not "
            "CPU/offline-runnable (require model weights + network)."
        ),
        "n": int(n),
        "d": int(d),
        "drop_frac": DROP_FRAC,
        "n_coords_after_drop": int(n_keep_after_drop),
        "recover_frac_target": RECOVER_FRAC,
        "auroc_dom_scalar_reference": auroc_dom_scalar,
        "auroc_dense_oof": auroc_dense,
        "corr_dense_oof_with_dom_scalar": corr_dense_scalar,
        "sparse_topk_curve": sparse_curve,
        "min_k_for_95pct_recovery": min_k_for_95,
        "verdict": verdict,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())