"""P11-FE349 — F-3 cross-check on untied-embedding Qwen2.5-7B.

F-3 reports cos(prefill_DoM, final_DoM) = 0.046 on Qwen2.5-1.5B (1024-tok),
near-orthogonality read as evidence of two functionally distinct circuits
(prefill-readout vs final-token). Qwen2.5-1.5B ties input/output embeddings
(Table 1 of 2412.15115); the 7B variant does not. If the small cosine is a
tied-embedding / vocabulary-space artifact rather than a circuit-level fact,
the untied 7B should show a substantially larger cosine.

This recomputes the DoM directions (mean_correct − mean_incorrect) separately
on cached 7B L19 prefill and final-token residuals, takes their cosine, and
compares to the 1.5B reference of 0.046. A fold-resampled distribution is
reported so the point estimate isn't read as exact. CPU-only, cached residuals.
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

# 7B cached residuals. Try a couple of plausible cache layouts: a single NPZ
# carrying both prefill and final, or two separate per-stage NPZs.
PREFILL_7B_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m7b_prefill.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m7b_residuals.npz",
    ROOT / "pathway11_h100/data/m7b_prefill.npz",
]
FINAL_7B_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m7b_final.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m7b_residuals.npz",
    ROOT / "pathway11_h100/data/m7b_final.npz",
]

OUT_JSON = ROOT / "pathway11_h100/f3_untied_7b/results.json"

REF_15B_COSINE = 0.046  # 1.5B tied-embedding value, scratch/pathway10_*.json
N_FOLDS = 5
SEED = 9999


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


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def pick_key(blob, names: list[str]) -> str | None:
    keys = list(blob.keys())
    for n in names:
        if n in keys:
            return n
    return None


def dom_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y].mean(axis=0) - X[~y].mean(axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    prefill_path = first_existing(PREFILL_7B_CANDIDATES)
    final_path = first_existing(FINAL_7B_CANDIDATES)
    if prefill_path is None:
        print("MISSING_REGEN_INPUT", PREFILL_7B_CANDIDATES[0], file=sys.stderr)
        return 2
    if final_path is None:
        print("MISSING_REGEN_INPUT", FINAL_7B_CANDIDATES[0], file=sys.stderr)
        return 2

    pre_blob = np.load(prefill_path)
    fin_blob = np.load(final_path)

    pre_key = pick_key(pre_blob, ["prefill", "prefill_resid", "residuals", "hidden", "X"])
    fin_key = pick_key(fin_blob, ["final", "final_resid", "final_token", "residuals", "hidden", "X"])
    if pre_key is None:
        print("MISSING_REGEN_INPUT", f"no prefill array in {prefill_path}", file=sys.stderr)
        return 2
    if fin_key is None:
        print("MISSING_REGEN_INPUT", f"no final array in {final_path}", file=sys.stderr)
        return 2

    X_pre = pre_blob[pre_key].astype(np.float64)
    X_fin = fin_blob[fin_key].astype(np.float64)

    y_key = pick_key(pre_blob, ["correct", "labels", "y"]) or pick_key(fin_blob, ["correct", "labels", "y"])
    if y_key is None:
        print("MISSING_REGEN_INPUT", "no correctness labels in 7B caches", file=sys.stderr)
        return 2
    y_src = pre_blob if y_key in pre_blob.keys() else fin_blob
    y = y_src[y_key].astype(bool)

    if X_pre.shape[0] != X_fin.shape[0] or X_pre.shape[0] != y.shape[0]:
        print("MISSING_REGEN_INPUT",
              f"row mismatch pre={X_pre.shape} fin={X_fin.shape} y={y.shape}",
              file=sys.stderr)
        return 2
    if X_pre.shape[1] != X_fin.shape[1]:
        print("MISSING_REGEN_INPUT",
              f"dim mismatch pre={X_pre.shape[1]} fin={X_fin.shape[1]}",
              file=sys.stderr)
        return 2

    n, d = X_pre.shape

    # Global DoM directions on the full 7B set.
    dom_pre = dom_direction(X_pre, y)
    dom_fin = dom_direction(X_fin, y)
    cos_global = cosine(dom_pre, dom_fin)

    # Single-direction separability sanity (in-sample, for context only).
    auroc_pre_insample = auroc(X_pre @ dom_pre, y)
    auroc_fin_insample = auroc(X_fin @ dom_fin, y)

    # Fold-resampled cosine distribution: recompute DoM on each held-out fold's
    # complement so the point estimate carries a spread, not a bare number.
    folds = stratified_kfold(y, N_FOLDS, SEED)
    fold_cosines = []
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        ytr = y[train_mask]
        if ytr.sum() == 0 or (~ytr).sum() == 0:
            continue
        dpre = dom_direction(X_pre[train_mask], ytr)
        dfin = dom_direction(X_fin[train_mask], ytr)
        fold_cosines.append(cosine(dpre, dfin))
    fold_cosines = [c for c in fold_cosines if not np.isnan(c)]

    cos_mean = float(np.mean(fold_cosines)) if fold_cosines else float("nan")
    cos_std = float(np.std(fold_cosines)) if fold_cosines else float("nan")

    out = {
        "experiment": "P11-FE349",
        "description": "F-3 cross-check: cos(prefill_DoM, final_DoM) on untied-embedding Qwen2.5-7B",
        "prefill_cache": str(prefill_path),
        "final_cache": str(final_path),
        "n_problems": int(n),
        "hidden_dim": int(d),
        "n_correct": int(y.sum()),
        "n_incorrect": int((~y).sum()),
        "cos_prefill_final_dom_7b": cos_global,
        "cos_prefill_final_dom_7b_fold_mean": cos_mean,
        "cos_prefill_final_dom_7b_fold_std": cos_std,
        "cos_prefill_final_dom_7b_folds": [float(c) for c in fold_cosines],
        "auroc_prefill_dom_insample": auroc_pre_insample,
        "auroc_final_dom_insample": auroc_fin_insample,
        "ref_cos_15b_tied": REF_15B_COSINE,
        "delta_vs_15b": float(cos_global - REF_15B_COSINE) if not np.isnan(cos_global) else float("nan"),
        "interpretation_note": (
            "Near-orthogonality (|cos| close to 0.046) on untied 7B supports a "
            "real circuit-level claim for F-3; a substantially larger |cos| would "
            "implicate a tied-embedding / vocabulary-space artifact in the 1.5B value."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"cos_7b={cos_global:.4f} (fold {cos_mean:.4f}±{cos_std:.4f}) vs 1.5B {REF_15B_COSINE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())