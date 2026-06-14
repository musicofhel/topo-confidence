"""P11-FE273 — TBG vs SLT cosine sanity check on Qwen-2.5-1.5B.

F-3 reports cos ≈ 0.046 between the prefill (TBG, last input token) and
final-token (SLT, last token of the generated answer) DoM directions for
correctness. Kossen et al. report similar AUROC at TBG and SLT for SE
prediction, which would be suggestive of *overlapping* directions.

This script trains two correctness probes — one at TBG (our prefill position)
and one at SLT (our final-token position) — as both raw DoM directions and
accuracy-supervised logistic-regression probes, then reports cos(w_TBG, w_SLT).

Decision rule: if cos(w_TBG, w_SLT) > 0.5, F-3 orthogonality is a property of
the residual-stream geometry rather than a correctness-specific divergence —
i.e. the falsifiable seam fails. If cos stays near 0.046, F-3 survives.

Requires a final-token (SLT) activation cache parallel to the prefill cache;
if none is present the script prints MISSING_REGEN_INPUT and returns 2.
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
from sklearn.linear_model import LogisticRegression

ROOT = Path("/home/musicofhel/topo-confidence")
PREFILL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/tbg_slt_cosine/results.json"

# Candidate locations / keys for the SLT (final-token) activation cache.
SLT_CACHE_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_slt.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_finaltoken.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final_token.npz",
    ROOT / "pathway11_h100/final_token/cache/m15b_final.npz",
    ROOT / "pathway11_h100/final_inversion/cache/m15b_final.npz",
]
SLT_KEY_CANDIDATES = ["final", "slt", "final_token", "finaltoken", "hidden", "h", "act"]

N_FOLDS = 5
SEED = 9999
COS_THRESHOLD = 0.5
LR_C = 0.001  # match the full-d L2-reg ceiling regime (best C from FE882)


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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def dom_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Difference-of-means correctness direction in raw activation space."""
    return X[y].mean(axis=0) - X[~y].mean(axis=0)


def lr_direction(X: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
    """Accuracy-supervised LR probe; weight mapped back to raw feature space."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd_safe = np.where(sd < 1e-12, 1.0, sd)
    Xs = (X - mu) / sd_safe
    clf = LogisticRegression(C=C, max_iter=5000, solver="lbfgs")
    clf.fit(Xs, y.astype(int))
    w_std = clf.coef_.ravel().astype(np.float64)
    return w_std / sd_safe  # back to raw space so cosine is comparable to DoM


def oof_auroc(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> float:
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        d = dom_direction(X[train_mask], y[train_mask])
        scores[test_idx] = X[test_idx] @ d
    return auroc(scores, y)


def load_slt() -> tuple[np.ndarray, str, str] | None:
    for path in SLT_CACHE_CANDIDATES:
        if not path.exists():
            continue
        blob = np.load(path)
        for key in SLT_KEY_CANDIDATES:
            if key in blob.files and blob[key].ndim == 2 and blob[key].shape == (500, 1536):
                return blob[key].astype(np.float64), str(path), key
        # fall back to the first (500, 1536) array if no named key matched
        for key in blob.files:
            arr = blob[key]
            if arr.ndim == 2 and arr.shape == (500, 1536):
                return arr.astype(np.float64), str(path), key
    return None


def fold_mean_cosine(Xa, Xb, y, folds, fit_fn) -> tuple[float, float]:
    cs = []
    for test_idx in folds:
        train_mask = np.ones(len(y), dtype=bool); train_mask[test_idx] = False
        wa = fit_fn(Xa[train_mask], y[train_mask])
        wb = fit_fn(Xb[train_mask], y[train_mask])
        cs.append(cosine(wa, wb))
    cs = np.array(cs, dtype=np.float64)
    return float(np.nanmean(cs)), float(np.nanstd(cs))


def main() -> int:
    if not PREFILL_CACHE.exists():
        print("MISSING_REGEN_INPUT", PREFILL_CACHE, file=sys.stderr); return 2

    pblob = np.load(PREFILL_CACHE)
    X_tbg = pblob["prefill"].astype(np.float64)
    y = pblob["correct"].astype(bool)
    assert X_tbg.shape == (500, 1536) and y.shape == (500,)

    slt = load_slt()
    if slt is None:
        print("MISSING_REGEN_INPUT", "no SLT (final-token) activation cache found",
              file=sys.stderr)
        for c in SLT_CACHE_CANDIDATES:
            print("  tried:", c, file=sys.stderr)
        return 2
    X_slt, slt_path, slt_key = slt

    folds = stratified_kfold(y, N_FOLDS, SEED)

    # Full-data directions (primary cosine, matches F-3's reported number).
    d_tbg = dom_direction(X_tbg, y)
    d_slt = dom_direction(X_slt, y)
    cos_dom = cosine(d_tbg, d_slt)

    w_tbg = lr_direction(X_tbg, y, LR_C)
    w_slt = lr_direction(X_slt, y, LR_C)
    cos_lr = cosine(w_tbg, w_slt)

    # Per-fold stability of the cosine.
    cos_dom_mean, cos_dom_std = fold_mean_cosine(
        X_tbg, X_slt, y, folds, dom_direction)
    cos_lr_mean, cos_lr_std = fold_mean_cosine(
        X_tbg, X_slt, y, folds, lambda X, yy: lr_direction(X, yy, LR_C))

    auroc_tbg = oof_auroc(X_tbg, y, folds)
    auroc_slt = oof_auroc(X_slt, y, folds)

    # Decision: cos > 0.5 (either probe family) refutes the F-3 seam.
    refutes_f3 = bool((cos_dom > COS_THRESHOLD) or (cos_lr > COS_THRESHOLD))

    out = {
        "experiment": "P11-FE273",
        "description": "TBG vs SLT correctness-direction cosine sanity check",
        "n": int(len(y)),
        "slt_cache": slt_path,
        "slt_key": slt_key,
        "cos_threshold": COS_THRESHOLD,
        "lr_C": LR_C,
        "cos_dom_fulldata": cos_dom,
        "cos_lr_fulldata": cos_lr,
        "cos_dom_fold_mean": cos_dom_mean,
        "cos_dom_fold_std": cos_dom_std,
        "cos_lr_fold_mean": cos_lr_mean,
        "cos_lr_fold_std": cos_lr_std,
        "auroc_tbg_oof": auroc_tbg,
        "auroc_slt_oof": auroc_slt,
        "f3_reference_cos": 0.046,
        "refutes_f3_orthogonality": refutes_f3,
        "verdict": (
            "REFUTES F-3 seam: TBG/SLT directions overlap (cos > 0.5)"
            if refutes_f3 else
            "F-3 seam survives: TBG/SLT correctness directions are near-orthogonal"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())