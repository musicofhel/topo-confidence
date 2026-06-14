"""FE331 — Dynamic-c steering coefficient for prefill_DoM (H-1 gate).

Derives the Stolfo et al. (Eq. 2) steering magnitude c instead of an arbitrary
alpha-sweep over {-4,-2,-1,0,1,2,4}. With the L19 prefill DoM direction
u_19 = (mu_correct - mu_incorrect) / ||.|| (unit norm), the principled
steering coefficient for an activation x is the value that maps x's projection
onto u_19 to the in-distribution mean projection of the *correct* class:

    c_i = target_proj - (x_i . u_19),   where (x_i + c_i u_19) . u_19 = target_proj
    target_proj = mean_{correct} (x . u_19)

We report the per-class distribution of c (the magnitude H-1 should actually
steer with), both on full data and out-of-fold (u_19 + target fit on train,
c evaluated on held-out test), so H-1 can replace the arbitrary multiplier with
a data-derived magnitude before spending any H100-days. The mean incorrect-class
c is the headline number — it is the residual gap the correct mean sits above.
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
OUT_JSON = ROOT / "pathway11_h100/dynamic_c_steering/results.json"

N_FOLDS = 5
SEED = 9999


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def dom_unit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unit-norm L19 DoM direction u_19 = (mu_correct - mu_incorrect)/||.||."""
    d = X[y].mean(axis=0) - X[~y].mean(axis=0)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-12:
        return d
    return d / nrm


def c_stats(c: np.ndarray, y: np.ndarray) -> dict:
    """Per-class summary of the dynamic-c coefficient distribution."""
    c_pos = c[y]
    c_neg = c[~y]
    return {
        "mean_all": float(c.mean()),
        "mean_correct": float(c_pos.mean()),
        "mean_incorrect": float(c_neg.mean()),
        "median_incorrect": float(np.median(c_neg)),
        "std_incorrect": float(c_neg.std(ddof=1)) if len(c_neg) > 1 else float("nan"),
        "p10_incorrect": float(np.percentile(c_neg, 10)),
        "p90_incorrect": float(np.percentile(c_neg, 90)),
        "frac_incorrect_positive": float((c_neg > 0).mean()),
    }


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)
    n = len(y)

    # --- Full-data c (Stolfo Eq. 2) -------------------------------------
    u = dom_unit(X, y)
    proj = X @ u                       # current projection of each residual
    target = float(proj[y].mean())     # in-distribution mean of correct class
    c_full = target - proj             # coefficient that maps proj -> target

    # Sanity: u recovers the DoM AUROC, and steered projections hit target.
    auroc_u = auroc(proj, y)
    steered = proj + c_full            # all collapse to `target` by construction
    max_residual = float(np.max(np.abs(steered - target)))

    # Cross-check against the cached canonical DoM scores (sign/orientation).
    dom_cached = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    cos_proj_cached = float(
        np.dot(proj - proj.mean(), dom_cached - dom_cached.mean())
        / (np.linalg.norm(proj - proj.mean()) * np.linalg.norm(dom_cached - dom_cached.mean()) + 1e-12)
    )

    # --- Out-of-fold c (u + target fit on train, c on held-out test) ----
    c_oof = np.zeros(n, dtype=np.float64)
    folds = stratified_kfold(y, N_FOLDS, SEED)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        u_tr = dom_unit(X[train_mask], y[train_mask])
        proj_tr = X[train_mask] @ u_tr
        target_tr = float(proj_tr[y[train_mask]].mean())
        c_oof[test_idx] = target_tr - (X[test_idx] @ u_tr)

    out = {
        "experiment": "FE331",
        "n": n,
        "n_correct": int(y.sum()),
        "n_incorrect": int((~y).sum()),
        "u19_norm_check": float(np.linalg.norm(u)),
        "target_proj_correct_mean": target,
        "auroc_u19_projection": float(auroc_u),
        "cos_proj_vs_cached_dom": cos_proj_cached,
        "steered_max_abs_residual_to_target": max_residual,
        "c_full_data": c_stats(c_full, y),
        "c_out_of_fold": c_stats(c_oof, y),
        "recommended_steer_magnitude": float(c_full[~y].mean()),
        "note": (
            "Stolfo Eq. 2 dynamic-c: steer incorrect-class residuals by "
            "c=target-proj toward the correct-class mean projection on u_19. "
            "recommended_steer_magnitude is the mean incorrect-class c — use "
            "this in place of the H-1 arbitrary alpha-sweep {-4,-2,-1,0,1,2,4}."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())