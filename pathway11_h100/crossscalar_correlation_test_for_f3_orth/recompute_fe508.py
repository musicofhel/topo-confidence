"""P11-FE508 — Cross-scalar correlation test for F-3 orthogonality.

F-3 rests on cos(prefill_DoM, final_DoM) = 0.046: the supervised probe
*directions* at L19 prefill and L19 final-token are near-orthogonal, which is
read as "two orthogonal correctness signals." This experiment asks whether the
*scalars* those probes emit are nevertheless the same underlying signal.

Two tests across the 500 MATH-500 problems:
  (1) Per-problem Pearson r between the OOF DoM scalar at prefill and at
      final-token. Orthogonal directions can still produce strongly correlated
      projections.
  (2) Sliding-window batch-dispersion correlation. Sort problems by per-problem
      accuracy (mean correct over K=8 self-consistency samples), slide a
      100-problem window, and within each window measure the dispersion (std)
      of the prefill scalar and of the final scalar. Correlate the two
      dispersion series across windows.

If either correlation exceeds r > 0.5, F-3's "two orthogonal signals" framing
collapses to "one shared scalar plus probe-fit noise" — the orthogonality is a
probe artifact, not two independent readouts.
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
from scipy.stats import pearsonr

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
OUT_JSON = ROOT / "pathway11_h100/cross_scalar_corr/results.json"

# Final-token L19 activations are not in the canonical prefill cache; probe the
# likely locations and key names produced by the pathway10/11 final-token dumps.
FINAL_CACHE_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final_token.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_finaltoken.npz",
    ROOT / "scratch/m15b_final.npz",
    ROOT / "scratch/pathway10_final_l19.npz",
]
FINAL_KEY_CANDIDATES = ["final", "final_token", "finaltoken", "final_hidden", "hidden", "l19_final"]

N_FOLDS = 5
SEED = 9999
WINDOW = 100
R_THRESHOLD = 0.5


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


def oof_dom_scalar(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    """Out-of-fold DoM projection: fit (mean_pos - mean_neg) on train, project test."""
    s = np.zeros(len(y), dtype=np.float64)
    for te in folds:
        tr = np.ones(len(y), dtype=bool); tr[te] = False
        Xtr, ytr = X[tr], y[tr]
        if ytr.all() or (~ytr).all():
            d = Xtr.mean(axis=0)
        else:
            d = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        s[te] = X[te] @ d
    return s


def full_dom_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y].mean(axis=0) - X[~y].mean(axis=0)


def load_final_cache() -> tuple[np.ndarray | None, str | None, str | None]:
    for path in FINAL_CACHE_CANDIDATES:
        if not path.exists():
            continue
        blob = np.load(path)
        for key in FINAL_KEY_CANDIDATES:
            if key in blob.files:
                arr = blob[key]
                if arr.ndim == 2 and arr.shape == (500, 1536):
                    return arr.astype(np.float64), str(path), key
        # last resort: any (500, 1536) array in the file
        for key in blob.files:
            arr = blob[key]
            if getattr(arr, "ndim", 0) == 2 and arr.shape == (500, 1536):
                return arr.astype(np.float64), str(path), key
    return None, None, None


def per_problem_accuracy() -> np.ndarray | None:
    if not K8_DIR.exists():
        return None
    accs = np.zeros(500, dtype=np.float64)
    for i in range(500):
        f = K8_DIR / f"problem_{i:03d}.npz"
        if not f.exists():
            return None
        c = np.load(f)["correct"].astype(bool)
        accs[i] = float(c.mean())
    return accs


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(a @ b / (na * nb))


def windowed_dispersion_corr(order: np.ndarray, s_pre: np.ndarray,
                             s_fin: np.ndarray, window: int) -> dict:
    n = len(order)
    disp_pre, disp_fin = [], []
    for start in range(0, n - window + 1):
        idx = order[start:start + window]
        disp_pre.append(float(s_pre[idx].std()))
        disp_fin.append(float(s_fin[idx].std()))
    disp_pre = np.asarray(disp_pre); disp_fin = np.asarray(disp_fin)
    if len(disp_pre) < 3 or disp_pre.std() < 1e-12 or disp_fin.std() < 1e-12:
        return {"n_windows": int(len(disp_pre)), "pearson_r": float("nan"),
                "p_value": float("nan")}
    r, p = pearsonr(disp_pre, disp_fin)
    return {"n_windows": int(len(disp_pre)), "pearson_r": float(r),
            "p_value": float(p)}


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr); return 2

    blob = np.load(CACHE)
    Xpre = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert Xpre.shape == (500, 1536) and y.shape == (500,)

    Xfin, final_path, final_key = load_final_cache()
    if Xfin is None:
        print("MISSING_REGEN_INPUT", "final-token L19 cache "
              f"(tried {[str(p) for p in FINAL_CACHE_CANDIDATES]})", file=sys.stderr)
        return 2

    canonical_prefill = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)

    folds = stratified_kfold(y, N_FOLDS, SEED)
    s_pre = oof_dom_scalar(Xpre, y, folds)
    s_fin = oof_dom_scalar(Xfin, y, folds)

    # Direction-level orthogonality cross-check (the F-3 anchor, ~0.046).
    d_pre = full_dom_direction(Xpre, y)
    d_fin = full_dom_direction(Xfin, y)
    cos_dirs = cosine(d_pre, d_fin)

    # Test 1 — per-problem scalar correlation.
    r_scalar, p_scalar = pearsonr(s_pre, s_fin)
    r_canon, _ = pearsonr(canonical_prefill, s_fin)

    # Test 2 — sliding-window batch-dispersion correlation.
    accs = per_problem_accuracy()
    if accs is not None:
        acc_source = "k8_selfconsistency_mean"
    else:
        accs = y.astype(np.float64)
        acc_source = "binary_correct_fallback"
    # Stable sort by accuracy; tie-break by canonical prefill scalar so windows
    # of equal accuracy are still deterministically ordered.
    order = np.lexsort((canonical_prefill, accs))
    disp_corr = windowed_dispersion_corr(order, s_pre, s_fin, WINDOW)

    rs = [abs(r_scalar), abs(r_canon)]
    if not np.isnan(disp_corr["pearson_r"]):
        rs.append(abs(disp_corr["pearson_r"]))
    max_abs_r = float(max(rs))
    f3_collapses = bool(max_abs_r > R_THRESHOLD)

    out = {
        "experiment": "P11-FE508",
        "description": "Cross-scalar correlation test for F-3 orthogonality",
        "final_cache_used": final_path,
        "final_cache_key": final_key,
        "accuracy_source": acc_source,
        "window": WINDOW,
        "r_threshold": R_THRESHOLD,
        "n_folds": N_FOLDS,
        "cos_prefill_final_dom_directions": cos_dirs,
        "auroc_oof_prefill_scalar": float(auroc(s_pre, y)),
        "auroc_oof_final_scalar": float(auroc(s_fin, y)),
        "test1_per_problem_scalar": {
            "pearson_r_oof_prefill_vs_oof_final": float(r_scalar),
            "p_value": float(p_scalar),
            "pearson_r_canonical_prefill_vs_oof_final": float(r_canon),
        },
        "test2_windowed_dispersion": disp_corr,
        "max_abs_r": max_abs_r,
        "f3_orthogonality_collapses": f3_collapses,
        "interpretation": (
            "COLLAPSE: scalars correlate > 0.5 despite orthogonal directions; "
            "F-3 'two orthogonal signals' is a probe artifact."
            if f3_collapses else
            "SURVIVES: scalars are weakly correlated; orthogonal directions "
            "carry genuinely distinct correctness signal."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())