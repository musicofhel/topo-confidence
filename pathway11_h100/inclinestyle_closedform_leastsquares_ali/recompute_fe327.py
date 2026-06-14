"""P11-FE327 — INCLINE-style closed-form alignment of the prefill subspace.

INCLINE (Hu et al.) shows that a closed-form least-squares map
    W = (X_s^T X_s)^{-1} X_s^T X_t
applied at the last prompt token causally shifts model behaviour from a
"source" distribution toward a "target" distribution. Here the source is the
set of K=1-INCORRECT Qwen-2.5-1.5B last-token L19 prefill residuals and the
target is the K=1-CORRECT residuals, paired by random resampling.

We test whether F-3 (cos(prefill_DoM, final_DoM) = 0.046, i.e. the prefill and
final-token mass-mean directions are orthogonal) survives a *learned* linear
rotation of the prefill subspace:

  1. cos((W @ prefill_DoM), final_DoM)  vs  cos(prefill_DoM, final_DoM).
     If the rotated prefill DoM aligns with final_DoM, F-3 is a coordinate
     artefact of raw mass-mean DoM rather than a functional separation.
  2. Held-out (5-fold OOF) ΔAUROC of the rotated readout
        (W h) · prefill_DoM
     against the raw readout
        h · prefill_DoM.
     A positive ΔAUROC that co-occurs with rising cos would corroborate the
     "coordinate artefact" reading.

Convention: W is fitted under the row convention X_s @ W ≈ X_t, so the
column-notation "W @ v" used in the brief is computed as the row product v @ W.

final_DoM requires a final-token L19 cache; if none is found the rotation/cos
geometry is reported as null and only the prefill-only ΔAUROC test runs.
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
OUT_JSON = ROOT / "pathway11_h100/incline_alignment/results.json"

# Candidate final-token L19 caches (graceful: absence degrades, never fails).
FINAL_CACHE_CANDIDATES = [
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz",
     ["final", "final_hidden", "final_token", "hidden_final", "prefill"]),
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_finaltoken.npz",
     ["final", "final_hidden", "final_token", "hidden_final"]),
    (ROOT / "pathway10/cache/m15b_final.npz",
     ["final", "final_hidden", "final_token", "hidden_final"]),
]

N_FOLDS = 5
SEED = 9999
RIDGE_REL = 1e-2  # relative to trace(G)/d


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


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def mass_mean_dom(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mass-mean DoM direction: mean(correct) - mean(incorrect)."""
    return X[y].mean(axis=0) - X[~y].mean(axis=0)


def fit_incline_W(Xs: np.ndarray, Xt: np.ndarray, ridge_rel: float) -> np.ndarray:
    """Closed-form W minimizing ||Xs @ W - Xt||_F with ridge stabilization."""
    d = Xs.shape[1]
    G = Xs.T @ Xs
    lam = ridge_rel * float(np.trace(G)) / d
    G = G + lam * np.eye(d, dtype=Xs.dtype)
    return np.linalg.solve(G, Xs.T @ Xt)  # (d, d), X @ W approx target


def make_pairs(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Pair incorrect (source) with correct (target) by random resampling."""
    inc = X[~y]; cor = X[y]
    if len(inc) == 0 or len(cor) == 0:
        return None, None
    n = max(len(inc), len(cor))
    si = rng.integers(0, len(inc), size=n)
    sc = rng.integers(0, len(cor), size=n)
    return inc[si], cor[sc]


def find_final_cache():
    for path, keys in FINAL_CACHE_CANDIDATES:
        if not path.exists():
            continue
        blob = np.load(path)
        for key in keys:
            if key in blob.files:
                arr = blob[key]
                if arr.ndim == 2 and arr.shape == (500, 1536):
                    return arr.astype(np.float64), f"{path.name}:{key}"
    return None, None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    prefill_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)

    # --- Global geometry: cos test (needs final-token cache) ---
    d_pre_global = mass_mean_dom(X, y)
    # Sanity: reproduce the supplied prefill DoM scores via mass-mean projection.
    proj = X @ d_pre_global
    dom_score_corr = float(np.corrcoef(proj, prefill_score)[0, 1])

    rng = np.random.default_rng(SEED)
    Xs, Xt = make_pairs(X, y, rng)
    W_global = fit_incline_W(Xs, Xt, RIDGE_REL)
    d_pre_rot_global = d_pre_global @ W_global  # row-convention "W @ prefill_DoM"

    X_final, final_source = find_final_cache()
    if X_final is not None:
        final_DoM = mass_mean_dom(X_final, y)
        cos_baseline = cos(d_pre_global, final_DoM)
        cos_rotated = cos(d_pre_rot_global, final_DoM)
    else:
        final_DoM = None
        cos_baseline = None
        cos_rotated = None

    # --- Held-out ΔAUROC: rotated vs raw prefill-DoM readout (5-fold OOF) ---
    n = len(y)
    raw_oof = np.zeros(n, dtype=np.float64)
    rot_oof = np.zeros(n, dtype=np.float64)
    folds = stratified_kfold(y, N_FOLDS, SEED)
    fold_rng = np.random.default_rng(SEED + 1)

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]
        d_pre = mass_mean_dom(Xtr, ytr)
        raw_oof[test_idx] = Xte @ d_pre
        Xs_tr, Xt_tr = make_pairs(Xtr, ytr, fold_rng)
        W = fit_incline_W(Xs_tr, Xt_tr, RIDGE_REL)
        # rotated readout: (W h) . prefill_DoM  ==  (Xte @ W) @ d_pre
        rot_oof[test_idx] = (Xte @ W) @ d_pre

    auroc_raw = auroc(raw_oof, y)
    auroc_rot = auroc(rot_oof, y)
    delta_auroc = auroc_rot - auroc_raw

    out = {
        "experiment": "P11-FE327",
        "description": "INCLINE closed-form prefill-subspace alignment vs F-3 orthogonality",
        "n": int(n),
        "n_correct": int(y.sum()),
        "n_incorrect": int((~y).sum()),
        "ridge_rel": RIDGE_REL,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "dom_score_recovery_corr": dom_score_corr,
        "final_cache_source": final_source,
        "final_dom_available": final_DoM is not None,
        "cos_prefillDoM_finalDoM_baseline": cos_baseline,
        "cos_W_prefillDoM_finalDoM_rotated": cos_rotated,
        "cos_delta": (None if cos_baseline is None
                      else float(cos_rotated - cos_baseline)),
        "auroc_raw_oof": float(auroc_raw),
        "auroc_rotated_oof": float(auroc_rot),
        "delta_auroc": float(delta_auroc),
        "f3_survives": (None if cos_rotated is None
                        else bool(abs(cos_rotated) < 0.2 and delta_auroc < 0.01)),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())