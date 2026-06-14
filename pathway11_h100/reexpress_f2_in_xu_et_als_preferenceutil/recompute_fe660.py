"""P11-FE660 — F-2 re-expressed in Xu et al.'s preference-utility log-odds framework.

Xu et al. decompose a model's choice between a polarity-paired correct completion
A_p and a plausible-distractor completion A_n into a preference component (concept
expression, direction alignment) and a utility component (output coherence,
magnitude) on a shared log-odds scale: PrefOdds = L_n - L_p (their Eq. 7).

We operationalize this CPU-only on cached Qwen-2.5-1.5B L19 prefill activations.
The supervised DoM contrast d = mean(correct) - mean(incorrect) is the population-
level polarity axis (the canonical A_p vs A_n direction). For each problem the
signed projection onto unit-d is the full PrefOdds; we factor it into:

  * PrefOdds (preference)  = cos(prefill, d)   -- pure directional/concept axis
  * UtilOdds  (utility)    = log||prefill||     -- magnitude/coherence axis

and verify (a) PrefOdds is monotone-positive in correctness rank, and (b) whether
F-2's signal lives on the preference axis (large PrefOdds gap, negligible UtilOdds
shift) or is utility-confounded (both shift together). The direction is fit OOF
over 5 folds to avoid leakage; we also AUROC the preference axis after residualizing
out the utility axis, and vice-versa.
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
from scipy.stats import spearmanr

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/pref_utility_logodds/results.json"

N_FOLDS = 5
N_BINS = 10
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


def residualize_oof(x: np.ndarray, z: np.ndarray, y: np.ndarray,
                    folds: list[np.ndarray]) -> np.ndarray:
    """OOF residual of x on z (+intercept): x - (a*z + b), fit on train folds."""
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        A = np.column_stack([z[train_mask], np.ones(train_mask.sum())])
        coeffs, _, _, _ = np.linalg.lstsq(A, x[train_mask], rcond=None)
        out[test_idx] = x[test_idx] - (z[test_idx] * coeffs[0] + coeffs[1])
    return out


def monotone_profile(score: np.ndarray, y: np.ndarray, n_bins: int) -> dict:
    """Bin by score rank, mean correctness per bin, monotonicity diagnostics."""
    order = np.argsort(score, kind="stable")
    bins = np.array_split(order, n_bins)
    bin_acc = np.array([float(y[b].mean()) for b in bins], dtype=np.float64)
    bin_mid = np.array([float(score[b].mean()) for b in bins], dtype=np.float64)
    diffs = np.diff(bin_acc)
    inversions = int((diffs < 0).sum())
    rho, _ = spearmanr(np.arange(n_bins), bin_acc)
    return {
        "bin_mean_score": bin_mid.tolist(),
        "bin_acc": bin_acc.tolist(),
        "n_inversions": inversions,
        "monotone_nondecreasing": bool(inversions == 0),
        "spearman_binidx_vs_acc": float(rho),
    }


def cohen_d(score: np.ndarray, y: np.ndarray) -> float:
    pos = score[y]; neg = score[~y]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    nv = len(pos) + len(neg) - 2
    sp = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) +
                  (len(neg) - 1) * neg.var(ddof=1)) / nv)
    if sp < 1e-12:
        return float("nan")
    return float((pos.mean() - neg.mean()) / sp)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,), f"bad shapes {X.shape} {y.shape}"
    n = len(y)

    folds = stratified_kfold(y, N_FOLDS, SEED)

    # OOF decomposition: PrefOdds (full projection), preference axis (cosine),
    # utility axis (log-norm). Direction fit on train folds only.
    prefodds_full = np.zeros(n, dtype=np.float64)   # X . d_unit  (Eq. 7 analogue)
    pref_axis = np.zeros(n, dtype=np.float64)        # cos(X, d)   -- concept expression
    util_axis = np.log(np.linalg.norm(X, axis=1) + 1e-12)  # log||X|| -- coherence

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr = X[train_mask]; ytr = y[train_mask]
        d = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        d_unit = d / (np.linalg.norm(d) + 1e-12)
        Xte = X[test_idx]
        proj = Xte @ d_unit
        prefodds_full[test_idx] = proj
        pref_axis[test_idx] = proj / (np.linalg.norm(Xte, axis=1) + 1e-12)

    # Disambiguate confounding: AUROC of each axis alone and after residualizing
    # out the other (OOF), so a shared shift cannot masquerade as separable signal.
    pref_resid_util = residualize_oof(pref_axis, util_axis, y, folds)
    util_resid_pref = residualize_oof(util_axis, pref_axis, y, folds)

    # Reference DoM score (existing F-2 readout) if available.
    dom_auroc = None
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        dom_auroc = float(auroc(dom_score, y))

    out = {
        "experiment": "P11-FE660",
        "framework": "Xu et al. preference-utility log-odds (Eq. 7)",
        "n": int(n),
        "n_correct": int(y.sum()),
        "auroc": {
            "prefodds_full_oof": float(auroc(prefodds_full, y)),
            "preference_axis_cos_oof": float(auroc(pref_axis, y)),
            "utility_axis_lognorm": float(auroc(util_axis, y)),
            "preference_resid_utility_oof": float(auroc(pref_resid_util, y)),
            "utility_resid_preference_oof": float(auroc(util_resid_pref, y)),
            "reference_dom_score": dom_auroc,
        },
        # Standardized correct-vs-incorrect gaps on the shared scale: a large
        # PrefOdds gap with a negligible UtilOdds gap => F-2 lives on preference.
        "cohen_d": {
            "preference_axis_cos": cohen_d(pref_axis, y),
            "utility_axis_lognorm": cohen_d(util_axis, y),
        },
        "preference_minus_utility_d": float(
            cohen_d(pref_axis, y) - cohen_d(util_axis, y)
        ),
        "utility_confounded": bool(
            abs(cohen_d(util_axis, y)) > 0.5 * abs(cohen_d(pref_axis, y))
        ),
        # Monotone-positive in correctness rank.
        "monotone_prefodds_full": monotone_profile(prefodds_full, y, N_BINS),
        "monotone_preference_axis": monotone_profile(pref_axis, y, N_BINS),
        "spearman_prefodds_full_vs_correct": float(spearmanr(prefodds_full, y.astype(float))[0]),
        "config": {"n_folds": N_FOLDS, "n_bins": N_BINS, "seed": SEED},
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print("OK", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())