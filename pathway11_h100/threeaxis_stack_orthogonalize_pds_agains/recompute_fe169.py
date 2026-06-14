"""P11-FE169 — Three-axis stack: residualized PDS over L19 DoM.

Tests whether the Pairwise Discrepancy Score (PDS, cross-chain text-NLI
agreement; the column produced by FE75) carries correctness signal that is
independent of the L19 DoM hidden-state direction.

Procedure (5-fold OOF, no leakage):
  1. Within each train fold, OLS-residualize PDS against DoM
     (PDS ~ a*DoM + b), apply the train coefficients to the test fold.
  2. Fit a logistic regressor for correctness on {DoM, residualized_PDS};
     score the held-out fold.
  3. Fit a DoM-alone logistic baseline the same way.
  4. Report ΔAUROC = AUROC(stack) - AUROC(DoM-alone).

Interpretation: if residualized PDS adds >= 0.02 AUROC, F-9's "CoE-60 is
redundant with single-layer DoM" claim is too narrow — PDS is a third
independent modality (text NLI agreement) rather than redundant geometry.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
# PDS column emitted by FE75 (cross-chain NLI agreement). Candidate locations
# are probed in order; the first that exists and carries a length-500 vector
# wins. If none exist, the experiment is not runnable yet.
PDS_CANDIDATES = [
    (ROOT / "pathway11_h100/results/fe75_pds.npz", "pds"),
    (ROOT / "pathway11_h100/pds_nli/results.npz", "pds"),
    (ROOT / "pathway11_h100/pds_nli/fe75_pds.npz", "pds"),
]
PDS_JSON_CANDIDATES = [
    ROOT / "pathway11_h100/results/fe75_pds.json",
    ROOT / "pathway11_h100/pds_nli/results.json",
]
OUT_JSON = ROOT / "pathway11_h100/pds_orthogonal_stack/results.json"

SEED = 9999
N_FOLDS = 5
DELTA_THRESHOLD = 0.02


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def _load_pds() -> np.ndarray | None:
    """Locate and load the FE75 PDS column as a length-500 float vector."""
    for path, key in PDS_CANDIDATES:
        if path.exists():
            blob = np.load(path)
            if key in blob:
                arr = np.asarray(blob[key], dtype=np.float64).ravel()
                if arr.shape == (500,):
                    return arr
            # fall back to the first 500-length array in the archive
            for k in blob.files:
                arr = np.asarray(blob[k], dtype=np.float64).ravel()
                if arr.shape == (500,):
                    return arr
    for path in PDS_JSON_CANDIDATES:
        if path.exists():
            obj = json.loads(path.read_text())
            for key in ("pds", "pds_scores", "pds_column", "scores"):
                if key in obj:
                    arr = np.asarray(obj[key], dtype=np.float64).ravel()
                    if arr.shape == (500,):
                        return arr
    return None


def _oof_logit_scores(X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    """Out-of-fold logistic-regression positive-class probabilities."""
    n = len(y)
    oof = np.zeros(n, dtype=np.float64)
    for train_idx, test_idx in folds:
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[train_idx])
        oof[test_idx] = clf.predict_proba(Xte)[:, 1]
    return oof


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    correct = cache["correct"].astype(bool)
    assert correct.shape == (500,)

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64).ravel()
    assert dom_score.shape == (500,)

    pds = _load_pds()
    if pds is None:
        print(
            "MISSING_REGEN_INPUT FE75 PDS column "
            f"(checked {[str(p) for p, _ in PDS_CANDIDATES]} "
            f"and {[str(p) for p in PDS_JSON_CANDIDATES]})",
            file=sys.stderr,
        )
        return 2

    n = len(correct)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(n), correct))

    # Within-fold OLS residualization of PDS against DoM (no leakage).
    pds_resid = np.zeros(n, dtype=np.float64)
    for train_idx, test_idx in folds:
        A_tr = np.column_stack([dom_score[train_idx], np.ones(len(train_idx))])
        coeffs, _, _, _ = np.linalg.lstsq(A_tr, pds[train_idx], rcond=None)
        pred_te = dom_score[test_idx] * coeffs[0] + coeffs[1]
        pds_resid[test_idx] = pds[test_idx] - pred_te

    # OOF AUROCs.
    dom_alone_oof = _oof_logit_scores(dom_score[:, None], correct, folds)
    stack_oof = _oof_logit_scores(
        np.column_stack([dom_score, pds_resid]), correct, folds
    )

    auroc_dom_alone = auroc(dom_alone_oof, correct)
    auroc_stack = auroc(stack_oof, correct)
    delta = auroc_stack - auroc_dom_alone

    # Diagnostics: raw single-feature AUROCs + how much DoM explains PDS.
    auroc_pds_raw = auroc(pds, correct)
    auroc_pds_resid = auroc(pds_resid, correct)
    dom_pds_corr = float(np.corrcoef(dom_score, pds)[0, 1])

    out = {
        "experiment": "P11-FE169",
        "n": int(n),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "auroc_dom_alone_oof": float(auroc_dom_alone),
        "auroc_stack_dom_plus_residpds_oof": float(auroc_stack),
        "delta_auroc": float(delta),
        "delta_threshold": DELTA_THRESHOLD,
        "pds_adds_independent_signal": bool(delta >= DELTA_THRESHOLD),
        "auroc_pds_raw": float(auroc_pds_raw),
        "auroc_pds_residualized": float(auroc_pds_resid),
        "corr_dom_pds": dom_pds_corr,
        "interpretation": (
            "If delta_auroc >= 0.02, residualized PDS adds correctness signal "
            "independent of L19 DoM, so F-9's redundancy claim is too narrow "
            "and PDS is a third modality (text NLI agreement)."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())