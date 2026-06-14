"""P11-FE737 — Sun et al. late-step trajectory features vs single-layer L19 DoM.

Refutation test for F-9 (CoE-60 redundant with single-layer DoM). Sun et al.
report a ~0.06 AUC margin between a late-step *trajectory* feature and a
final-marker feature (0.87 vs 0.81) along a step-difference axis we have not
tested. This script reconstructs their feature on the Qwen-1.5B MATH-500
cached L19 activations:

    feat = h(L19, answer-marker)  ⊕  ( h(L19, step_{N-1}) - h(L19, step_N) )

then PCA (dim=128) + out-of-fold logistic regression, and compares the OOF
AUROC against (a) the answer-marker-only feature and (b) the canonical L19
prefill DoM (0.7731). A genuine refutation of F-9 requires the trajectory
feature to beat L19 prefill DoM by >0.04 AUC on the same labels.

The step-difference axis requires a per-step trajectory cache of L19 hidden
states (the prefill answer-marker plus the last two reasoning-step markers).
That is a distinct extraction from the single-vector prefill cache; if it is
absent the script reports MISSING_REGEN_INPUT and exits 2 rather than faking
the trajectory axis with the prefill vector alone.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
# Per-step L19 trajectory cache (answer-marker + last two step markers).
# Distinct from the single-vector prefill cache; required for the step axis.
TRAJ_NPZ = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_step_trajectory.npz"
OUT_JSON = ROOT / "pathway11_h100/sun_trajectory/results.json"

SEED = 9999
N_FOLDS = 5
PCA_DIM = 128
DOM_BASELINE = 0.7731  # canonical L19 prefill DoM OOF AUROC (1024-tok labels)
REFUTE_MARGIN = 0.04   # trajectory must beat DoM by this to refute F-9


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def oof_logistic_auroc(X: np.ndarray, y: np.ndarray, pca_dim: int) -> float:
    """PCA(pca_dim) + logistic regression, fit per fold, scored out-of-fold."""
    n = len(y)
    oof = np.zeros(n, dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr = y[train_idx]
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0) + 1e-8
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd
        dim = min(pca_dim, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
        pca = PCA(n_components=dim, random_state=SEED)
        Ztr = pca.fit_transform(Xtr_s)
        Zte = pca.transform(Xte_s)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Ztr, ytr)
        oof[test_idx] = clf.predict_proba(Zte)[:, 1]
    return auroc(oof, y)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    prefill = cache["prefill"].astype(np.float64)
    correct = cache["correct"].astype(bool)
    assert prefill.shape == (500, 1536) and correct.shape == (500,)

    # The step-difference axis is the whole point of the Sun et al. feature;
    # without the per-step trajectory cache the experiment cannot run honestly.
    if not TRAJ_NPZ.exists():
        print("MISSING_REGEN_INPUT", TRAJ_NPZ, file=sys.stderr)
        return 2

    traj = np.load(TRAJ_NPZ)
    required = ("h_answer_marker", "h_step_nm1", "h_step_n")
    missing = [k for k in required if k not in traj.files]
    if missing:
        print("MISSING_REGEN_INPUT", TRAJ_NPZ, "keys=" + ",".join(missing), file=sys.stderr)
        return 2

    h_marker = traj["h_answer_marker"].astype(np.float64)
    h_nm1 = traj["h_step_nm1"].astype(np.float64)
    h_n = traj["h_step_n"].astype(np.float64)
    for name, arr in (("h_answer_marker", h_marker), ("h_step_nm1", h_nm1), ("h_step_n", h_n)):
        if arr.shape != (500, 1536):
            print("MISSING_REGEN_INPUT", TRAJ_NPZ, f"bad_shape:{name}:{arr.shape}", file=sys.stderr)
            return 2

    step_diff = h_nm1 - h_n
    feat_traj = np.concatenate([h_marker, step_diff], axis=1)  # (500, 3072)

    auroc_traj = oof_logistic_auroc(feat_traj, correct, PCA_DIM)
    auroc_marker = oof_logistic_auroc(h_marker, correct, PCA_DIM)
    auroc_stepdiff = oof_logistic_auroc(step_diff, correct, PCA_DIM)

    # DoM baseline read straight from cached scores (no refit).
    dom_baseline_measured = float("nan")
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        if dom_score.shape == (500,):
            dom_baseline_measured = auroc(dom_score, correct)

    margin_vs_dom = auroc_traj - DOM_BASELINE
    refutes_f9 = bool(np.isfinite(auroc_traj) and margin_vs_dom > REFUTE_MARGIN)

    out = {
        "experiment": "P11-FE737",
        "n": int(len(correct)),
        "pca_dim": PCA_DIM,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "auroc_trajectory_oof": auroc_traj,
        "auroc_answer_marker_oof": auroc_marker,
        "auroc_step_diff_oof": auroc_stepdiff,
        "dom_baseline_canonical": DOM_BASELINE,
        "dom_baseline_measured": dom_baseline_measured,
        "margin_trajectory_vs_dom": float(margin_vs_dom),
        "refute_margin_threshold": REFUTE_MARGIN,
        "refutes_F9": refutes_f9,
        "verdict": (
            "REFUTES_F9: step-wise trajectory feature beats L19 prefill DoM by "
            f">{REFUTE_MARGIN} AUC" if refutes_f9 else
            "F9_HOLDS: trajectory feature does not beat L19 prefill DoM by margin"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())