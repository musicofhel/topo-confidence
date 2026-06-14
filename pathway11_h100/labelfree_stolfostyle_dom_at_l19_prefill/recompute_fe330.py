"""P11-FE330 — Label-free Stolfo-style DoM at L19 prefill.

Stolfo et al. derive a behavior-controlling direction without any logistic
probe: form a paired contrast set of correct vs incorrect traces, take the mean
activation difference at the target token, and normalize to a unit direction.
Here we build v_19 = mean(h(correct) - h(incorrect)) over the L19 prefill
last-input-token hidden states, normalize to u_19, and score held-out problems
by <h, u_19>. We report a single 400-train / 100-held-out split (the brief's
"held-out 100" evaluation) plus a 5-fold OOF AUROC for a stable estimate, and
compare against the supervised L19 DoM AUROC of 0.7731 (F-2).

The "label-free" framing: no logistic regression, no covariance whitening — the
direction is the raw normalized class-mean contrast, the cheapest possible probe.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np


ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/labelfree_dom/results.json"

SUPERVISED_DOM_AUROC = 0.7731  # F-2 supervised L19 DoM (OOF 5-fold)
N_HELDOUT = 100
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


def labelfree_dom(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Normalized class-mean contrast direction u_19 (label-free / no probe)."""
    mu_correct = X_train[y_train].mean(axis=0)
    mu_incorrect = X_train[~y_train].mean(axis=0)
    v = mu_correct - mu_incorrect
    nrm = float(np.linalg.norm(v))
    if nrm < 1e-12:
        return v
    return v / nrm


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    if X.shape != (500, 1536) or y.shape != (500,):
        print("UNEXPECTED_SHAPE", X.shape, y.shape, file=sys.stderr); return 3

    n = len(y)

    # --- Single 400-train / 100-held-out stratified split ---
    rng = np.random.default_rng(SEED)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    frac_held = N_HELDOUT / n
    n_pos_held = int(round(len(pos) * frac_held))
    n_neg_held = N_HELDOUT - n_pos_held
    held_idx = np.concatenate([pos[:n_pos_held], neg[:n_neg_held]])
    train_mask = np.ones(n, dtype=bool); train_mask[held_idx] = False

    u_split = labelfree_dom(X[train_mask], y[train_mask])
    held_scores = X[held_idx] @ u_split
    auroc_held = auroc(held_scores, y[held_idx])

    # --- 5-fold OOF estimate (stable, comparable to supervised OOF 0.7731) ---
    folds = stratified_kfold(y, N_FOLDS, SEED)
    oof_scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        tr = np.ones(n, dtype=bool); tr[test_idx] = False
        u = labelfree_dom(X[tr], y[tr])
        oof_scores[test_idx] = X[test_idx] @ u
    auroc_oof = auroc(oof_scores, y)

    # --- In-sample (all-data) direction, for reference / cos with OOF dirs ---
    u_full = labelfree_dom(X, y)
    auroc_insample = auroc(X @ u_full, y)

    delta_vs_supervised = float(auroc_oof - SUPERVISED_DOM_AUROC)
    matches_within_001 = bool(abs(delta_vs_supervised) <= 0.01)

    out = {
        "experiment": "P11-FE330",
        "n_problems": int(n),
        "n_heldout": int(len(held_idx)),
        "n_pos_heldout": int(n_pos_held),
        "n_neg_heldout": int(n_neg_held),
        "auroc_heldout_100": float(auroc_held),
        "auroc_oof_5fold": float(auroc_oof),
        "auroc_insample": float(auroc_insample),
        "supervised_dom_auroc_ref": SUPERVISED_DOM_AUROC,
        "delta_oof_vs_supervised": delta_vs_supervised,
        "matches_supervised_within_0.01": matches_within_001,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())