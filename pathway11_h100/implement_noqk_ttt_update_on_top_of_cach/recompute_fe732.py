"""P11-FE732 — No-QK test-time-training (TTT) update on cached Qwen-1.5B prefill L19 DoM.

Implements an online "no-QK" TTT layer over the cached L19 prefill states: the
hidden model is a linear confidence probe W (no query/key projections, just a
direct value-reconstruction style update). W_0 is initialized from the fitted
DoM direction (mean_pos - mean_neg in standardized feature space, per train
fold), then updated by online Brier-loss gradient descent (eta=0.01) as the
problems stream past — the "reasoning trajectory" here being the sequence of
online updates.

For each of 5 stratified folds, W lives in lockstep with the other folds; at
every step t we (a) score the held-out test fold with the current W_t and pool
all 500 OOF scores to compute a leakage-free OOF AUROC, and (b) log
cos(W_0, W_t) averaged across folds. Trajectory index 0 is the static DoM
(no update), so trajectory[0] reproduces the static prefill DoM AUROC. We
compare the final/best step against the static prefill DoM (AUROC 0.7731).

Refutation logic: if TTT lifts OOF AUROC > 2 points over static, F-2's single
static-direction framing fails; if cos(W_0, W_T) drifts toward ~0.05 naturally,
F-3's orthogonal-circuits reading reduces to a rotating-probe artifact.
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
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/ttt_noqk/results.json"

SEED = 9999
N_FOLDS = 5
ETA = 0.01
STATIC_DOM_AUROC = 0.7731  # 1024tok prefill L19 DoM OOF reference


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    if X.shape != (500, 1536) or y.shape != (500,):
        print("MISSING_REGEN_INPUT bad-shape", X.shape, y.shape, file=sys.stderr)
        return 2

    n = len(y)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))
    rng = np.random.default_rng(SEED)

    # Per-fold state: standardized features, init DoM probe W0, online probe W, bias b.
    Xstd, W0, W, b, train_idx, test_idx, order = [], [], [], [], [], [], []
    for tr, te in folds:
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0)
        sd[sd < 1e-8] = 1.0
        Xs = (X - mu) / sd
        tr_pos = tr[y[tr]]
        tr_neg = tr[~y[tr]]
        w0 = Xs[tr_pos].mean(axis=0) - Xs[tr_neg].mean(axis=0)
        s_tr = Xs[tr] @ w0
        scale = float(s_tr.std()) + 1e-12
        w0 = w0 / scale  # scores ~ unit variance so the sigmoid/Brier gradient is well-conditioned
        b0 = -float((Xs[tr] @ w0).mean())
        o = tr.copy()
        rng.shuffle(o)

        Xstd.append(Xs)
        W0.append(w0.copy())
        W.append(w0.copy())
        b.append(b0)
        train_idx.append(tr)
        test_idx.append(te)
        order.append(o)

    n_steps = min(len(o) for o in order)

    auroc_traj = []
    cos_traj = []

    def record():
        oof = np.zeros(n, dtype=np.float64)
        for f in range(N_FOLDS):
            te = test_idx[f]
            oof[te] = Xstd[f][te] @ W[f] + b[f]
        auroc_traj.append(auroc(oof, y))
        cos_traj.append(float(np.mean([cos(W0[f], W[f]) for f in range(N_FOLDS)])))

    # Step 0: static DoM (no update applied yet).
    record()

    # Online Brier-loss gradient descent, lockstep across folds.
    for t in range(n_steps):
        for f in range(N_FOLDS):
            i = order[f][t]
            x = Xstd[f][i]
            s = float(x @ W[f] + b[f])
            p = float(sigmoid(np.array(s)))
            yi = 1.0 if y[i] else 0.0
            # d Brier / d s = 2 (p - y) p (1 - p)
            g = 2.0 * (p - yi) * p * (1.0 - p)
            W[f] -= ETA * g * x
            b[f] -= ETA * g
        record()

    auroc_traj = np.asarray(auroc_traj, dtype=np.float64)
    cos_traj = np.asarray(cos_traj, dtype=np.float64)

    static_auroc = float(auroc_traj[0])
    final_auroc = float(auroc_traj[-1])
    best_step = int(np.nanargmax(auroc_traj))
    best_auroc = float(auroc_traj[best_step])

    out = {
        "experiment": "P11-FE732",
        "method": "no-QK TTT online Brier-loss gradient descent on prefill L19 DoM",
        "eta": ETA,
        "n_folds": N_FOLDS,
        "n_steps": int(n_steps),
        "static_dom_auroc_reference": STATIC_DOM_AUROC,
        "static_auroc_oof": static_auroc,
        "final_auroc_oof": final_auroc,
        "best_auroc_oof": best_auroc,
        "best_step": best_step,
        "lift_final_vs_static": final_auroc - static_auroc,
        "lift_best_vs_static": best_auroc - static_auroc,
        "ttt_beats_static_by_2pts": bool((best_auroc - static_auroc) > 0.02),
        "cos_w0_final": float(cos_traj[-1]),
        "cos_w0_min": float(np.nanmin(cos_traj)),
        "cos_drift_to_005": bool(np.nanmin(cos_traj) < 0.10),
        "auroc_trajectory": [float(v) for v in auroc_traj],
        "cos_w0_wt_trajectory": [float(v) for v in cos_traj],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())