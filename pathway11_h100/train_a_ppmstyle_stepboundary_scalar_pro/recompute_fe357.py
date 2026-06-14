"""P11-FE357 — PPM-style step-boundary scalar probe with rStar-Math Bradley-Terry loss.

Direct test of Refutation #2: does a step-boundary / process-reward style probe beat
F-2's prefill DoM AUROC (0.7731)? rStar-Math (Guan et al., Eq. 4) trains a Process
Preference Model by a Bradley-Terry pairwise loss over step boundaries, taking the
top-2-Q steps within a problem as positives and the bottom-2-Q steps as negatives,
where Q is a Monte-Carlo rollout value estimate.

Faithful adaptation to our cache: we have no per-step trajectory activations, only the
L19 prefill hidden state per problem. We therefore (a) estimate the MC Q-value per
problem from the K=8 self-consistency cache (Q_i = fraction of 8 rollouts correct,
exactly rStar-Math's rollout estimator), (b) within each training fold take the
top-quantile-Q problems as positives and the bottom-quantile-Q as negatives — the
global analog of "top-2 / bottom-2 Q" — and (c) fit a scalar probe s(x)=w·x on the
standardized prefill features by minimizing the Bradley-Terry pairwise loss
  L = mean_{i in pos, j in neg} -log sigmoid(s(x_i) - s(x_j)) + l2*||w||^2   (Eq. 4).
OOF probe scores are then AUROC-evaluated against the binary correctness labels and
compared to the prefill DoM 0.7731 baseline.

If the BT step-boundary probe does not exceed 0.7731, Refutation #2 fails and F-2's
"best signal" framing stands; if it does, F-2 is downgraded to "best prefill-time signal."
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
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
OUT_JSON = ROOT / "pathway11_h100/ppm_step_boundary/results.json"

SEED = 9999
N_FOLDS = 5
F2_PREFILL_AUROC = 0.7731

# BT probe hyperparameters
POS_QUANTILE = 0.75   # top 25% Q within train fold -> positives
NEG_QUANTILE = 0.25   # bottom 25% Q within train fold -> negatives
L2 = 1e-2
LR = 0.05
ITERS = 400
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def load_q_values(n: int, correct: np.ndarray) -> tuple[np.ndarray, int]:
    """MC Q-value per problem = fraction of K=8 rollouts correct.

    Falls back to the binary correctness label for any problem whose K=8 file
    is missing (degrades gracefully rather than aborting).
    """
    q = correct.astype(np.float64).copy()
    found = 0
    for i in range(n):
        f = K8_DIR / f"problem_{i:03d}.npz"
        if not f.exists():
            continue
        try:
            c8 = np.load(f)["correct"].astype(bool)
        except Exception:
            continue
        if c8.size == 0:
            continue
        q[i] = float(c8.mean())
        found += 1
    return q, found


def fit_bt_probe(Xpos: np.ndarray, Xneg: np.ndarray) -> np.ndarray:
    """Minimize the rStar-Math Bradley-Terry pairwise loss (Eq. 4) via Adam.

    Returns the scalar-probe weight vector w (operating in standardized space).
    """
    d = Xpos.shape[1]
    w = np.zeros(d, dtype=np.float64)
    m = np.zeros(d, dtype=np.float64)
    v = np.zeros(d, dtype=np.float64)
    P, N = len(Xpos), len(Xneg)
    norm = float(P * N)
    for t in range(1, ITERS + 1):
        s_pos = Xpos @ w
        s_neg = Xneg @ w
        M = s_pos[:, None] - s_neg[None, :]          # (P, N) pairwise margins
        # d/dM of -log sigmoid(M) = sigmoid(M) - 1
        A = 1.0 / (1.0 + np.exp(-M)) - 1.0           # (P, N)
        row_w = A.sum(axis=1)                        # (P,)
        col_w = A.sum(axis=0)                        # (N,)
        grad = (Xpos.T @ row_w - Xneg.T @ col_w) / norm + 2.0 * L2 * w
        m = ADAM_B1 * m + (1 - ADAM_B1) * grad
        v = ADAM_B2 * v + (1 - ADAM_B2) * (grad * grad)
        mhat = m / (1 - ADAM_B1 ** t)
        vhat = v / (1 - ADAM_B2 ** t)
        w -= LR * mhat / (np.sqrt(vhat) + ADAM_EPS)
    return w


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)
    n = len(y)

    # Reference DoM baseline (optional — for context only).
    auroc_dom = float("nan")
    if DOM_NPZ.exists():
        try:
            dom = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
            auroc_dom = auroc(dom, y)
        except Exception:
            pass

    q, n_q_found = load_q_values(n, y)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(n, dtype=np.float64)
    pos_counts, neg_counts = [], []

    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        qtr = q[train_idx]

        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd < 1e-8] = 1.0
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd

        hi = np.quantile(qtr, POS_QUANTILE)
        lo = np.quantile(qtr, NEG_QUANTILE)
        pos_mask = qtr >= hi
        neg_mask = qtr <= lo
        # Guard against degenerate quantiles (e.g. constant Q): fall back to
        # the binary correctness split within the training fold.
        if pos_mask.sum() == 0 or neg_mask.sum() == 0 or hi <= lo:
            ytr = y[train_idx]
            pos_mask, neg_mask = ytr, ~ytr
        pos_counts.append(int(pos_mask.sum()))
        neg_counts.append(int(neg_mask.sum()))

        w = fit_bt_probe(Xtr_s[pos_mask], Xtr_s[neg_mask])
        oof[test_idx] = Xte_s @ w

    auroc_bt = auroc(oof, y)

    out = {
        "experiment": "P11-FE357",
        "description": "rStar-Math BT step-boundary scalar probe at L19 vs F-2 prefill DoM",
        "auroc_bt_step_boundary_oof": float(auroc_bt),
        "f2_prefill_dom_auroc": F2_PREFILL_AUROC,
        "auroc_dom_reference_recomputed": auroc_dom,
        "delta_vs_f2": float(auroc_bt - F2_PREFILL_AUROC),
        "beats_f2": bool(auroc_bt > F2_PREFILL_AUROC),
        "refutation2_verdict": (
            "F-2 downgraded to 'best prefill-time signal'"
            if auroc_bt > F2_PREFILL_AUROC
            else "F-2 'best signal' framing stands"
        ),
        "q_source": "k8_mc_rollout_fraction",
        "n_problems_with_k8_q": int(n_q_found),
        "n_problems": int(n),
        "pos_quantile": POS_QUANTILE,
        "neg_quantile": NEG_QUANTILE,
        "mean_pos_per_fold": float(np.mean(pos_counts)),
        "mean_neg_per_fold": float(np.mean(neg_counts)),
        "n_folds": N_FOLDS,
        "l2": L2,
        "seed": SEED,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())