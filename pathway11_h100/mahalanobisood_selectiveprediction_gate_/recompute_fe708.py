"""P11-FE708 — Mahalanobis-OOD selective-prediction gate (CAT shrinkage precision).

Implements the Mahalanobis class-conditional confidence score of 2603.03163
(Eq. 4 distance + Eq. 6 selective gate) on the cached L19 prefill activations.
At d=1536, N≈500 the empirical within-class covariance is rank-deficient (d>N),
so the precision matrix is estimated with a Kubokawa-Srivastava-style
optimal-shrinkage estimator (analytic Ledoit-Wolf toward a scaled identity).

Two variants are scored out-of-fold:
  - shared (pooled within-class) precision   → linear / LDA boundary
  - per-class precision                       → ellipsoidal / QDA boundary

Each yields a per-problem correctness-confidence score; we report AUROC and the
selective-prediction accuracy at coverage 0.5, comparing against the DoM probe
baseline and F-8's reported 71.6% answered accuracy at coverage 0.5.
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
OUT_JSON = ROOT / "pathway11_h100/mahalanobis_ood/results.json"

N_FOLDS = 5
SEED = 9999
F8_SELECTIVE_ACC = 0.716  # F-8 logistic probe, answered accuracy @ coverage 0.5


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


def lw_shrinkage_cov(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf / Kubokawa-Srivastava optimal shrinkage toward scaled identity.

    Returns a full-rank d×d covariance estimate Sigma = (1-d)*S + d*m*I, with the
    analytic optimal shrinkage intensity. S is the MLE (1/n) within-class scatter.
    """
    Xc = X - mu
    n, d = Xc.shape
    S = (Xc.T @ Xc) / n
    m = float(np.trace(S)) / d                       # grand mean eigenvalue
    target = m * np.eye(d, dtype=S.dtype)
    d2 = float(np.sum((S - target) ** 2)) / d        # dispersion of S from target
    # b2: mean Frobenius dist of rank-1 sample cov to S, via Gram-matrix identities
    norms2 = np.einsum("ij,ij->i", Xc, Xc)           # ||x_k||^2
    G = Xc @ Xc.T                                     # n×n Gram
    sum_x4 = float(np.sum(norms2 ** 2))               # sum ||x_k x_k^T||_F^2
    sum_xSx = float(np.sum(G ** 2)) / n               # sum <x_k x_k^T, S>
    sumF_S = float(np.sum(S ** 2))                    # ||S||_F^2
    b2_raw = (sum_x4 - 2.0 * sum_xSx + n * sumF_S) / (n * n)
    b2 = max(0.0, b2_raw) / d
    delta = 0.0 if d2 <= 0 else float(np.clip(b2 / d2, 0.0, 1.0))
    return (1.0 - delta) * S + delta * target


def mahal_batch(X: np.ndarray, mu: np.ndarray, P: np.ndarray) -> np.ndarray:
    D = X - mu
    return np.einsum("ij,jk,ik->i", D, P, D)


def selective_accuracy(conf: np.ndarray, correct: np.ndarray, coverage: float):
    n = len(conf)
    k = int(round(coverage * n))
    k = max(1, min(n, k))
    answered = np.argsort(-conf)[:k]
    return float(correct[answered].mean()), int(k)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)
    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert dom_score.shape == (500,)

    n = len(y)
    shared_conf = np.zeros(n, dtype=np.float64)
    qda_conf = np.zeros(n, dtype=np.float64)

    folds = stratified_kfold(y, N_FOLDS, SEED)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]
        Xp, Xn = Xtr[ytr], Xtr[~ytr]
        mu_p, mu_n = Xp.mean(0), Xn.mean(0)
        np_, nn_ = len(Xp), len(Xn)
        log_prior = float(np.log(np_ / nn_))

        # --- per-class (QDA / ellipsoidal) shrinkage precision ---
        Sig_p = lw_shrinkage_cov(Xp, mu_p)
        Sig_n = lw_shrinkage_cov(Xn, mu_n)
        P_p = np.linalg.inv(Sig_p); P_n = np.linalg.inv(Sig_n)
        ld_p = float(np.linalg.slogdet(P_p)[1])
        ld_n = float(np.linalg.slogdet(P_n)[1])
        ll_p = -0.5 * mahal_batch(Xte, mu_p, P_p) + 0.5 * ld_p
        ll_n = -0.5 * mahal_batch(Xte, mu_n, P_n) + 0.5 * ld_n
        qda_conf[test_idx] = (ll_p - ll_n) + log_prior

        # --- shared / pooled within-class (LDA / linear) shrinkage precision ---
        Xpooled = np.vstack([Xp - mu_p, Xn - mu_n])
        Sig_pool = lw_shrinkage_cov(Xpooled, np.zeros(Xpooled.shape[1]))
        P_sh = np.linalg.inv(Sig_pool)
        shared_conf[test_idx] = (
            -mahal_batch(Xte, mu_p, P_sh) + mahal_batch(Xte, mu_n, P_sh)
        ) + log_prior

    base_acc = float(y.mean())
    coverages = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def sweep(conf):
        return {f"{c:.1f}": selective_accuracy(conf, y, c)[0] for c in coverages}

    qda_acc50, k50 = selective_accuracy(qda_conf, y, 0.5)
    shared_acc50, _ = selective_accuracy(shared_conf, y, 0.5)
    dom_acc50, _ = selective_accuracy(dom_score, y, 0.5)

    out = {
        "experiment": "P11-FE708",
        "n": n,
        "d": int(X.shape[1]),
        "base_accuracy": base_acc,
        "estimator": "ledoit_wolf_shrinkage_precision (Kubokawa-Srivastava style, scaled-identity target)",
        "auroc": {
            "mahalanobis_qda": auroc(qda_conf, y),
            "mahalanobis_shared": auroc(shared_conf, y),
            "dom_baseline": auroc(dom_score, y),
        },
        "selective_acc_at_coverage_0.5": {
            "mahalanobis_qda": qda_acc50,
            "mahalanobis_shared": shared_acc50,
            "dom_baseline": dom_acc50,
            "f8_logistic_probe": F8_SELECTIVE_ACC,
            "n_answered": k50,
        },
        "selective_acc_sweep": {
            "mahalanobis_qda": sweep(qda_conf),
            "mahalanobis_shared": sweep(shared_conf),
            "dom_baseline": sweep(dom_score),
        },
        "beats_f8_at_coverage_0.5": bool(max(qda_acc50, shared_acc50) > F8_SELECTIVE_ACC),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["selective_acc_at_coverage_0.5"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())