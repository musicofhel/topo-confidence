"""P11-FE1116 — Label-weighted moment operator probe at L19.

Tests whether the label-weighted second-moment operator
C = (1/n) Σ yᵢ zᵢ zᵢᵀ captures second-order label-feature correlations
that unsupervised PCA (cov-spectrum, 0.7928) and first-order DoM (0.7731)
miss, per Neural LoFi (2605.13612).

For each OOF fold: center/scale on train, build the label-weighted moment
operator on train (using {0,1} and centered ±1 label weightings), extract
top-k eigenvectors, project train+test onto them, fit logistic regression,
and score the held-out fold. Aggregate OOF AUROC and compare against the
DoM, PC1+PC9, cov-spectrum, and full-dim L2 ceilings.
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

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/label_moment_operator/results.json"

N_FOLDS = 5
SEED = 9999
TOP_K = [1, 2, 5, 10, 20, 50]

# Published comparison baselines (1024-tok canonical).
DOM_AUROC = 0.7731
PC1_PC9_AUROC = 0.7856
COV_SPECTRUM_AUROC = 0.7928
FULL_DIM_CEILING = 0.7847


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


def label_moment_operator(Z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """C = (1/n) Σ wᵢ zᵢ zᵢᵀ — weighted second-moment, symmetrized."""
    n = Z.shape[0]
    C = (Z * w[:, None]).T @ Z / n
    return 0.5 * (C + C.T)


def oof_top_eigvec_auroc(X, y, weighting, top_k, mu_eig=False):
    """OOF AUROC for a logistic probe on the top-k label-moment eigenvectors.

    weighting: 'binary' uses wᵢ = yᵢ∈{0,1}; 'centered' uses wᵢ = ±1
    (y centered to mean 0), isolating the label-contrastive second moment.
    """
    n = len(y)
    oof = np.zeros(n, dtype=np.float64)
    folds = stratified_kfold(y, N_FOLDS, SEED)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]

        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0) + 1e-8
        Ztr = (Xtr - mu) / sd
        Zte = (Xte - mu) / sd

        if weighting == "binary":
            w = ytr.astype(np.float64)
        else:  # centered ±1 contrast
            w = ytr.astype(np.float64) - ytr.mean()

        C = label_moment_operator(Ztr, w)
        evals, evecs = np.linalg.eigh(C)
        # Largest-magnitude eigenvalues carry the strongest label-feature
        # correlation (eigh returns ascending; take both ends by |λ|).
        order = np.argsort(np.abs(evals))[::-1]
        V = evecs[:, order[:top_k]]

        Ftr = Ztr @ V
        Fte = Zte @ V
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Ftr, ytr)
        oof[test_idx] = clf.predict_proba(Fte)[:, 1]
    return float(auroc(oof, y))


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

    results = {"binary": {}, "centered": {}}
    best = {"auroc": float("-inf"), "weighting": None, "top_k": None}
    for weighting in ("binary", "centered"):
        for k in TOP_K:
            a = oof_top_eigvec_auroc(X, y, weighting, k)
            results[weighting][str(k)] = a
            if a > best["auroc"]:
                best = {"auroc": a, "weighting": weighting, "top_k": k}

    out = {
        "experiment": "P11-FE1116",
        "description": "Label-weighted moment operator (C = 1/n Σ yᵢ zᵢ zᵢᵀ) eigenvector probe at L19",
        "n": int(len(y)),
        "d": int(X.shape[1]),
        "top_k_grid": TOP_K,
        "auroc_label_moment": results,
        "best_label_moment": best,
        "auroc_dom_reference_oof": float(auroc(dom_score, y)),
        "baselines": {
            "dom": DOM_AUROC,
            "pc1_pc9": PC1_PC9_AUROC,
            "cov_spectrum": COV_SPECTRUM_AUROC,
            "full_dim_ceiling": FULL_DIM_CEILING,
        },
        "beats_cov_spectrum": bool(best["auroc"] > COV_SPECTRUM_AUROC),
        "beats_full_dim_ceiling": bool(best["auroc"] > FULL_DIM_CEILING),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())