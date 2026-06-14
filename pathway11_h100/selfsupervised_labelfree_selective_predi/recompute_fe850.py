"""P11-FE850 — Self-supervised, label-free selective prediction via SpecExit targets.

Generates SpecExit-style minimum-sufficient-prefix regression targets from cached
per-paragraph truncated-completion checks on the MATH-500 K=1 generations (a
label-free, self-supervised signal: the earliest paragraph boundary at which a
truncated reasoning prefix still completes to the correct answer). A ridge
regression head is fit OOF on the L19 prefill activations against this target;
the predicted minimum-sufficient-prefix fraction is inverted into a confidence
score, and a risk-coverage curve is computed and compared against the F-8
supervised-DoM selective-prediction operating point (71.6% answered accuracy at
coverage 0.5).

The truncation-completion checks (the "2h label generation" phase) are produced
upstream and cached per problem; this script is the pure-CPU reduction + probe
fit. If the truncation cache is absent the script exits MISSING_REGEN_INPUT.
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
TRUNC_DIR = ROOT / "pathway11_h100/data/specexit_truncations"
OUT_JSON = ROOT / "pathway11_h100/specexit_selfsup/results.json"

N = 500
N_FOLDS = 5
SEED = 9999
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0)
F8_BASELINE_COV05 = 0.716  # F-8 supervised selective-prediction acc at coverage 0.5
COVERAGES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
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


def specexit_target(para_correct: np.ndarray) -> float:
    """Minimum-sufficient-prefix fraction.

    para_correct[k] = truncated completion from the first k+1 paragraphs is
    correct. The minimum-sufficient prefix is the earliest boundary from which
    correctness holds through the end of the chain (a stable early-exit point).
    Returns a fraction in (0, 1]; 1.0 if no stable sufficient prefix exists.
    """
    pc = np.asarray(para_correct).astype(bool)
    p = len(pc)
    if p == 0:
        return 1.0
    suffix_all = True
    min_idx = p  # default: never sufficient -> full chain
    for k in range(p - 1, -1, -1):
        suffix_all = suffix_all and pc[k]
        if suffix_all:
            min_idx = k
        else:
            break
    return float((min_idx + 1) / p)


def load_targets() -> np.ndarray | None:
    if not TRUNC_DIR.exists():
        return None
    targets = np.full(N, np.nan, dtype=np.float64)
    for i in range(N):
        f = TRUNC_DIR / f"problem_{i:03d}.npz"
        if not f.exists():
            return None
        blob = np.load(f)
        if "para_correct" not in blob:
            return None
        targets[i] = specexit_target(blob["para_correct"])
    if np.isnan(targets).any():
        return None
    return targets


def ridge_fit(Xtr, ytr, alpha):
    n, d = Xtr.shape
    A = Xtr.T @ Xtr + alpha * np.eye(d, dtype=Xtr.dtype)
    b = Xtr.T @ ytr
    return np.linalg.solve(A, b)


def risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> list[dict]:
    """Answer the most-confident fraction; report accuracy among answered."""
    order = np.argsort(-confidence, kind="stable")
    sorted_correct = correct.astype(np.float64)[order]
    curve = []
    n = len(confidence)
    for cov in COVERAGES:
        m = max(1, int(round(cov * n)))
        acc = float(sorted_correct[:m].mean())
        curve.append({"coverage": float(cov), "n_answered": m, "accuracy": acc})
    return curve


def acc_at_cov(curve: list[dict], cov: float) -> float:
    for pt in curve:
        if abs(pt["coverage"] - cov) < 1e-9:
            return pt["accuracy"]
    return float("nan")


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    correct = blob["correct"].astype(bool)
    assert X.shape == (N, 1536) and correct.shape == (N,)
    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)

    targets = load_targets()
    if targets is None:
        print("MISSING_REGEN_INPUT", TRUNC_DIR, file=sys.stderr); return 2

    # OOF ridge regression: prefill activations -> self-supervised SpecExit target.
    folds = stratified_kfold(correct, N_FOLDS, SEED)
    oof_pred = np.zeros(N, dtype=np.float64)
    chosen_alpha = np.zeros(N, dtype=np.float64)

    for test_idx in folds:
        train_mask = np.ones(N, dtype=bool); train_mask[test_idx] = False
        Xtr_raw, Xte_raw = X[train_mask], X[test_idx]
        ttr = targets[train_mask]

        mu = Xtr_raw.mean(axis=0)
        sd = Xtr_raw.std(axis=0); sd[sd < 1e-8] = 1.0
        Xtr = (Xtr_raw - mu) / sd
        Xte = (Xte_raw - mu) / sd
        tmu = ttr.mean()
        ttr_c = ttr - tmu

        # Inner split to pick alpha (minimize held-out target MSE, label-free).
        inner = stratified_kfold(correct[train_mask], 3, SEED + 1)
        best_alpha, best_mse = RIDGE_ALPHAS[0], np.inf
        for alpha in RIDGE_ALPHAS:
            mse_acc = 0.0
            for ival in inner:
                imask = np.ones(len(ttr), dtype=bool); imask[ival] = False
                w = ridge_fit(Xtr[imask], ttr_c[imask], alpha)
                pred = Xtr[ival] @ w + tmu
                mse_acc += float(np.mean((pred - ttr[ival]) ** 2))
            if mse_acc < best_mse:
                best_mse, best_alpha = mse_acc, alpha

        w = ridge_fit(Xtr, ttr_c, best_alpha)
        oof_pred[test_idx] = Xte @ w + tmu
        chosen_alpha[test_idx] = best_alpha

    # Lower predicted minimum-sufficient-prefix => problem resolves early => more
    # confident => answer it. Confidence is the negated predicted target.
    selfsup_conf = -oof_pred

    selfsup_curve = risk_coverage(selfsup_conf, correct)
    dom_curve = risk_coverage(dom_score, correct)

    selfsup_cov05 = acc_at_cov(selfsup_curve, 0.5)
    dom_cov05 = acc_at_cov(dom_curve, 0.5)

    out = {
        "experiment": "P11-FE850",
        "n": N,
        "target_describe": {
            "mean": float(targets.mean()),
            "std": float(targets.std()),
            "min": float(targets.min()),
            "max": float(targets.max()),
            "frac_full_chain": float(np.mean(targets >= 1.0 - 1e-9)),
        },
        "auroc_selfsup_probe_oof": auroc(selfsup_conf, correct),
        "auroc_supervised_dom": auroc(dom_score, correct),
        "selfsup_risk_coverage": selfsup_curve,
        "supervised_dom_risk_coverage": dom_curve,
        "acc_at_coverage_0.5": {
            "selfsup_labelfree": float(selfsup_cov05),
            "supervised_dom": float(dom_cov05),
            "f8_baseline": F8_BASELINE_COV05,
            "selfsup_minus_f8": float(selfsup_cov05 - F8_BASELINE_COV05),
            "selfsup_minus_dom": float(selfsup_cov05 - dom_cov05),
        },
        "chosen_alpha_hist": {
            str(a): int(np.sum(chosen_alpha == a)) for a in RIDGE_ALPHAS
        },
        "matches_f8_operating_point": bool(selfsup_cov05 >= F8_BASELINE_COV05 - 0.02),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())