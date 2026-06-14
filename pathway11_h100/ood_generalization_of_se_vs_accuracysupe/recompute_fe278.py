"""FE278 — OOD generalization of SE- vs accuracy-supervised probes across MATH-500 categories.

Leave-one-category-out (Kossen et al. 2024, Tab. 2 methodology) in our domain.
MATH-500 is partitioned by subject (algebra, geometry, number theory, ...).
For each held-out category a linear (ridge) probe is fit on the remaining
categories under two supervision signals:

  * accuracy-supervised  — target is ground-truth correctness (the F-2/F-8 signal)
  * SE-supervised        — target is a negative semantic-entropy proxy derived
                           from the K=8 self-consistency cache (FE69 input)

Both probes are scored by AUROC for *correctness* on the held-out category.
Headline is the OOD gap: mixed-category 5-fold OOF AUROC (in-distribution, the
way F-2/F-8 are reported) minus the mean leave-one-category-out AUROC, plus the
per-category ∆AUROC = SE-supervised − accuracy-supervised.

SE proxy note: answer strings are not cached, so true semantic-entropy
clustering is unavailable. We use the binary entropy of the K=8 correct-fraction
H(p_correct) as a confidence proxy; the SE-supervised target is -H so that
higher score => more confident => predicted correct. If a precomputed FE69 SE
label cache exists it is preferred over the proxy.
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
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
# Precomputed semantic-entropy labels (FE69), if available, take priority.
SE_NPZ = ROOT / "pathway11_h100/data/fe69_se_labels.npz"
# MATH-500 subject labels, one per problem (list of 500 strings or {idx: subject}).
CATEGORIES_JSON = ROOT / "pathway11_h100/data/math500_categories.json"
OUT_JSON = ROOT / "pathway11_h100/se_ood_categories/results.json"

SEED = 9999
N_FOLDS = 5
RIDGE_ALPHA = 100.0


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
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


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def fit_ridge(X: np.ndarray, t: np.ndarray, alpha: float):
    """Standardised ridge regression. Returns (mu, sd, w, b)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    Xs = (X - mu) / sd
    n, d = Xs.shape
    tc = t - t.mean()
    A = Xs.T @ Xs + alpha * np.eye(d, dtype=np.float64)
    w = np.linalg.solve(A, Xs.T @ tc)
    return mu, sd, w, float(t.mean())


def score_ridge(X: np.ndarray, mu, sd, w, b) -> np.ndarray:
    return ((X - mu) / sd) @ w + b


def load_categories(n: int) -> np.ndarray | None:
    if not CATEGORIES_JSON.exists():
        return None
    obj = json.loads(CATEGORIES_JSON.read_text())
    if isinstance(obj, dict):
        # Either {idx: subject} or {"subjects": [...]} / {"categories": [...]}.
        for key in ("subjects", "categories", "subject", "category"):
            if key in obj and isinstance(obj[key], list):
                obj = obj[key]
                break
        else:
            obj = [obj[str(i)] if str(i) in obj else obj.get(i) for i in range(n)]
    cats = [str(c).strip().lower().replace(" ", "_") if c is not None else "unknown"
            for c in obj]
    if len(cats) != n:
        return None
    return np.array(cats, dtype=object)


def load_se_target(n: int) -> np.ndarray | None:
    """Return continuous SE proxy (higher => more uncertain). Prefer FE69 cache."""
    if SE_NPZ.exists():
        blob = np.load(SE_NPZ)
        for key in ("se", "semantic_entropy", "se_label"):
            if key in blob:
                arr = blob[key].astype(np.float64)
                if arr.shape[0] == n:
                    return arr
    # Fall back to K=8 binary-entropy proxy.
    if not K8_DIR.exists():
        return None
    p = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        f = K8_DIR / f"problem_{i:03d}.npz"
        if not f.exists():
            return None
        c = np.load(f)["correct"].astype(bool)
        p[i] = float(c.mean())
    return binary_entropy(p)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    correct = blob["correct"].astype(bool)
    n = len(correct)
    assert X.shape == (n, 1536)

    cats = load_categories(n)
    if cats is None:
        print("MISSING_REGEN_INPUT", CATEGORIES_JSON, file=sys.stderr); return 2

    se = load_se_target(n)
    if se is None:
        print("MISSING_REGEN_INPUT", f"{SE_NPZ} or {K8_DIR}", file=sys.stderr); return 2

    acc_target = correct.astype(np.float64)
    se_target = -se  # higher => more confident => predict correct

    # ---- In-distribution reference: mixed-category 5-fold OOF (F-2/F-8 style) ----
    folds = stratified_kfold(correct, N_FOLDS, SEED)
    oof_acc = np.zeros(n, dtype=np.float64)
    oof_se = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        tr = np.ones(n, dtype=bool); tr[test_idx] = False
        mu, sd, w, b = fit_ridge(X[tr], acc_target[tr], RIDGE_ALPHA)
        oof_acc[test_idx] = score_ridge(X[test_idx], mu, sd, w, b)
        mu, sd, w, b = fit_ridge(X[tr], se_target[tr], RIDGE_ALPHA)
        oof_se[test_idx] = score_ridge(X[test_idx], mu, sd, w, b)
    id_auroc_acc = auroc(oof_acc, correct)
    id_auroc_se = auroc(oof_se, correct)

    # ---- OOD: leave-one-category-out ----
    uniq = sorted({str(c) for c in cats})
    per_category = {}
    ood_acc_list, ood_se_list, delta_list = [], [], []
    for held in uniq:
        test_mask = np.array([str(c) == held for c in cats], dtype=bool)
        train_mask = ~test_mask
        yte = correct[test_mask]
        rec = {
            "n_test": int(test_mask.sum()),
            "n_train": int(train_mask.sum()),
            "n_test_correct": int(yte.sum()),
        }
        if yte.sum() == 0 or yte.sum() == len(yte) or train_mask.sum() < 10:
            rec["auroc_acc_supervised"] = float("nan")
            rec["auroc_se_supervised"] = float("nan")
            rec["delta_auroc_se_minus_acc"] = float("nan")
            per_category[held] = rec
            continue
        mu, sd, w, b = fit_ridge(X[train_mask], acc_target[train_mask], RIDGE_ALPHA)
        s_acc = score_ridge(X[test_mask], mu, sd, w, b)
        a_acc = auroc(s_acc, yte)
        mu, sd, w, b = fit_ridge(X[train_mask], se_target[train_mask], RIDGE_ALPHA)
        s_se = score_ridge(X[test_mask], mu, sd, w, b)
        a_se = auroc(s_se, yte)
        rec["auroc_acc_supervised"] = a_acc
        rec["auroc_se_supervised"] = a_se
        rec["delta_auroc_se_minus_acc"] = a_se - a_acc
        per_category[held] = rec
        ood_acc_list.append(a_acc); ood_se_list.append(a_se); delta_list.append(a_se - a_acc)

    mean_ood_acc = float(np.mean(ood_acc_list)) if ood_acc_list else float("nan")
    mean_ood_se = float(np.mean(ood_se_list)) if ood_se_list else float("nan")
    mean_delta = float(np.mean(delta_list)) if delta_list else float("nan")

    out = {
        "experiment": "P11-FE278",
        "n_problems": int(n),
        "n_categories": len(uniq),
        "categories": uniq,
        "ridge_alpha": RIDGE_ALPHA,
        "se_source": "fe69_cache" if SE_NPZ.exists() else "k8_binary_entropy_proxy",
        "in_distribution_5fold_oof": {
            "auroc_acc_supervised": id_auroc_acc,
            "auroc_se_supervised": id_auroc_se,
        },
        "ood_leave_one_category_out": {
            "mean_auroc_acc_supervised": mean_ood_acc,
            "mean_auroc_se_supervised": mean_ood_se,
            "mean_delta_auroc_se_minus_acc": mean_delta,
        },
        "ood_gap": {
            "acc_supervised_id_minus_ood": id_auroc_acc - mean_ood_acc,
            "se_supervised_id_minus_ood": id_auroc_se - mean_ood_se,
        },
        "per_category": per_category,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())