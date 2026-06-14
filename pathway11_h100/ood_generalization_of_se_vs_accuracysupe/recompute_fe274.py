"""P11-FE274 — Leave-one-category-out OOD generalization of SE- vs accuracy-supervised probes.

Partition MATH-500 by topic (algebra, geometry, number-theory, prealgebra,
precalculus, intermediate-algebra, counting-&-probability). For each held-out
category, train a difference-of-means (DoM) probe direction on the other six
categories under two supervision signals — ground-truth accuracy and a
semantic-entropy (SE) label (from FE69, K=8 self-consistency proxy as fallback) —
then evaluate AUROC for predicting *true correctness* on the held-out category.

The headline is the OOD gap: Kossen et al. (Tab. 2 leave-one-task-out) report
SE-supervision degrades less under domain shift (+7-10 AUROC) than
accuracy-supervision. F-2 / F-8 are scored 5-fold OOF inside MATH-500's mixed
category pool, which hides this fragility; this script measures it directly.

SE-supervision direction: low SE == confident, so the SE-positive class is the
low-entropy half (median split on the train categories only — no leakage).
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
OUT_JSON = ROOT / "pathway11_h100/ood_category_loto/results.json"

# Category metadata candidates (per-problem MATH-500 subject, length 500).
CAT_CANDIDATES = [
    ROOT / "pathway11_h100/data/math500_meta.json",
    ROOT / "pathway11_h100/data/math500_categories.json",
    ROOT / "data/math500_meta.json",
    ROOT / "data/math500_categories.json",
]
# FE69 semantic-entropy label candidates (continuous SE per problem, length 500).
SE_CANDIDATES = [
    ROOT / "pathway11_h100/fe69_semantic_entropy/se_labels.npz",
    ROOT / "pathway11_h100/data/se_labels.npz",
    ROOT / "pathway11_h100/semantic_entropy/se_labels.npz",
]
SE_KEYS = ("se", "semantic_entropy", "se_label", "entropy")

N_PROBLEMS = 500
MIN_TRAIN_CATS = 2


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def _normalize_cat(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("&", "and")
        .replace(" ", "-")
        .replace("_", "-")
    )


def load_categories() -> np.ndarray | None:
    """Return a length-500 array of normalized category strings, or None."""
    for path in CAT_CANDIDATES:
        if not path.exists():
            continue
        try:
            blob = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        cats = None
        if isinstance(blob, list):
            cats = blob
        elif isinstance(blob, dict):
            for key in ("category", "categories", "subject", "subjects", "type", "topic"):
                if key in blob and isinstance(blob[key], list):
                    cats = blob[key]
                    break
            if cats is None and all(str(k).isdigit() for k in blob):
                cats = [blob[str(i)] if str(i) in blob else blob.get(i) for i in range(N_PROBLEMS)]
            if cats is None and "items" in blob and isinstance(blob["items"], list):
                items = blob["items"]
                for key in ("subject", "category", "type", "topic"):
                    if items and isinstance(items[0], dict) and key in items[0]:
                        cats = [it[key] for it in items]
                        break
        if cats is not None and len(cats) == N_PROBLEMS:
            return np.array([_normalize_cat(c) for c in cats], dtype=object)
    return None


def load_se_labels() -> tuple[np.ndarray | None, str]:
    """Return (length-500 SE array, source-tag), or (None, '')."""
    for path in SE_CANDIDATES:
        if not path.exists():
            continue
        try:
            blob = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            continue
        for key in SE_KEYS:
            if key in blob:
                arr = np.asarray(blob[key], dtype=np.float64).ravel()
                if arr.shape[0] == N_PROBLEMS:
                    return arr, f"fe69:{path.name}:{key}"
    # Fallback: binary-entropy proxy from K=8 self-consistency correctness.
    if K8_DIR.exists():
        se = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
        found = 0
        for i in range(N_PROBLEMS):
            f = K8_DIR / f"problem_{i:03d}.npz"
            if not f.exists():
                continue
            c = np.load(f)["correct"].astype(np.float64)
            p = float(c.mean()) if len(c) else 0.5
            p = min(max(p, 1e-9), 1 - 1e-9)
            se[i] = -(p * np.log(p) + (1 - p) * np.log(1 - p))
            found += 1
        if found == N_PROBLEMS:
            return se, "k8_binary_entropy_proxy"
    return None, ""


def dom_direction(X: np.ndarray, pos_mask: np.ndarray) -> np.ndarray:
    return X[pos_mask].mean(axis=0) - X[~pos_mask].mean(axis=0)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    correct = blob["correct"].astype(bool)
    assert X.shape == (N_PROBLEMS, 1536) and correct.shape == (N_PROBLEMS,)

    cats = load_categories()
    if cats is None:
        print("MISSING_REGEN_INPUT", "math500-category-metadata", file=sys.stderr)
        return 2

    se, se_source = load_se_labels()
    if se is None:
        print("MISSING_REGEN_INPUT", "fe69-semantic-entropy-labels", file=sys.stderr)
        return 2

    valid_se = np.isfinite(se)
    unique_cats = sorted(set(cats.tolist()))

    per_cat: dict[str, dict] = {}
    for held in unique_cats:
        test_mask = cats == held
        train_mask = ~test_mask
        n_train_cats = len(set(cats[train_mask].tolist()))
        if n_train_cats < MIN_TRAIN_CATS:
            continue

        rec: dict[str, float] = {
            "n_test": int(test_mask.sum()),
            "n_train": int(train_mask.sum()),
            "n_train_cats": int(n_train_cats),
            "test_acc": float(correct[test_mask].mean()),
        }

        # Accuracy-supervised probe.
        ytr = correct[train_mask]
        if ytr.any() and (~ytr).any():
            d_acc = dom_direction(X[train_mask], ytr)
            s_acc = X[test_mask] @ d_acc
            rec["auroc_acc"] = auroc(s_acc, correct[test_mask])
        else:
            rec["auroc_acc"] = float("nan")

        # SE-supervised probe: low SE == confident == positive class.
        se_train_ok = train_mask & valid_se
        if se_train_ok.sum() >= 4:
            thr = float(np.median(se[se_train_ok]))
            se_pos = se <= thr  # low entropy -> confident
            tr = train_mask & valid_se
            if se_pos[tr].any() and (~se_pos[tr]).any():
                d_se = dom_direction(X[tr], se_pos[tr])
                te = test_mask  # evaluate on all held-out vs true correctness
                s_se = X[te] @ d_se
                rec["auroc_se"] = auroc(s_se, correct[te])
            else:
                rec["auroc_se"] = float("nan")
        else:
            rec["auroc_se"] = float("nan")

        if np.isfinite(rec["auroc_acc"]) and np.isfinite(rec["auroc_se"]):
            rec["delta_se_minus_acc"] = rec["auroc_se"] - rec["auroc_acc"]
        per_cat[held] = rec

    # Aggregate over categories where both probes scored.
    accs, ses, deltas = [], [], []
    for rec in per_cat.values():
        if np.isfinite(rec.get("auroc_acc", np.nan)) and np.isfinite(rec.get("auroc_se", np.nan)):
            accs.append(rec["auroc_acc"])
            ses.append(rec["auroc_se"])
            deltas.append(rec["auroc_se"] - rec["auroc_acc"])

    # In-distribution reference: pooled 5-fold-free full-data probe AUROC
    # (sanity anchor against the OOD numbers).
    if correct.any() and (~correct).any():
        d_full = dom_direction(X, correct)
        auroc_indist_acc = auroc(X @ d_full, correct)
    else:
        auroc_indist_acc = float("nan")

    out = {
        "experiment": "P11-FE274",
        "method": "leave-one-category-out (Kossen et al. Tab. 2 analogue)",
        "se_source": se_source,
        "n_categories": len(unique_cats),
        "categories": unique_cats,
        "scored_categories": int(len(deltas)),
        "mean_auroc_acc_ood": float(np.mean(accs)) if accs else float("nan"),
        "mean_auroc_se_ood": float(np.mean(ses)) if ses else float("nan"),
        "mean_delta_se_minus_acc": float(np.mean(deltas)) if deltas else float("nan"),
        "auroc_indistribution_acc_fulldata": float(auroc_indist_acc),
        "per_category": per_cat,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(
        f"OK scored_cats={len(deltas)} "
        f"acc_ood={out['mean_auroc_acc_ood']:.4f} "
        f"se_ood={out['mean_auroc_se_ood']:.4f} "
        f"delta={out['mean_delta_se_minus_acc']:.4f} se_source={se_source}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())