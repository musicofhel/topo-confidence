"""P11-FE131 — Free no-train discreteness sanity check.

K-means clusters the cached pathway-11 L19 prefill activations (k in {64, 256,
1024}) and treats cluster membership as a synthetic discrete 'code'. Each
cluster's training-fold correctness rate becomes a per-sample predictor of K=1
correctness, scored out-of-fold. Compares the cluster-code precision-at-matched-
recall and AUROC against the L19 DoM probe.

Motivation (Tamkin §3): even k=1 codebooks at attention recover near-baseline LM
performance, suggesting much discrete-code structure already lives in the frozen
activations and may not require straight-through fine-tuning to extract. K-means
is the cheapest possible discretization. If k-means cluster precision matches DoM
(~0.77), a full codebook fine-tune (P11-FE129) is likely worth the H100 cost. If
k-means cluster precision is at chance, codebook fine-tuning is the load-bearing
component.
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
OUT_JSON = ROOT / "pathway11_h100/kmeans_discrete_code/results.json"

KS = [64, 256, 1024]
N_FOLDS = 5
SEED = 9999
RECALL_TARGETS = [round(0.1 * i, 2) for i in range(1, 10)]


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


def precision_at_recall(scores: np.ndarray, labels: np.ndarray, target_recall: float) -> float:
    """Precision at the lowest threshold whose recall >= target_recall.

    Scores ranked descending; the positive-class prevalence floor applies when
    no threshold reaches the target.
    """
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = labels[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    hit = np.flatnonzero(recall >= target_recall)
    if len(hit) == 0:
        return float(precision[-1])
    return float(precision[hit[0]])


def oof_cluster_scores(labels: np.ndarray, cluster_ids: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    """Per-sample OOF predictor: its cluster's training-fold correctness rate.

    Empty (test-only) clusters fall back to the training global mean, so no test
    label leaks into its own score.
    """
    n = len(labels)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        global_rate = float(labels[train_mask].mean())
        rate = {}
        for cid in np.unique(cluster_ids[train_mask]):
            members = train_mask & (cluster_ids == cid)
            rate[int(cid)] = float(labels[members].mean())
        for i in test_idx:
            scores[i] = rate.get(int(cluster_ids[i]), global_rate)
    return scores


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr); return 2

    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("MISSING_REGEN_INPUT sklearn", file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)
    n = len(y)

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert dom_score.shape == (n,)

    folds = stratified_kfold(y, N_FOLDS, SEED)

    dom_auroc = auroc(dom_score, y)
    dom_pr = {str(r): precision_at_recall(dom_score, y, r) for r in RECALL_TARGETS}

    per_k = {}
    for k_req in KS:
        k_eff = min(k_req, n)  # k cannot exceed sample count
        km = KMeans(n_clusters=k_eff, n_init=10, random_state=SEED)
        cluster_ids = km.fit_predict(X)
        n_nonempty = int(len(np.unique(cluster_ids)))

        clu_score = oof_cluster_scores(y, cluster_ids, folds)
        clu_auroc = auroc(clu_score, y)
        clu_pr = {str(r): precision_at_recall(clu_score, y, r) for r in RECALL_TARGETS}

        per_k[str(k_req)] = {
            "k_requested": k_req,
            "k_effective": k_eff,
            "n_nonempty_clusters": n_nonempty,
            "auroc_oof_cluster_code": clu_auroc,
            "precision_at_recall_cluster_code": clu_pr,
            "precision_gap_vs_dom_at_recall": {
                str(r): (clu_pr[str(r)] - dom_pr[str(r)]) for r in RECALL_TARGETS
            },
        }

    out = {
        "experiment": "P11-FE131",
        "n_samples": n,
        "base_rate_correct": float(y.mean()),
        "auroc_dom_probe": dom_auroc,
        "precision_at_recall_dom_probe": dom_pr,
        "recall_targets": RECALL_TARGETS,
        "per_k": per_k,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())