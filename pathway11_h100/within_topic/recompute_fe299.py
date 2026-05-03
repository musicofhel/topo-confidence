"""FE299 — Topic-stratified within-topic prefill DoM AUROC.

Sister to FE719 (LOCO-CV) but the *opposite* test: train and test within the
same MATH-500 subject. If within-topic AUROC collapses toward 0.5 while
overall stays at 0.77, the prefill DoM signal is dominated by topic-
familiarity base-rates — refuting H-5's "per-problem decomposability"
framing.

Method: for each subject, run k-fold stratified OOF DoM (k = min(5, min-
class-count)), compute AUROC over the subject's OOF scores.

Inputs: same as FE719 — m15b_prefill.npz + HuggingFaceH4/MATH-500.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/within_topic/results.json"

SUBJECTS = [
    "Algebra",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Number Theory",
    "Prealgebra",
    "Precalculus",
]
SEED = 9999
MAX_FOLDS = 5


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


def within_topic_auroc(X: np.ndarray, y: np.ndarray) -> dict:
    """k-fold OOF DoM AUROC within a single topic."""
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    n = len(y)
    if n_pos == 0 or n_neg == 0:
        return {"auroc": None, "n": n, "n_pos": n_pos, "n_neg": n_neg, "k": 0,
                "skip_reason": "single-class within topic"}
    k = min(MAX_FOLDS, n_pos, n_neg)
    if k < 2:
        return {"auroc": None, "n": n, "n_pos": n_pos, "n_neg": n_neg, "k": k,
                "skip_reason": f"min-class-count {min(n_pos, n_neg)} < 2"}
    folds = stratified_kfold(y, k, SEED)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        train_pos = train_mask & y; train_neg = train_mask & ~y
        if not train_pos.any() or not train_neg.any():
            return {"auroc": None, "n": n, "n_pos": n_pos, "n_neg": n_neg, "k": k,
                    "skip_reason": "fold has single-class training set"}
        dom = X[train_pos].mean(axis=0) - X[train_neg].mean(axis=0)
        scores[test_idx] = X[test_idx] @ dom
    return {"auroc": auroc(scores, y), "n": n, "n_pos": n_pos, "n_neg": n_neg, "k": k}


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    d = np.load(CACHE)
    X = d["prefill"].astype(np.float32)
    y = d["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    subj = np.array(ds["subject"])

    per_subject = {}
    aurocs = []
    for s in SUBJECTS:
        mask = subj == s
        res = within_topic_auroc(X[mask], y[mask])
        per_subject[s] = res
        if res["auroc"] is not None:
            aurocs.append(res["auroc"])
    arr = np.array(aurocs, dtype=float)
    out = {
        "per_subject": per_subject,
        "within_topic_auroc_mean": float(arr.mean()) if len(arr) else None,
        "within_topic_auroc_worst": float(arr.min()) if len(arr) else None,
        "within_topic_auroc_best": float(arr.max()) if len(arr) else None,
        "n_topics_scored": int(len(arr)),
        "f2_topic_detector_threshold": 0.55,
        "f2_topic_detector_only": (
            out_mean := float(arr.mean()) if len(arr) else None
        ) is not None and out_mean < 0.55,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    print(f"within_topic_auroc_mean={out['within_topic_auroc_mean']:.10f}")
    print(f"within_topic_auroc_worst={out['within_topic_auroc_worst']:.10f}")
    print(f"within_topic_auroc_best={out['within_topic_auroc_best']:.10f}")
    print(f"within_topic_n_scored={out['n_topics_scored']}")
    for s in SUBJECTS:
        a = per_subject[s]["auroc"]
        if a is not None:
            key = "within_" + s.lower().replace(" & ", "_").replace(" ", "_")
            print(f"{key}_auroc={a:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
