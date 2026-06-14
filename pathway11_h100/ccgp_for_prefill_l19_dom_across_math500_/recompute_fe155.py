"""FE155 — CCGP for prefill L19 DoM across MATH-500 topic strata.

Tests whether the prefill L19 DoM is a label-aligned correctness direction or an
input-geometry (topic-clustered) detector. Following Paper 2401.13558's
Cross-Condition Generalization Performance (CCGP): fit the DoM (class-mean
difference) probe on the problems of one MATH-500 topic, then score the problems
of every *other* topic and measure AUROC. A true correctness axis should
generalize across topics (cross-topic AUROC ≈ the pooled in-domain 0.7731); a
topic-detector that merely correlates with correctness should degrade sharply.

Reports per-(train_topic → test_topic) AUROC, the mean across-topic AUROC, the
pooled in-domain OOF AUROC reference, and the mean degradation.
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
OUT_JSON = ROOT / "pathway11_h100/ccgp_topic/results.json"

# Candidate locations for per-problem MATH-500 subject/topic metadata. The
# cache NPZ carries no topic field, so we look for an aligned (500,) source.
TOPIC_CANDIDATES = [
    ROOT / "pathway11_h100/data/math500_subjects.json",
    ROOT / "pathway11_h100/data/math500_meta.json",
    ROOT / "data/math500_subjects.json",
    ROOT / "data/math500_meta.json",
    ROOT / "data/math500.json",
    ROOT / "configs/math500_subjects.json",
]
TOPIC_NPZ_CANDIDATES = [
    ROOT / "pathway11_h100/data/math500_subjects.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz",
]

N_FOLDS = 5
SEED = 9999
MIN_PER_TOPIC = 10   # require enough problems to fit/score a topic
MIN_PER_CLASS = 2    # require both classes present on each side


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


def _canonical_topic(raw: str) -> str:
    """Collapse MATH-500 subject strings to coarse topic strata."""
    s = str(raw).strip().lower()
    if "geometry" in s:
        return "geometry"
    if "number" in s:          # "Number Theory"
        return "number_theory"
    if "count" in s or "probab" in s:   # "Counting & Probability"
        return "counting"
    if "prealgebra" in s or "pre-algebra" in s:
        return "prealgebra"
    if "intermediate" in s:    # "Intermediate Algebra"
        return "intermediate_algebra"
    if "precalc" in s or "pre-calc" in s:
        return "precalculus"
    if "algebra" in s:
        return "algebra"
    return s or "unknown"


def _extract_subjects(obj) -> list[str] | None:
    """Pull a (500,) list of subject strings out of a parsed JSON object."""
    if isinstance(obj, dict):
        for key in ("subjects", "subject", "topics", "topic", "type"):
            if key in obj and isinstance(obj[key], list):
                return [str(x) for x in obj[key]]
        # dict-of-records keyed by index
        if all(isinstance(v, dict) for v in obj.values()) and len(obj) == 500:
            items = sorted(obj.items(), key=lambda kv: int(kv[0]))
            return [str(_record_subject(v)) for _, v in items]
        return None
    if isinstance(obj, list):
        if all(isinstance(x, str) for x in obj):
            return [str(x) for x in obj]
        if all(isinstance(x, dict) for x in obj):
            out = [_record_subject(x) for x in obj]
            if all(s is not None for s in out):
                return [str(s) for s in out]
    return None


def _record_subject(rec: dict):
    for key in ("subject", "type", "topic", "category"):
        if key in rec:
            return rec[key]
    return None


def load_topics() -> np.ndarray | None:
    for path in TOPIC_CANDIDATES:
        if path.exists():
            try:
                obj = json.loads(path.read_text())
            except (ValueError, OSError):
                continue
            subs = _extract_subjects(obj)
            if subs is not None and len(subs) == 500:
                return np.array([_canonical_topic(s) for s in subs], dtype=object)
    for path in TOPIC_NPZ_CANDIDATES:
        if path.exists():
            try:
                blob = np.load(path, allow_pickle=True)
            except (ValueError, OSError):
                continue
            for key in ("subject", "subjects", "topic", "topics", "type"):
                if key in getattr(blob, "files", []):
                    arr = blob[key]
                    if arr.shape[0] == 500:
                        return np.array([_canonical_topic(s) for s in arr], dtype=object)
    return None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    topics = load_topics()
    if topics is None:
        print("MISSING_REGEN_INPUT topic_strata (no aligned MATH-500 subject "
              "metadata found in TOPIC_CANDIDATES)", file=sys.stderr)
        return 2

    # Usable topics: enough problems and both classes present.
    uniq = sorted(set(topics.tolist()))
    usable = []
    topic_counts = {}
    for t in uniq:
        m = topics == t
        n = int(m.sum())
        n_pos = int(y[m].sum())
        topic_counts[t] = {"n": n, "n_correct": n_pos, "n_wrong": n - n_pos}
        if n >= MIN_PER_TOPIC and n_pos >= MIN_PER_CLASS and (n - n_pos) >= MIN_PER_CLASS:
            usable.append(t)

    if len(usable) < 2:
        print("MISSING_REGEN_INPUT insufficient usable topic strata", file=sys.stderr)
        return 2

    # In-domain reference: pooled stratified 5-fold OOF DoM AUROC.
    folds = stratified_kfold(y, N_FOLDS, SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    n = len(y)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        ytr = y[train_mask]; Xtr = X[train_mask]
        d = Xtr[ytr].mean(0) - Xtr[~ytr].mean(0)
        oof[test_idx] = X[test_idx] @ d
    in_domain_oof_auroc = auroc(oof, y)

    # Within-topic OOF AUROC (per-topic in-domain ceiling) for context.
    within_topic = {}
    for t in usable:
        idx = np.flatnonzero(topics == t)
        yt = y[idx]; Xt = X[idx]
        tf = stratified_kfold(yt, min(N_FOLDS, int(min(yt.sum(), (~yt).sum()))), SEED)
        oof_t = np.zeros(len(idx), dtype=np.float64)
        for te in tf:
            tr = np.ones(len(idx), dtype=bool); tr[te] = False
            ytr = yt[tr]
            d = Xt[tr][ytr].mean(0) - Xt[tr][~ytr].mean(0)
            oof_t[te] = Xt[te] @ d
        within_topic[t] = float(auroc(oof_t, yt))

    # CCGP: train DoM on one topic, test on each other topic.
    cross_pairs = {}
    cross_vals = []
    for tr_t in usable:
        tr_idx = np.flatnonzero(topics == tr_t)
        ytr = y[tr_idx]
        d = X[tr_idx][ytr].mean(0) - X[tr_idx][~ytr].mean(0)
        for te_t in usable:
            if te_t == tr_t:
                continue
            te_idx = np.flatnonzero(topics == te_t)
            a = auroc(X[te_idx] @ d, y[te_idx])
            cross_pairs[f"{tr_t}->{te_t}"] = a
            if not np.isnan(a):
                cross_vals.append(a)

    mean_cross = float(np.mean(cross_vals)) if cross_vals else float("nan")
    mean_within = float(np.mean(list(within_topic.values()))) if within_topic else float("nan")

    out = {
        "experiment": "P11-FE155",
        "description": "CCGP for prefill L19 DoM across MATH-500 topic strata",
        "n_problems": int(n),
        "topic_counts": topic_counts,
        "usable_topics": usable,
        "in_domain_oof_auroc": float(in_domain_oof_auroc),
        "within_topic_oof_auroc": within_topic,
        "mean_within_topic_auroc": mean_within,
        "cross_topic_pairs": cross_pairs,
        "mean_cross_topic_auroc": mean_cross,
        "ccgp_degradation_vs_pooled_oof": float(in_domain_oof_auroc - mean_cross)
        if not np.isnan(mean_cross) else float("nan"),
        "ccgp_degradation_vs_within_topic": float(mean_within - mean_cross)
        if not (np.isnan(mean_cross) or np.isnan(mean_within)) else float("nan"),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())