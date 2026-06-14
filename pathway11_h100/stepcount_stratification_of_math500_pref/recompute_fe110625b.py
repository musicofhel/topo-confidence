"""P11-FE110625B — Step-count stratification of MATH-500 prefill DoM AUROC.

Tests the paper's single-step vs multi-step boundary: Math Ops (single-step)
prefill-DoM AUROC transfers cleanly (0.782–0.858) while GSM8K (multi-step)
collapses (0.499–0.601). Here we stratify the 500 MATH-500 problems by
GPT-4-judged step count into bins (1, 2–3, 4–5, 6+) and recompute the L19
prefill DoM AUROC within each bin. A monotonic drop with step count would
replicate the "multi-step kills the signal" claim and kill H-5's familiarity
framing.

NETWORK CONSTRAINT: GPT-4 step-count labelling cannot run in-process (no
network). The 500 per-problem step counts must be pre-computed and cached to
one of STEP_LABEL_CANDIDATES as either a JSON list/dict of length 500 or an
NPZ with a (500,) integer "step_count" array. If absent we emit
MISSING_REGEN_INPUT and return 2.
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
OUT_JSON = ROOT / "pathway11_h100/step_stratification/results.json"

# Pre-computed GPT-4 step-count labels (one of these must exist).
STEP_LABEL_CANDIDATES = [
    ROOT / "pathway11_h100/step_stratification/step_counts.npz",
    ROOT / "pathway11_h100/step_stratification/step_counts.json",
    ROOT / "pathway11_h100/exp1_cross_model/qwen2.5-1.5b/step_counts.npz",
    ROOT / "pathway11_h100/exp1_cross_model/qwen2.5-1.5b/step_counts.json",
]

N_FOLDS = 5
SEED = 9999

# (label, inclusive-min, inclusive-max) for step-count bins; np.inf = open.
BINS = [
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-5", 4, 5),
    ("6+", 6, np.inf),
]


def auroc(scores, labels):
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def load_step_counts(n: int):
    """Return a (n,) int array of step counts, or None if no cache found."""
    for path in STEP_LABEL_CANDIDATES:
        if not path.exists():
            continue
        if path.suffix == ".npz":
            blob = np.load(path)
            key = "step_count" if "step_count" in blob.files else blob.files[0]
            sc = np.asarray(blob[key]).astype(np.int64).ravel()
        else:
            obj = json.loads(path.read_text())
            if isinstance(obj, dict):
                # keys may be string indices "0".."499"
                sc = np.array([int(obj[str(i)]) for i in range(n)], dtype=np.int64)
            else:
                sc = np.asarray(obj, dtype=np.int64).ravel()
        if sc.shape[0] != n:
            print(f"BAD_STEP_LABEL_SHAPE {path} {sc.shape}", file=sys.stderr)
            continue
        return sc, path
    return None, None


def oof_dom_scores(X, y):
    """5-fold OOF DoM projection: fit class-mean-difference on train, score test."""
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for train_idx, test_idx in skf.split(X, y):
        ytr = y[train_idx]
        d = X[train_idx][ytr].mean(axis=0) - X[train_idx][~ytr].mean(axis=0)
        scores[test_idx] = X[test_idx] @ d
    return scores


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)
    n = len(y)

    step_counts, label_src = load_step_counts(n)
    if step_counts is None:
        print("MISSING_REGEN_INPUT", "step_counts (GPT-4 labels)", file=sys.stderr)
        return 2

    # Recompute OOF prefill DoM scores once; stratify the scores by bin.
    oof = oof_dom_scores(X, y)
    overall_auroc = auroc(oof, y)

    # Sanity cross-check against the cached precomputed DoM score, if present.
    cached_dom_auroc = None
    if DOM_NPZ.exists():
        cached = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        cached_dom_auroc = auroc(cached, y)

    per_bin = []
    monotonic_seq = []
    for label, lo, hi in BINS:
        mask = (step_counts >= lo) & (step_counts <= hi)
        n_bin = int(mask.sum())
        n_pos = int(y[mask].sum())
        n_neg = int(n_bin - n_pos)
        bin_auroc = auroc(oof[mask], y[mask]) if n_bin > 0 else float("nan")
        per_bin.append({
            "bin": label,
            "step_min": lo,
            "step_max": (None if np.isinf(hi) else hi),
            "n": n_bin,
            "n_correct": n_pos,
            "n_incorrect": n_neg,
            "auroc_oof_dom": bin_auroc,
        })
        if n_bin > 0 and not np.isnan(bin_auroc):
            monotonic_seq.append(bin_auroc)

    # Monotone-decreasing test across bins with valid AUROCs.
    monotonic_decreasing = bool(
        len(monotonic_seq) >= 2
        and all(monotonic_seq[i] >= monotonic_seq[i + 1] for i in range(len(monotonic_seq) - 1))
    )
    auroc_drop = (
        float(monotonic_seq[0] - monotonic_seq[-1]) if len(monotonic_seq) >= 2 else None
    )

    out = {
        "experiment": "P11-FE110625B",
        "description": "Step-count stratified prefill DoM AUROC (MATH-500, qwen2.5-1.5b L19)",
        "n_folds": N_FOLDS,
        "seed": SEED,
        "step_label_source": str(label_src),
        "step_count_min": int(step_counts.min()),
        "step_count_max": int(step_counts.max()),
        "step_count_mean": float(step_counts.mean()),
        "overall_auroc_oof_dom": overall_auroc,
        "cached_dom_auroc_crosscheck": cached_dom_auroc,
        "bins": per_bin,
        "monotonic_decreasing_with_steps": monotonic_decreasing,
        "auroc_drop_first_to_last_bin": auroc_drop,
        "multistep_kills_signal_replicated": bool(
            monotonic_decreasing and auroc_drop is not None and auroc_drop >= 0.10
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"overall_auroc={overall_auroc:.4f} monotonic={monotonic_decreasing} drop={auroc_drop}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())