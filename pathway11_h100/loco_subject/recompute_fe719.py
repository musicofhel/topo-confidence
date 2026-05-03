"""FE719 — Leave-One-Subject-Out CV on F-2 (prefill L19 DoM AUROC, 1.5B).

Refutation test: if mean AUROC drops below 0.65, F-2 is largely a
subject/topic detector, not a correctness signal.

Inputs (cached, deterministic):
  - pathway11_h100/prefill_inversion/cache/m15b_prefill.npz  (Phase 0(b))
  - HuggingFaceH4/MATH-500 test split (subject column)

Output: pathway11_h100/loco_subject/results.json
Stdout: key=value lines for Tier-1 regen (validate_claims.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/loco_subject/results.json"

SUBJECTS = [
    "Algebra",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Number Theory",
    "Prealgebra",
    "Precalculus",
]


def load_subjects() -> np.ndarray:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    subj = np.array(ds["subject"])
    assert subj.shape == (500,), f"unexpected MATH-500 size: {subj.shape}"
    return subj


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U AUROC. Ties contribute 0.5."""
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def loco_dom(X: np.ndarray, y: np.ndarray, subj: np.ndarray) -> dict:
    """Leave-one-subject-out DoM. Return per-subject AUROC + summary stats."""
    per_subject = {}
    aurocs = []
    for held in SUBJECTS:
        mask_test = subj == held
        mask_train = ~mask_test
        if mask_test.sum() == 0:
            per_subject[held] = {"auroc": None, "n_test": 0, "n_pos": 0, "n_neg": 0}
            continue
        if y[mask_train].sum() == 0 or (~y[mask_train]).sum() == 0:
            per_subject[held] = {
                "auroc": None,
                "n_test": int(mask_test.sum()),
                "n_pos": int(y[mask_test].sum()),
                "n_neg": int((~y[mask_test]).sum()),
                "skip_reason": "train fold is single-class",
            }
            continue
        mu_pos = X[mask_train & y].mean(axis=0)
        mu_neg = X[mask_train & ~y].mean(axis=0)
        dom = mu_pos - mu_neg
        scores_test = X[mask_test] @ dom
        n_pos = int(y[mask_test].sum())
        n_neg = int((~y[mask_test]).sum())
        if n_pos == 0 or n_neg == 0:
            a = None
            note = "test fold is single-class"
        else:
            a = auroc(scores_test, y[mask_test])
            note = ""
        per_subject[held] = {
            "auroc": a,
            "n_test": int(mask_test.sum()),
            "n_pos": n_pos,
            "n_neg": n_neg,
        }
        if note:
            per_subject[held]["skip_reason"] = note
        if a is not None:
            aurocs.append(a)
    aurocs_arr = np.array(aurocs, dtype=float)
    return {
        "per_subject": per_subject,
        "auroc_mean": float(aurocs_arr.mean()) if len(aurocs_arr) else None,
        "auroc_worst": float(aurocs_arr.min()) if len(aurocs_arr) else None,
        "auroc_best": float(aurocs_arr.max()) if len(aurocs_arr) else None,
        "auroc_std": float(aurocs_arr.std(ddof=0)) if len(aurocs_arr) else None,
        "n_folds_scored": int(len(aurocs_arr)),
    }


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    d = np.load(CACHE)
    X = d["prefill"].astype(np.float32)
    y = d["correct"].astype(bool)
    assert X.shape == (500, 1536)
    assert y.shape == (500,)

    subj = load_subjects()

    out = loco_dom(X, y, subj)

    # Anchor: random-fold AUROC on the same cache, for comparison with the
    # 0.7731 figure quoted in headlines (which is 5-fold OOF; this is a
    # full-data DoM with no held-out, so AUROCs aren't directly comparable —
    # we add it only for sanity).
    mu_pos_all = X[y].mean(axis=0)
    mu_neg_all = X[~y].mean(axis=0)
    dom_all = mu_pos_all - mu_neg_all
    in_sample_auroc = auroc(X @ dom_all, y)
    out["in_sample_full_auroc"] = in_sample_auroc

    # Refutation test
    out["f2_refutation_threshold"] = 0.65
    out["f2_refuted"] = (
        out["auroc_mean"] is not None and out["auroc_mean"] < 0.65
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    # Tier-1 regen lines
    print(f"loco_auroc_mean={out['auroc_mean']:.10f}")
    print(f"loco_auroc_worst={out['auroc_worst']:.10f}")
    print(f"loco_auroc_best={out['auroc_best']:.10f}")
    print(f"loco_n_folds={out['n_folds_scored']}")
    print(f"in_sample_full_auroc={out['in_sample_full_auroc']:.10f}")
    for s in SUBJECTS:
        ps = out["per_subject"][s]
        if ps["auroc"] is not None:
            key = "loco_" + s.lower().replace(" & ", "_").replace(" ", "_")
            print(f"{key}_auroc={ps['auroc']:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
