"""P11-FE722 — Length+latex surface-feature null-control for F-8 selective prediction.

Paper 2603.18280 critiques bare probe accuracy; F-8's selective-prediction stack
(71.6% answered accuracy at coverage 0.5, K=2.5 avg compute) inherits that risk.
This is the null control: a logistic regression on FOUR problem-surface features
only — [problem_length, latex_token_count, digit_count, log1p(numerical_constant_count)] —
with ZERO hidden states. We compute the selective-prediction accuracy curve at
coverages 0.3 / 0.5 / 0.7 using out-of-fold predicted correctness probability as
the routing confidence, and report Δ vs F-8's 71.6% at coverage 0.5. If this
surface-only null reaches ≥65% at coverage 0.5, F-8's geometric routing premium
is overstated.

No regex (re) used: latex/digit/number features are computed by manual char
scanning to stay within the allowed-import constraint.
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
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
OUT_JSON = ROOT / "pathway11_h100/length_latex_null/results.json"

N = 500
SEED = 9999
N_FOLDS = 5
COVERAGES = [0.3, 0.5, 0.7]
F8_ANSWERED_ACC = 0.716  # F-8 selective-prediction acc at coverage 0.5 (1024tok)

# Candidate locations for the MATH-500 problem texts (index-aligned to `correct`).
PROBLEM_CANDIDATES = [
    ROOT / "data/math500.jsonl",
    ROOT / "data/math500.json",
    ROOT / "data/math500_problems.jsonl",
    ROOT / "data/math500_problems.json",
    ROOT / "data/MATH-500.jsonl",
    ROOT / "data/MATH500.jsonl",
    ROOT / "data/test.jsonl",
    ROOT / "configs/math500.jsonl",
]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def _extract_problem(obj) -> str | None:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("problem", "question", "prompt", "text", "Problem"):
            if key in obj and isinstance(obj[key], str):
                return obj[key]
    return None


def load_problems() -> list[str] | None:
    """Return the index-aligned list of >=500 MATH-500 problem texts, or None."""
    files = list(PROBLEM_CANDIDATES)
    data_dir = ROOT / "data"
    if data_dir.is_dir():
        for p in sorted(data_dir.glob("**/*.json*")):
            name = p.name.lower()
            if "math" in name and "500" in name and p not in files:
                files.append(p)

    for path in files:
        if not path.exists():
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        problems: list[str] = []
        if path.suffix == ".jsonl":
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    problems = []
                    break
                prob = _extract_problem(obj)
                if prob is None:
                    problems = []
                    break
                problems.append(prob)
        else:
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            seq = obj if isinstance(obj, list) else obj.get("data") or obj.get("problems") or []
            for item in seq:
                prob = _extract_problem(item)
                if prob is None:
                    problems = []
                    break
                problems.append(prob)
        if len(problems) >= N:
            return problems[:N]
    return None


def surface_features(problems: list[str]) -> np.ndarray:
    """[problem_length, latex_token_count, digit_count, log1p(num_constant_count)]."""
    feats = np.zeros((len(problems), 4), dtype=np.float64)
    for i, text in enumerate(problems):
        problem_length = len(text)
        latex_count = text.count("\\")  # backslash-initiated LaTeX commands
        digit_count = 0
        num_constant_count = 0
        prev_digit = False
        for ch in text:
            is_digit = ch.isdigit()
            if is_digit:
                digit_count += 1
                if not prev_digit:
                    num_constant_count += 1
            prev_digit = is_digit
        feats[i, 0] = problem_length
        feats[i, 1] = latex_count
        feats[i, 2] = digit_count
        feats[i, 3] = float(np.log1p(num_constant_count))
    return feats


def oof_proba(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Out-of-fold positive-class probability from standardized logistic regression."""
    proba = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for train_idx, test_idx in skf.split(X, y):
        mu = X[train_idx].mean(axis=0)
        sd = X[train_idx].std(axis=0)
        sd[sd < 1e-12] = 1.0
        Xtr = (X[train_idx] - mu) / sd
        Xte = (X[test_idx] - mu) / sd
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[train_idx])
        proba[test_idx] = clf.predict_proba(Xte)[:, 1]
    return proba


def selective_curve(confidence: np.ndarray, correct: np.ndarray) -> dict:
    """Answer the top-`coverage` fraction by confidence; report answered accuracy."""
    order = np.argsort(-confidence, kind="stable")
    out = {}
    for c in COVERAGES:
        n_answer = max(1, int(round(c * len(correct))))
        answered = order[:n_answer]
        out[f"{c:.1f}"] = {
            "n_answered": int(n_answer),
            "answered_accuracy": float(correct[answered].mean()),
        }
    return out


def k8_majority_correct() -> np.ndarray | None:
    if not K8_DIR.is_dir():
        return None
    maj = np.zeros(N, dtype=bool)
    for i in range(N):
        f = K8_DIR / f"problem_{i:03d}.npz"
        if not f.exists():
            return None
        c = np.load(f)["correct"].astype(bool)
        maj[i] = c.mean() >= 0.5
    return maj


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    problems = load_problems()
    if problems is None:
        print("MISSING_REGEN_INPUT", "MATH-500 problem texts", file=sys.stderr)
        return 2

    correct = np.load(CACHE)["correct"].astype(bool)
    assert correct.shape == (N,), f"unexpected correct shape {correct.shape}"

    X = surface_features(problems)
    y = correct

    surface_proba = oof_proba(X, y)
    surface_auroc = auroc(surface_proba, y)

    curve_k1 = selective_curve(surface_proba, correct)
    null_acc_at_50 = curve_k1["0.5"]["answered_accuracy"]
    delta_vs_f8 = null_acc_at_50 - F8_ANSWERED_ACC

    out = {
        "experiment": "P11-FE722",
        "description": "Length+latex surface-feature null-control for F-8 selective prediction",
        "features": [
            "problem_length",
            "latex_token_count",
            "digit_count",
            "log1p(numerical_constant_count)",
        ],
        "n_problems": int(N),
        "n_correct_k1": int(correct.sum()),
        "surface_only_auroc_oof": float(surface_auroc),
        "selective_curve_k1": curve_k1,
        "f8_answered_acc_at_coverage_0.5": F8_ANSWERED_ACC,
        "null_answered_acc_at_coverage_0.5": float(null_acc_at_50),
        "delta_vs_f8_at_coverage_0.5": float(delta_vs_f8),
        "null_reaches_65pct_at_0.5": bool(null_acc_at_50 >= 0.65),
        "verdict": (
            "GEOMETRIC_PREMIUM_SHRINKS" if null_acc_at_50 >= 0.65 else "GEOMETRIC_PREMIUM_HOLDS"
        ),
    }

    maj = k8_majority_correct()
    if maj is not None:
        # Re-route the same surface confidence against the K=8 majority-vote labels,
        # the closer analogue to F-8's multi-sample answered accuracy.
        maj_proba = oof_proba(X, maj)
        out["k8_majority_available"] = True
        out["n_correct_k8_majority"] = int(maj.sum())
        out["surface_only_auroc_oof_k8maj"] = float(auroc(maj_proba, maj))
        out["selective_curve_k8_majority"] = selective_curve(maj_proba, maj)
    else:
        out["k8_majority_available"] = False

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())