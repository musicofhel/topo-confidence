"""P11-FE713 — Confidence-variance trajectory feature vs L19 prefill DoM.

Tests whether F-9's "trajectory features are redundant with L19 DoM" claim
(established for CoE-60) generalises to a confidence-*variance* trajectory
feature, as ReBalance argues it should not.

Procedure:
  1. Load cached Stage 2 generation traces (per-step token confidence) for all
     500 MATH-500 problems (Qwen-2.5-1.5B-Instruct, K=1).
  2. For each problem compute a sliding-window (W=2) variance over the per-step
     confidence sequence, then aggregate to a per-problem summary:
       - mean of window variances
       - max of window variances
       - fraction of steps in each of 3 variance buckets
  3. Fit OOF (5-fold stratified) logistic regression on the variance features
     for K=1 correctness; compute AUROC.
  4. Compare against L19 prefill DoM (raw + OOF logistic) and an
     L19-DoM + variance ensemble.
  5. Decision: does variance add >= 0.02 AUROC over L19 DoM alone?
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
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
TRACE_DIR = ROOT / "pathway11_h100/data/stage2_traces"
OUT_JSON = ROOT / "pathway11_h100/confidence_variance/results.json"

N_FOLDS = 5
SEED = 9999
WINDOW = 2
DOM_F2_REFERENCE = 0.7731  # F-2 L19 prefill DoM AUROC (1024tok, OOF 5-fold)
DELTA_THRESHOLD = 0.02

# Candidate NPZ keys that may hold per-step confidence / log-prob sequences.
CONF_KEYS = (
    "step_confidence", "confidence", "step_conf",
    "token_logprobs", "step_logprobs", "logprobs", "step_logprob",
)
# Variance-bucket edges (population var of a 2-point prob window lives in [0, 0.25]).
BUCKET_EDGES = (0.0, 0.01, 0.05, np.inf)
N_BUCKETS = len(BUCKET_EDGES) - 1


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def _to_confidence(arr: np.ndarray) -> np.ndarray:
    """Coerce a per-step trace to a confidence (probability) sequence."""
    arr = np.asarray(arr, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr
    # Log-probs are <= 0; exponentiate them into probabilities.
    if np.any(arr < 0.0):
        arr = np.exp(arr)
    return np.clip(arr, 0.0, 1.0)


def _load_conf(path: Path) -> np.ndarray | None:
    blob = np.load(path, allow_pickle=True)
    for key in CONF_KEYS:
        if key in blob.files:
            return _to_confidence(blob[key])
    return None


def window_variance_features(conf: np.ndarray) -> np.ndarray:
    """Per-problem summary of sliding W-window variances over a confidence seq.

    Returns [mean_var, max_var, frac_bucket_0, ..., frac_bucket_{N-1}].
    """
    feats = np.zeros(2 + N_BUCKETS, dtype=np.float64)
    if conf is None or conf.size < WINDOW:
        return feats
    # Sliding population variance over each contiguous window of size WINDOW.
    win_vars = np.array(
        [conf[i:i + WINDOW].var() for i in range(conf.size - WINDOW + 1)],
        dtype=np.float64,
    )
    if win_vars.size == 0:
        return feats
    feats[0] = float(win_vars.mean())
    feats[1] = float(win_vars.max())
    counts = np.zeros(N_BUCKETS, dtype=np.float64)
    for b in range(N_BUCKETS):
        lo, hi = BUCKET_EDGES[b], BUCKET_EDGES[b + 1]
        if b == N_BUCKETS - 1:
            counts[b] = np.sum((win_vars >= lo) & (win_vars <= hi))
        else:
            counts[b] = np.sum((win_vars >= lo) & (win_vars < hi))
    feats[2:] = counts / max(win_vars.size, 1)
    return feats


def oof_logistic_auroc(X: np.ndarray, y: np.ndarray) -> float:
    """OOF 5-fold stratified logistic-regression AUROC on feature matrix X."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd < 1e-12] = 1.0
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr_s, y[train_idx])
        oof[test_idx] = clf.decision_function(Xte_s)
    return float(auroc(oof, y))


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2
    if not TRACE_DIR.exists():
        print("MISSING_REGEN_INPUT", TRACE_DIR, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    correct = cache["correct"].astype(bool)
    n = len(correct)
    assert n == 500, f"expected 500 problems, got {n}"

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert dom_score.shape == (n,)

    # Load per-problem Stage 2 traces and build the variance feature matrix.
    var_feats = np.zeros((n, 2 + N_BUCKETS), dtype=np.float64)
    found = 0
    missing = []
    for i in range(n):
        path = TRACE_DIR / f"problem_{i:03d}.npz"
        if not path.exists():
            missing.append(i)
            continue
        conf = _load_conf(path)
        if conf is None:
            missing.append(i)
            continue
        var_feats[i] = window_variance_features(conf)
        found += 1

    if found == 0:
        print("MISSING_REGEN_INPUT", TRACE_DIR, "(no readable traces)", file=sys.stderr)
        return 2

    feat_names = ["mean_var", "max_var"] + [f"frac_bucket_{b}" for b in range(N_BUCKETS)]

    # AUROCs.
    auroc_dom_raw = auroc(dom_score, correct)
    auroc_dom_oof = oof_logistic_auroc(dom_score.reshape(-1, 1), correct)
    auroc_var_oof = oof_logistic_auroc(var_feats, correct)
    ensemble_X = np.column_stack([dom_score, var_feats])
    auroc_ensemble_oof = oof_logistic_auroc(ensemble_X, correct)

    delta_vs_dom_oof = auroc_ensemble_oof - auroc_dom_oof
    delta_vs_f2 = auroc_ensemble_oof - DOM_F2_REFERENCE

    out = {
        "experiment": "P11-FE713",
        "n_problems": int(n),
        "n_traces_found": int(found),
        "n_traces_missing": int(len(missing)),
        "window": WINDOW,
        "bucket_edges": [float(e) if np.isfinite(e) else "inf" for e in BUCKET_EDGES],
        "feature_names": feat_names,
        "dom_f2_reference": DOM_F2_REFERENCE,
        "delta_threshold": DELTA_THRESHOLD,
        "auroc_dom_raw": auroc_dom_raw,
        "auroc_dom_oof": auroc_dom_oof,
        "auroc_variance_oof": auroc_var_oof,
        "auroc_ensemble_oof": auroc_ensemble_oof,
        "delta_ensemble_minus_dom_oof": float(delta_vs_dom_oof),
        "delta_ensemble_minus_f2": float(delta_vs_f2),
        "variance_adds_signal_vs_dom_oof": bool(delta_vs_dom_oof >= DELTA_THRESHOLD),
        "variance_adds_signal_vs_f2": bool(delta_vs_f2 >= DELTA_THRESHOLD),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())