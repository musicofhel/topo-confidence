"""P11-FE581 — Output-side aggregates (NuclearNorm / IM / ClassEntropy) on the
per-problem K=8 outcome matrix vs F-2's L19 DoM probe.

Unsupervised baseline against the program's headline supervised number. For each
of the 500 MATH-500 problems we have K=8 self-consistency samples (correct/incorrect
per sample). We treat the per-problem K=8 outcomes as a one-hot "answer matrix"
M ∈ {0,1}^(8×2) (correct vs incorrect column) and compute three zero-training
output-side scores from the NuclearNorm OOD-detection literature:

  - NuclearNorm  = sum of singular values of M  (= sqrt(c) + sqrt(8-c))
  - IM           = inverse margin 1 - |p_correct - p_incorrect|
  - ClassEntropy = H([c/8, (8-c)/8])

All three are unsupervised consensus/uncertainty signals (high = disagreement).
We score each problem and report the AUROC for predicting K=1 correctness
(main-cache `correct`), oriented to >0.5 since the metrics predict *in*correctness.
Compared head-to-head against F-2's L19 DoM (0.7731 OOF). If any metric reaches
AUROC >= 0.74, output-side aggregates are competitive with hidden-state probes.

NOTE on data availability: the K=8 cache exposes only per-sample `correct`
booleans (not per-class softmax logits), so the answer matrix is the binary
outcome matrix — its singular-value / margin / entropy structure collapses to
functions of the correct-count c. Per-token softmax re-extraction (the ~30min
path in the FE description) requires a GPU and is out of scope for this CPU
recompute; only 1.5B has a committed prefill/K=8 cache, so 7B is reported as
unavailable rather than fabricated.
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
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
OUT_JSON = ROOT / "pathway11_h100/k8_output_aggregates/results.json"

N_PROBLEMS = 500
DOM_OOF_REF = 0.7731  # F-2 L19 DoM OOF AUROC, the supervised number to beat
COMPETITIVE_THRESHOLD = 0.74


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def directed_auroc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, str]:
    """Return (max(a, 1-a), orientation). Unsupervised metrics have arbitrary sign."""
    a = auroc(scores, labels)
    if np.isnan(a):
        return float("nan"), "nan"
    if a >= 0.5:
        return a, "raw"
    return 1.0 - a, "negated"


def _load_k8_correct(idx: int) -> np.ndarray | None:
    """Locate problem_NNN.npz for problem idx (try 0- and 1-indexed naming)."""
    for name in (f"problem_{idx:03d}.npz", f"problem_{idx + 1:03d}.npz"):
        p = K8_DIR / name
        if p.exists():
            try:
                blob = np.load(p)
            except Exception:
                return None
            if "correct" in blob.files:
                return np.asarray(blob["correct"]).astype(bool).ravel()
    return None


def _problem_metrics(correct8: np.ndarray) -> tuple[float, float, float]:
    """NuclearNorm, IM, ClassEntropy of the K=8 binary outcome matrix."""
    n = int(correct8.size)
    c = int(correct8.sum())
    # One-hot answer matrix M (n x 2): column 0 = correct, column 1 = incorrect.
    M = np.zeros((n, 2), dtype=np.float64)
    M[correct8, 0] = 1.0
    M[~correct8, 1] = 1.0
    svals = np.linalg.svd(M, compute_uv=False)
    nuclear_norm = float(svals.sum())

    p_correct = c / n if n else 0.0
    p_incorrect = 1.0 - p_correct
    margin = abs(p_correct - p_incorrect)
    im = float(1.0 - margin)

    ent = 0.0
    for p in (p_correct, p_incorrect):
        if p > 0.0:
            ent -= p * np.log(p)
    class_entropy = float(ent)
    return nuclear_norm, im, class_entropy


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not K8_DIR.exists():
        print("MISSING_REGEN_INPUT", K8_DIR, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    correct_k1 = cache["correct"].astype(bool)
    assert correct_k1.shape == (N_PROBLEMS,), correct_k1.shape

    dom_score = None
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)

    nuc = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    im = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    ent = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    n_loaded = 0
    for i in range(N_PROBLEMS):
        c8 = _load_k8_correct(i)
        if c8 is None or c8.size == 0:
            continue
        nuc[i], im[i], ent[i] = _problem_metrics(c8)
        n_loaded += 1

    if n_loaded == 0:
        print("MISSING_REGEN_INPUT", K8_DIR, "(no problem_NNN.npz loaded)", file=sys.stderr)
        return 2

    valid = ~np.isnan(nuc)
    y = correct_k1[valid]

    metrics = {}
    best_metric = None
    best_auroc = -1.0
    for name, arr in (("nuclear_norm", nuc), ("im", im), ("class_entropy", ent)):
        a, orient = directed_auroc(arr[valid], y)
        metrics[name] = {
            "auroc": a,
            "orientation": orient,
            "auroc_raw": auroc(arr[valid], y),
        }
        if not np.isnan(a) and a > best_auroc:
            best_auroc = a
            best_metric = name

    dom_auroc_raw = (
        auroc(dom_score[valid], y) if dom_score is not None else None
    )

    out = {
        "experiment": "P11-FE581",
        "model": "Qwen-2.5-1.5B",
        "model_7b": "UNAVAILABLE_NO_CACHE",
        "n_problems": N_PROBLEMS,
        "n_loaded": n_loaded,
        "k1_accuracy": float(y.mean()),
        "metrics": metrics,
        "best_metric": best_metric,
        "best_auroc": float(best_auroc),
        "dom_auroc_raw": dom_auroc_raw,
        "dom_oof_reference": DOM_OOF_REF,
        "competitive_threshold": COMPETITIVE_THRESHOLD,
        "output_aggregates_competitive": bool(best_auroc >= COMPETITIVE_THRESHOLD),
        "note": (
            "K=8 cache exposes only per-sample correctness, not per-class softmax "
            "logits; the answer matrix is the binary outcome matrix, so all three "
            "metrics collapse to functions of the correct-count. Per-token softmax "
            "re-extraction requires GPU and is out of scope for this CPU recompute."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())