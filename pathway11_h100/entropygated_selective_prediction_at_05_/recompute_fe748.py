"""P11-FE748 — Entropy-gated selective prediction at 0.5 coverage.

Refutation #4 for F-8 (71.6% answered-set accuracy via prefill-DoM gating at
0.5 coverage). Rank the 500 MATH-500 traces by mean per-token entropy (low
entropy = high confidence), answer the most-confident half, abstain on the
rest, and measure answered-set accuracy. If entropy alone reaches the same
71.6% frontier, F-8's mechanism attribution shifts from "DoM-driven" to
"entropy-driven, DoM coincident," supporting H-12 ("SEP replaces DoM").

The mean per-token entropy is a P11-FE74 prerequisite; this script reads it
from a cached NPZ and never recomputes it on-GPU. For control, the same gate
is also run on prefill-DoM scores (the F-8 mechanism) and on a difficulty
stratification derived from the K=8 self-consistency pass-rate, since TIP
predicts entropy fails on the hard (D) bucket.
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
ENTROPY_NPZ = ROOT / "pathway11_h100/entropy_gated_selective/fe74_mean_entropy.npz"
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
OUT_JSON = ROOT / "pathway11_h100/entropy_gated_selective/results.json"

COVERAGE = 0.5
F8_DOM_REFERENCE = 0.716  # F-8 answered-set accuracy at 0.5 coverage (1024tok)
ENTROPY_KEYS = ("mean_entropy", "mean_token_entropy", "entropy", "prefill_entropy")
N_DIFF_BUCKETS = 4


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def load_entropy() -> np.ndarray:
    """Mean per-token entropy from the FE74 cache; first matching key wins."""
    blob = np.load(ENTROPY_NPZ)
    for key in ENTROPY_KEYS:
        if key in blob.files:
            return blob[key].astype(np.float64)
    raise KeyError(f"no entropy key in {ENTROPY_NPZ.name}; have {blob.files}")


def load_k8_passrate() -> np.ndarray | None:
    """Per-problem K=8 pass-rate (difficulty proxy); None if cache absent."""
    if not K8_DIR.is_dir():
        return None
    rates = np.full(500, np.nan, dtype=np.float64)
    found = 0
    for i in range(500):
        f = K8_DIR / f"problem_{i:03d}.npz"
        if not f.exists():
            continue
        c = np.load(f)["correct"].astype(bool)
        rates[i] = float(c.mean())
        found += 1
    if found == 0:
        return None
    return rates


def gate_answered_acc(confidence: np.ndarray, correct: np.ndarray,
                      coverage: float) -> dict:
    """Answer the most-confident `coverage` fraction; report answered accuracy.

    `confidence` is higher = more confident. Ranks descending and keeps the
    top-`coverage` as the answered set.
    """
    n = len(correct)
    n_ans = int(round(coverage * n))
    order = np.argsort(-confidence, kind="stable")  # most confident first
    answered = order[:n_ans]
    abstained = order[n_ans:]
    ans_acc = float(correct[answered].mean()) if n_ans else float("nan")
    abs_acc = (float(correct[abstained].mean())
               if len(abstained) else float("nan"))
    return {
        "coverage": float(n_ans / n),
        "n_answered": int(n_ans),
        "answered_accuracy": ans_acc,
        "abstained_accuracy": abs_acc,
    }


def main() -> int:
    for path in (CACHE, DOM_NPZ, ENTROPY_NPZ):
        if not path.exists():
            print("MISSING_REGEN_INPUT", path, file=sys.stderr)
            return 2

    cache = np.load(CACHE)
    correct = cache["correct"].astype(bool)
    assert correct.shape == (500,)

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)

    try:
        entropy = load_entropy()
    except KeyError as e:
        print("MISSING_REGEN_INPUT", e, file=sys.stderr)
        return 2
    if entropy.shape != (500,):
        print("MISSING_REGEN_INPUT", f"entropy shape {entropy.shape}",
              file=sys.stderr)
        return 2

    overall_acc = float(correct.mean())

    # Lower entropy = more confident, so confidence = -entropy.
    entropy_gate = gate_answered_acc(-entropy, correct, COVERAGE)
    dom_gate = gate_answered_acc(dom_score, correct, COVERAGE)

    # AUROC of each predictor over the full set (sanity / mechanism overlap).
    auroc_entropy = auroc(-entropy, correct)
    auroc_dom = auroc(dom_score, correct)
    # Correlation of the two confidence signals (coincidence vs causation).
    es = -entropy
    es = (es - es.mean()) / (es.std() + 1e-12)
    ds = (dom_score - dom_score.mean()) / (dom_score.std() + 1e-12)
    corr_entropy_dom = float((es * ds).mean())

    # Coverage sweep for the entropy gate (frontier comparison).
    sweep = []
    order = np.argsort(entropy, kind="stable")  # ascending entropy
    for cov in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        k = int(round(cov * 500))
        if k == 0:
            continue
        sweep.append({
            "coverage": float(k / 500),
            "answered_accuracy": float(correct[order[:k]].mean()),
        })

    # Difficulty (D-bucket) stratification via K=8 pass-rate, if available.
    diff_buckets = None
    passrate = load_k8_passrate()
    if passrate is not None and np.isfinite(passrate).all():
        # Bucket 0 = hardest (lowest pass-rate) = D-bucket per TIP.
        bucket_idx = np.argsort(passrate, kind="stable")
        groups = np.array_split(bucket_idx, N_DIFF_BUCKETS)
        diff_buckets = []
        for b, idx in enumerate(groups):
            c = correct[idx]
            diff_buckets.append({
                "bucket": int(b),
                "label": "D" if b == 0 else f"Q{b}",
                "n": int(len(idx)),
                "base_accuracy": float(c.mean()),
                "mean_passrate": float(passrate[idx].mean()),
                "entropy_gate": gate_answered_acc(-entropy[idx], c, COVERAGE),
                "dom_gate": gate_answered_acc(dom_score[idx], c, COVERAGE),
            })

    refutes_f8 = bool(entropy_gate["answered_accuracy"] >= F8_DOM_REFERENCE)

    out = {
        "experiment": "P11-FE748",
        "coverage_target": COVERAGE,
        "overall_accuracy": overall_acc,
        "f8_dom_reference": F8_DOM_REFERENCE,
        "entropy_gate_at_0.5": entropy_gate,
        "dom_gate_at_0.5": dom_gate,
        "auroc_neg_entropy": auroc_entropy,
        "auroc_dom": auroc_dom,
        "corr_neg_entropy_dom": corr_entropy_dom,
        "entropy_coverage_sweep": sweep,
        "difficulty_buckets": diff_buckets,
        "entropy_meets_f8_frontier": refutes_f8,
        "verdict": (
            "ENTROPY_SUFFICIENT" if refutes_f8 else "DOM_RETAINS_EDGE"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())