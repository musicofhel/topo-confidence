"""P11-FE768 — MaxProb / token-entropy selective-prediction baseline at 50% coverage.

Patel et al. argue that 'pure-incorrectness' features are functionally inert and
that selective-prediction lift is really just confounded uncertainty filtering. If
F-8's 71.6% answered accuracy at 50% coverage is matched by a vanilla MaxProb
(mean-token-probability) confidence filter at the *same* coverage, then F-8's lift
is largely a confidence filter and DoM's correctness framing collapses into
uncertainty-filtering.

This script:
  1. Loads ground-truth K=1 correctness for the 500-problem F-8 MATH-500 set.
  2. Locates a cached token-probability / token-entropy log for the 1.5B
     completions (mean-token-prob, token-entropy, and optional semantic-entropy
     approximation from K=8 answer diversity).
  3. Ranks the 500 completions by each confidence signal, takes the top-250
     (coverage = 0.5), and reports answered accuracy.
  4. Compares against F-8's 71.6%.

If no token-probability log is present, the script prints MISSING_REGEN_INPUT and
returns 2 (the logits/logprobs are not part of the committed NPZ schema and must
be regenerated from the cached generations).
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
OUT_JSON = ROOT / "pathway11_h100/maxprob_baseline/results.json"

# F-8 headline: 71.6% answered accuracy at coverage 0.5 (selective-prediction stack).
F8_SELECTIVE_ACC_AT_HALF = 0.716
COVERAGE = 0.5

# Candidate caches that may carry per-completion token-probability signals.
# Keys we know how to consume: mean-token-prob (higher = more confident),
# token-entropy (lower = more confident), and raw per-token logprob matrices.
TOKENPROB_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_gated_compute/token_probs.npz",
    ROOT / "pathway11_h100/maxprob_baseline/token_probs.npz",
    ROOT / "pathway11_h100/data/m15b_token_logprobs.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_token_logprobs.npz",
    ROOT / "pathway11_h100/data/k1_token_logprobs.npz",
]

MEANPROB_KEYS = ("mean_token_prob", "meanprob", "mean_prob", "maxprob", "seq_prob")
MEANLOGPROB_KEYS = ("mean_token_logprob", "mean_logprob", "meanlogprob", "avg_logprob")
ENTROPY_KEYS = ("token_entropy", "mean_token_entropy", "entropy")
LOGPROB_MATRIX_KEYS = ("token_logprobs", "logprobs", "per_token_logprob")
TOKEN_MASK_KEYS = ("token_mask", "mask", "valid_mask")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def selective_accuracy(confidence: np.ndarray, correct: np.ndarray, coverage: float) -> dict:
    """Rank by confidence (descending), answer the top `coverage` fraction."""
    n = len(correct)
    k = int(round(coverage * n))
    order = np.argsort(-confidence, kind="stable")  # highest confidence first
    answered = order[:k]
    acc_answered = float(correct[answered].mean()) if k > 0 else float("nan")
    return {
        "coverage": coverage,
        "n_answered": int(k),
        "answered_accuracy": acc_answered,
        "auroc_vs_correct": float(auroc(confidence, correct)),
    }


def _find_key(blob, keys):
    for key in keys:
        if key in blob.files:
            return key
    return None


def load_token_signals(correct: np.ndarray):
    """Return dict of {signal_name: (confidence_array, note)} or None if unavailable.

    Confidence arrays are oriented so that HIGHER = more confident.
    """
    n = len(correct)
    src = next((p for p in TOKENPROB_CANDIDATES if p.exists()), None)
    if src is None:
        return None, None

    blob = np.load(src, allow_pickle=False)
    signals = {}

    key = _find_key(blob, MEANPROB_KEYS)
    if key is not None:
        v = np.asarray(blob[key], dtype=np.float64).reshape(-1)
        if v.shape[0] == n:
            signals["maxprob"] = (v, f"{src.name}:{key} (mean token prob, higher=confident)")

    key = _find_key(blob, MEANLOGPROB_KEYS)
    if key is not None and "maxprob" not in signals:
        v = np.asarray(blob[key], dtype=np.float64).reshape(-1)
        if v.shape[0] == n:
            signals["maxprob"] = (v, f"{src.name}:{key} (mean token logprob, higher=confident)")

    key = _find_key(blob, ENTROPY_KEYS)
    if key is not None:
        v = np.asarray(blob[key], dtype=np.float64).reshape(-1)
        if v.shape[0] == n:
            # entropy: lower = more confident → negate
            signals["token_entropy"] = (-v, f"{src.name}:{key} (token entropy, negated so higher=confident)")

    # Derive from a per-token logprob matrix if scalars weren't precomputed.
    key = _find_key(blob, LOGPROB_MATRIX_KEYS)
    if key is not None:
        M = np.asarray(blob[key], dtype=np.float64)
        if M.ndim == 2 and M.shape[0] == n:
            mask_key = _find_key(blob, TOKEN_MASK_KEYS)
            if mask_key is not None:
                mask = np.asarray(blob[mask_key]).astype(bool)
                counts = np.maximum(mask.sum(axis=1), 1)
                mean_lp = (np.where(mask, M, 0.0)).sum(axis=1) / counts
            else:
                valid = np.isfinite(M)
                counts = np.maximum(valid.sum(axis=1), 1)
                mean_lp = np.where(valid, M, 0.0).sum(axis=1) / counts
            if "maxprob" not in signals:
                signals["maxprob"] = (mean_lp, f"{src.name}:{key} (derived mean token logprob)")

    if not signals:
        return None, src
    return signals, src


def semantic_entropy_approx(correct: np.ndarray):
    """Cheap semantic-entropy surrogate from K=8 answer agreement.

    True semantic entropy needs answer-cluster logits we don't cache. As an
    approximation we use the K=8 self-consistency agreement fraction: the
    fraction of the 8 samples whose final answer is correct is a proxy for
    answer-cluster concentration. Higher agreement → lower entropy → higher
    confidence. This is a weak surrogate, flagged as such in the output.
    """
    if not K8_DIR.exists():
        return None
    n = len(correct)
    agree = np.full(n, np.nan, dtype=np.float64)
    found = 0
    for i in range(n):
        f = K8_DIR / f"problem_{i:03d}.npz"
        if not f.exists():
            continue
        c = np.load(f)["correct"].astype(bool)
        # proxy confidence = how concentrated the K=8 outcomes are around the modal label
        p = float(c.mean())
        agree[i] = max(p, 1.0 - p)  # consensus strength, higher=more confident
        found += 1
    if found < n:
        return None
    return agree


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    correct = cache["correct"].astype(bool)
    assert correct.shape == (500,), correct.shape

    signals, src = load_token_signals(correct)
    if signals is None:
        # No cached token-probability log → cannot compute MaxProb baseline.
        searched = ", ".join(str(p) for p in TOKENPROB_CANDIDATES)
        print("MISSING_REGEN_INPUT token-probability log not found among:", searched,
              file=sys.stderr)
        return 2

    results = {
        "experiment": "P11-FE768",
        "description": "MaxProb / token-entropy selective baseline at 50% coverage vs F-8 71.6%",
        "coverage": COVERAGE,
        "n_problems": int(len(correct)),
        "overall_k1_accuracy": float(correct.mean()),
        "f8_selective_acc_at_half": F8_SELECTIVE_ACC_AT_HALF,
        "token_prob_source": str(src),
        "baselines": {},
    }

    for name, (conf, note) in signals.items():
        sel = selective_accuracy(conf, correct, COVERAGE)
        sel["note"] = note
        sel["delta_vs_f8"] = sel["answered_accuracy"] - F8_SELECTIVE_ACC_AT_HALF
        sel["matches_f8"] = bool(abs(sel["delta_vs_f8"]) <= 0.02)  # within 2 pts
        results["baselines"][name] = sel

    sem = semantic_entropy_approx(correct)
    if sem is not None:
        sel = selective_accuracy(sem, correct, COVERAGE)
        sel["note"] = "K=8 agreement-fraction surrogate for semantic entropy (weak proxy)"
        sel["delta_vs_f8"] = sel["answered_accuracy"] - F8_SELECTIVE_ACC_AT_HALF
        sel["matches_f8"] = bool(abs(sel["delta_vs_f8"]) <= 0.02)
        results["baselines"]["semantic_entropy_approx"] = sel

    # Interpretation flag: does ANY pure-confidence filter match F-8?
    matched = [k for k, v in results["baselines"].items() if v.get("matches_f8")]
    results["any_baseline_matches_f8"] = bool(matched)
    results["matching_baselines"] = matched
    results["verdict"] = (
        "F8_IS_CONFIDENCE_FILTER" if matched else "F8_BEATS_PURE_CONFIDENCE"
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print("wrote", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())