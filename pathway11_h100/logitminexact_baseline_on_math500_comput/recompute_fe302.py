"""P11-FE302 — Logit-min-exact baseline on MATH-500.

Orgad et al. (Section 4) show that much of what looks like hidden-state "probe
generalization" is already captured by a logit-min-exact baseline: the minimum
log-probability the model assigns to any token of its own (boxed) answer span.
This script computes that scalar per problem and reports its AUROC against
ground-truth correctness. It is the null-baseline subtractor that any future
F-2 probe AUROC claim must be reported on top of — without it, F-2's 0.7731
cannot be cleanly attributed to hidden-state geometry vs output logits.

Predictor convention: a higher (less negative) min answer-token log-prob means
the model was more confident across the entire boxed answer, so the raw
min-logprob is used directly as the correctness score (no sign flip).

Input is the cached generation log-probs over each problem's boxed answer span.
We probe the documented cache locations for those log-probs; if none carry
answer-token log-probs the script reports MISSING_REGEN_INPUT and exits 2
(re-extract the boxed-answer logprobs from the generation pass first).
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
OUT_JSON = ROOT / "pathway11_h100/logit_min_exact/results.json"

N = 500

# Candidate npz files that may carry per-problem boxed-answer token log-probs,
# in priority order. Each candidate lists the field names we accept.
LOGPROB_CANDIDATES = [
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_answer_logprobs.npz",
     ["answer_token_logprobs", "answer_logprobs", "boxed_token_logprobs", "token_logprobs"]),
    (ROOT / "pathway11_h100/data/generation_logprobs.npz",
     ["answer_token_logprobs", "answer_logprobs", "boxed_token_logprobs", "token_logprobs"]),
    (CACHE,
     ["answer_token_logprobs", "answer_logprobs", "boxed_token_logprobs"]),
]
# A field that already holds the reduced per-problem minimum, if present.
MIN_FIELDS = ["answer_min_logprob", "min_answer_logprob", "logit_min_exact"]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def _row_min(entry) -> float:
    """Minimum log-prob over one problem's boxed-answer token span."""
    arr = np.asarray(entry, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.min())


def _reduce_min(field) -> np.ndarray:
    """Reduce a stored logprob field to a (N,) min-over-answer-tokens vector."""
    if isinstance(field, np.ndarray) and field.dtype == object:
        return np.array([_row_min(field[i]) for i in range(len(field))], dtype=np.float64)
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim == 1:
        # Already a per-problem scalar (e.g. a precomputed min).
        return arr.astype(np.float64)
    if arr.ndim == 2:
        # (N, T) padded with NaN/inf for absent tokens.
        out = np.full(arr.shape[0], np.nan, dtype=np.float64)
        for i in range(arr.shape[0]):
            row = arr[i][np.isfinite(arr[i])]
            if row.size:
                out[i] = float(row.min())
        return out
    raise ValueError(f"unsupported logprob field shape {arr.shape}")


def _load_min_logprobs() -> tuple[np.ndarray | None, str]:
    """Find a cache carrying answer-token log-probs; return (min_vec, source)."""
    for path, fields in LOGPROB_CANDIDATES:
        if not path.exists():
            continue
        blob = np.load(path, allow_pickle=True)
        keys = set(blob.files)
        for mf in MIN_FIELDS:
            if mf in keys:
                vec = np.asarray(blob[mf], dtype=np.float64)
                if vec.shape[0] == N:
                    return vec, f"{path.name}:{mf}"
        for f in fields:
            if f in keys:
                vec = _reduce_min(blob[f])
                if vec.shape[0] == N:
                    return vec, f"{path.name}:{f}"

    # Fall back to per-problem K=8 files, if they carry answer-token logprobs.
    if K8_DIR.exists():
        sample = K8_DIR / "problem_000.npz"
        if sample.exists():
            sblob = np.load(sample, allow_pickle=True)
            lp_key = next((k for k in ("answer_token_logprobs", "answer_logprobs",
                                       "boxed_token_logprobs", "token_logprobs")
                           if k in set(sblob.files)), None)
            if lp_key is not None:
                vec = np.full(N, np.nan, dtype=np.float64)
                for i in range(N):
                    fp = K8_DIR / f"problem_{i:03d}.npz"
                    if not fp.exists():
                        continue
                    pb = np.load(fp, allow_pickle=True)
                    # Take the first (greedy / K=1) generation's answer span.
                    entry = pb[lp_key]
                    if isinstance(entry, np.ndarray) and entry.dtype == object:
                        entry = entry[0]
                    elif np.asarray(entry).ndim == 2:
                        entry = np.asarray(entry)[0]
                    vec[i] = _row_min(entry)
                return vec, f"k8_selfconsistency:{lp_key}"

    return None, ""


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    correct = np.load(CACHE)["correct"].astype(bool)
    assert correct.shape == (N,)

    min_logprob, source = _load_min_logprobs()
    if min_logprob is None:
        print("MISSING_REGEN_INPUT", "boxed-answer token logprobs not cached",
              file=sys.stderr)
        return 2

    valid = np.isfinite(min_logprob)
    if valid.sum() < 2 or not (correct[valid].any() and (~correct[valid]).any()):
        print("MISSING_REGEN_INPUT", "insufficient valid answer-token logprobs",
              file=sys.stderr)
        return 2

    score = min_logprob[valid]
    lab = correct[valid]
    auroc_logit_min_exact = auroc(score, lab)

    out = {
        "experiment": "P11-FE302",
        "name": "logit_min_exact_baseline",
        "source": source,
        "n_total": int(N),
        "n_valid": int(valid.sum()),
        "n_missing": int((~valid).sum()),
        "accuracy_on_valid": float(lab.mean()),
        "auroc_logit_min_exact": float(auroc_logit_min_exact),
        "min_logprob_mean": float(np.mean(score)),
        "min_logprob_std": float(np.std(score)),
        "note": ("Null-baseline subtractor for F-2. Report any hidden-state "
                 "probe AUROC relative to this logit-min-exact value (Orgad "
                 "et al. Section 4). Higher min answer-token logprob => more "
                 "confident => predicted correct (no sign flip)."),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())