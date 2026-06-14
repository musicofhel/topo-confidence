"""P11-FE704 — OTV window-and-filter aggregation sweep on cached L19 DoM scores.

Replicates OTV (Table 4) window-and-filter aggregation on cached Qwen2.5-1.5B
per-token L19 DoM scores for MATH-500. For each problem we have a sequence of
per-token DoM projection scores; we sweep:

    method  ∈ {prefill, final-token,
               (mean|min|max) over last {100,400,1600,all} tokens}
    filter  ∈ {0%, 25%, 50%, 75%}  (drop the earliest fraction of the trace,
                                     keep the late portion, then window+pool)

and report AUROC against ground-truth K=1 correctness. The sweep is anchored to
F-2's 0.7731 prefill DoM AUROC and F-8's 71.6%@50% selective-prediction number
(selective answered-accuracy at coverage 0.5 reported for the best aggregator).

OTV's claim under test: mean/min over the last ~100 tokens with 50-75% filtering
dominates prefill-anchored aggregation. If it replicates here, F-2's
'prefill > final-token' headline is mechanistically incomplete.

Requires a per-token DoM cache (Stage 4a). Schema (either form accepted):
  - flat/offsets: dom_flat (sum_tok,) float, offsets (501,) int  [CSR-style]
  - object array: dom_tokens (500,) of 1D float arrays
  plus optional prefill_score/final_score/correct; correct falls back to the
  main prefill cache. If the per-token cache is absent the script prints
  MISSING_REGEN_INPUT and returns 2 (no per-token activations are committed).
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
PERTOKEN = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_pertoken_dom.npz"
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/otv_window_filter/results.json"

# Anchors from the source-of-truth headline table (1024-tok canonical).
F2_PREFILL_ANCHOR = 0.7731   # F-2 prefill L19 DoM AUROC, OOF 5-fold
F8_SELECTIVE_ANCHOR = 0.716  # F-8 answered accuracy @ coverage 0.50

WINDOWS = {"100": 100, "400": 400, "1600": 1600, "all": None}
POOLS = ("mean", "min", "max")
FILTERS = (0.0, 0.25, 0.5, 0.75)
COVERAGE = 0.5


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s)
    s = s[finite]
    lab = np.asarray(labels, dtype=bool)[finite]
    pos = s[lab]; neg = s[~lab]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def selective_accuracy(scores: np.ndarray, labels: np.ndarray, coverage: float) -> float:
    s = np.asarray(scores, dtype=np.float64)
    lab = np.asarray(labels, dtype=bool)
    finite = np.isfinite(s)
    s = s[finite]; lab = lab[finite]
    n = len(s)
    k = int(round(n * coverage))
    if k < 1:
        return float("nan")
    order = np.argsort(-s)  # high score = predicted-more-correct
    answered = order[:k]
    return float(lab[answered].mean())


def load_pertoken() -> tuple[list[np.ndarray], np.ndarray | None]:
    """Return list of per-problem per-token DoM arrays and optional final scores."""
    blob = np.load(PERTOKEN, allow_pickle=True)
    keys = set(blob.files)
    tokens: list[np.ndarray] = []
    if "dom_flat" in keys and "offsets" in keys:
        flat = blob["dom_flat"].astype(np.float64)
        off = blob["offsets"].astype(np.int64)
        for i in range(len(off) - 1):
            tokens.append(flat[off[i]:off[i + 1]])
    elif "dom_tokens" in keys:
        obj = blob["dom_tokens"]
        for arr in obj:
            tokens.append(np.asarray(arr, dtype=np.float64).ravel())
    else:
        raise KeyError("per-token cache lacks dom_flat/offsets or dom_tokens")
    final = blob["final_score"].astype(np.float64) if "final_score" in keys else None
    return tokens, final


def pooled_score(toks: np.ndarray, window: int | None, pool: str, filt: float) -> float:
    n = len(toks)
    if n == 0:
        return float("nan")
    keep = max(1, int(round(n * (1.0 - filt))))
    seg = toks[-keep:]
    if window is not None:
        seg = seg[-window:]
    if len(seg) == 0:
        return float("nan")
    if pool == "mean":
        return float(seg.mean())
    if pool == "min":
        return float(seg.min())
    if pool == "max":
        return float(seg.max())
    return float("nan")


def main() -> int:
    if not PERTOKEN.exists():
        print("MISSING_REGEN_INPUT", PERTOKEN, file=sys.stderr)
        return 2
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    correct = np.load(CACHE)["correct"].astype(bool)
    n_prob = len(correct)

    try:
        tokens, final_provided = load_pertoken()
    except KeyError as exc:
        print("MISSING_REGEN_INPUT", PERTOKEN, str(exc), file=sys.stderr)
        return 2

    if len(tokens) != n_prob:
        print("MISSING_REGEN_INPUT", PERTOKEN,
              f"n_problems mismatch {len(tokens)} != {n_prob}", file=sys.stderr)
        return 2

    # Prefill anchor: prefer the committed DoM score vector.
    if DOM_NPZ.exists():
        prefill_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    else:
        # Fall back to the first per-token score as a prefill proxy.
        prefill_score = np.array([t[0] if len(t) else np.nan for t in tokens])

    # Final-token score: provided, else the last per-token value.
    if final_provided is not None:
        final_score = final_provided
    else:
        final_score = np.array([t[-1] if len(t) else np.nan for t in tokens])

    grid: dict[str, dict] = {}

    # Single-point references (filtering N/A).
    grid["prefill"] = {
        "auroc": auroc(prefill_score, correct),
        "selective_acc@0.5": selective_accuracy(prefill_score, correct, COVERAGE),
    }
    grid["final_token"] = {
        "auroc": auroc(final_score, correct),
        "selective_acc@0.5": selective_accuracy(final_score, correct, COVERAGE),
    }

    # Window × pool × filter sweep.
    best = {"method": None, "auroc": -1.0, "selective_acc@0.5": float("nan")}
    for wname, wsize in WINDOWS.items():
        for pool in POOLS:
            for filt in FILTERS:
                scores = np.array(
                    [pooled_score(t, wsize, pool, filt) for t in tokens],
                    dtype=np.float64,
                )
                key = f"{pool}_last{wname}_filter{int(filt * 100)}"
                a = auroc(scores, correct)
                sel = selective_accuracy(scores, correct, COVERAGE)
                grid[key] = {
                    "auroc": a,
                    "selective_acc@0.5": sel,
                    "window": wname,
                    "pool": pool,
                    "filter": filt,
                }
                if np.isfinite(a) and a > best["auroc"]:
                    best = {"method": key, "auroc": a, "selective_acc@0.5": sel}

    beats_prefill = bool(np.isfinite(best["auroc"]) and best["auroc"] > F2_PREFILL_ANCHOR)

    out = {
        "experiment": "P11-FE704",
        "description": "OTV window-and-filter aggregation sweep on cached L19 DoM scores",
        "n_problems": int(n_prob),
        "anchors": {
            "F2_prefill_auroc": F2_PREFILL_ANCHOR,
            "F8_selective_acc@0.5": F8_SELECTIVE_ANCHOR,
        },
        "grid": grid,
        "best_windowed": best,
        "best_windowed_beats_F2_prefill": beats_prefill,
        "best_windowed_auroc_minus_F2": (
            float(best["auroc"] - F2_PREFILL_ANCHOR)
            if np.isfinite(best["auroc"]) else None
        ),
        "best_windowed_selective_minus_F8": (
            float(best["selective_acc@0.5"] - F8_SELECTIVE_ANCHOR)
            if np.isfinite(best["selective_acc@0.5"]) else None
        ),
        "verdict": (
            "OTV_REPLICATES_late_token_dominates"
            if beats_prefill else
            "OTV_DOES_NOT_REPLICATE_prefill_remains_strongest"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())