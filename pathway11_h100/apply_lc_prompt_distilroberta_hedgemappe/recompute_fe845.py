"""P11-FE845 — LC+ prompt + DistilRoBERTa hedge-mapper black-box baseline vs F-2.

Direct refutation test for F-2's mechanistic ("deep geometric finding") framing.
The protocol from 2509.24202 is: (a) re-generate the MATH-500 K=1 answers with the
"LC+" linguistic-confidence prompt that asks the model to verbalize its confidence,
then (b) score each generation's hedging language with an ~80M-param DistilRoBERTa
hedge-mapper that emits a scalar confidence in [0,1]. Steps (a) and (b) require a
GPU + the transformers stack and are run UPSTREAM; they deposit their per-problem
confidence scores into an NPZ aligned to the same 500 MATH-500 problems as the
main prefill cache.

This recompute script is the CPU-only tail: it loads that cached confidence vector,
computes its AUROC against the existing correctness labels, lines it up directly
against the F-2 prefill-DoM AUROC (0.7731), and renders a verdict. If a black-box
prompt + small hedge mapper reaches >= 0.77 here, F-2 is recovering a signal the
model already exposes verbally rather than a deep residual-stream geometric finding.

If the upstream hedge-score NPZ has not been generated, the script reports
MISSING_REGEN_INPUT and exits 2 (it cannot itself run the LC+ generation or the
transformer mapper under the CPU-only / no-network constraints).
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
# Upstream LC+ generation + DistilRoBERTa hedge-mapper deposit confidence here,
# aligned to the same 500 MATH-500 problems (same order) as CACHE.
HEDGE_NPZ = ROOT / "pathway11_h100/lcplus_hedge_mapper/cache/lcplus_hedge_scores.npz"
OUT_JSON = ROOT / "pathway11_h100/lcplus_hedge_mapper/results.json"

F2_DOM_AUROC = 0.7731   # FINDINGS F-2, prefill L19 DoM, 1.5B, OOF 5-fold (1024tok)
REFUTE_THRESHOLD = 0.77
N_BOOT = 2000
SEED = 9999


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def bootstrap_ci(scores: np.ndarray, labels: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = np.empty(n_boot, dtype=np.float64)
    k = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = labels[idx]
        if yb.all() or (~yb).all():
            continue
        vals[k] = auroc(scores[idx], yb)
        k += 1
    if k == 0:
        return (float("nan"), float("nan"))
    vals = vals[:k]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _load_confidence(blob) -> np.ndarray:
    """Accept whichever key the upstream mapper wrote; return higher=more confident."""
    keys = set(blob.files)
    if "confidence" in keys:
        return blob["confidence"].astype(np.float64)
    if "lc_confidence" in keys:
        return blob["lc_confidence"].astype(np.float64)
    if "hedge_score" in keys:
        # hedge_score: higher = more hedging = less confident -> negate for confidence
        return -blob["hedge_score"].astype(np.float64)
    raise KeyError(f"no recognized confidence key in {sorted(keys)}")


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not HEDGE_NPZ.exists():
        # Upstream LC+ generation + transformer hedge-mapper not yet run; this
        # CPU-only / no-GPU / no-network script cannot produce it.
        print("MISSING_REGEN_INPUT", HEDGE_NPZ, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    correct = cache["correct"].astype(bool)
    assert correct.shape == (500,), correct.shape

    hedge_blob = np.load(HEDGE_NPZ)
    try:
        confidence = _load_confidence(hedge_blob)
    except KeyError as exc:
        print("MISSING_REGEN_INPUT", HEDGE_NPZ, str(exc), file=sys.stderr)
        return 2

    if confidence.shape != (500,):
        print("MISSING_REGEN_INPUT", HEDGE_NPZ, f"bad shape {confidence.shape}",
              file=sys.stderr)
        return 2

    finite = np.isfinite(confidence)
    if not finite.all():
        # Drop unparseable generations rather than poison the AUROC.
        confidence = confidence[finite]
        eval_correct = correct[finite]
    else:
        eval_correct = correct

    lc_auroc = auroc(confidence, eval_correct)
    lo, hi = bootstrap_ci(confidence, eval_correct, N_BOOT, SEED)

    # F-2 DoM baseline on the identical labels, when the DoM scores are present.
    dom_auroc = float("nan")
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        if dom_score.shape == (500,):
            dom_auroc = auroc(dom_score, correct)

    delta = lc_auroc - F2_DOM_AUROC
    refutes_f2 = bool(np.isfinite(lc_auroc) and lc_auroc >= REFUTE_THRESHOLD)

    if not np.isfinite(lc_auroc):
        verdict = "INCONCLUSIVE"
    elif refutes_f2:
        verdict = ("REFUTES_F2_MECHANISTIC_FRAMING: black-box LC+ prompt + 80M "
                   "hedge-mapper matches prefill-DoM; F-2 recovers an "
                   "already-verbalized signal")
    elif lc_auroc >= 0.60:
        verdict = ("PARTIAL: hedge-mapper carries signal but stays below F-2 "
                   "prefill-DoM; mechanistic framing survives the black-box check")
    else:
        verdict = ("F2_SURVIVES: black-box linguistic confidence is weak; "
                   "prefill-DoM is not merely a re-read of verbal hedging")

    out = {
        "experiment": "P11-FE845",
        "source_paper": "2509.24202",
        "n_eval": int(eval_correct.shape[0]),
        "n_dropped_nonfinite": int((~finite).sum()),
        "lcplus_hedge_auroc": lc_auroc,
        "lcplus_hedge_auroc_ci95": [lo, hi],
        "f2_prefill_dom_auroc_reference": F2_DOM_AUROC,
        "f2_prefill_dom_auroc_recomputed": dom_auroc,
        "delta_lcplus_minus_f2": float(delta),
        "refute_threshold": REFUTE_THRESHOLD,
        "refutes_f2": refutes_f2,
        "verdict": verdict,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())