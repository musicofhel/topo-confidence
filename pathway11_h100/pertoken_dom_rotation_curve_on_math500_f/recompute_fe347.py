"""P11-FE347 — Per-token DoM rotation curve on MATH-500.

For each generated-token position in the cached exp1_cross_model traces, build a
per-position direction-of-mean (DoM) from the L19 residual stream
(mean_correct - mean_incorrect across problems) and measure its angle against
the *fixed* prefill-DoM direction. Tracking that angle across token position
adjudicates between two readings of F-3's cos(prefill_DoM, final_DoM) = 0.046:

  - Kudo iterative-computation: rotation is CONTINUOUS — 0.046 is the natural
    endpoint of a long, monotone drift across positions.
  - F-3 two-signal: rotation is ABRUPT — prefill and final are qualitatively
    different signals, not a smooth interpolation.

Method (CPU, cached NPZs only):
  1. prefill-DoM := mean(prefill[correct]) - mean(prefill[~correct]) at L19,
     unit-normalized, from m15b_prefill.npz.
  2. For each problem trace, pull the L19 residual sequence (T_i, 1536).
  3. Bin tokens by fractional position (handles variable trace lengths), and
     for each bin compute DoM_bin across problems, then angle to prefill-DoM.
  4. Spearman(angle vs bin index) tests monotonicity; the largest single-step
     jump / total range flags abruptness. cos at the final bin is compared to
     the F-3 reference value 0.046.

If the per-token trace cache is absent (common on a fresh CPU box — these are
large and gitignored), the script prints MISSING_REGEN_INPUT and returns 2.
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
from scipy.stats import spearmanr

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
# Per-token, all-layer traces from the exp1_cross_model run. One NPZ per problem.
TRACE_DIRS = [
    ROOT / "pathway11_h100/exp1_cross_model/traces",
    ROOT / "pathway11_h100/exp1_cross_model/cache",
    ROOT / "pathway11_h100/data/exp1_cross_model",
]
OUT_JSON = ROOT / "pathway11_h100/dom_rotation_curve/results.json"

LAYER = 19          # L19 prefill direction is the canonical DoM layer
HIDDEN = 1536       # 1.5B residual width
N_BINS = 20         # fractional-position bins over the generated sequence
F3_REF_COS = 0.046  # cos(prefill_DoM, final_DoM) reference from scratch JSON
SEED = 9999

# Candidate keys for the per-token residual tensor inside each trace NPZ.
RESID_KEYS = ("resid", "residuals", "hidden", "hidden_states", "states", "h")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def find_trace_dir() -> Path | None:
    for d in TRACE_DIRS:
        if d.exists() and any(d.glob("*.npz")):
            return d
    return None


def extract_l19_sequence(blob) -> np.ndarray | None:
    """Return an (T, HIDDEN) L19 residual sequence from one trace NPZ, or None."""
    for key in RESID_KEYS:
        if key not in blob:
            continue
        arr = np.asarray(blob[key])
        if arr.ndim == 3:
            # (T, n_layers, D) or (n_layers, T, D) — pick by locating the layer axis.
            if arr.shape[1] > LAYER and arr.shape[-1] == HIDDEN:
                return arr[:, LAYER, :].astype(np.float64)
            if arr.shape[0] > LAYER and arr.shape[-1] == HIDDEN:
                return arr[LAYER, :, :].astype(np.float64)
        elif arr.ndim == 2 and arr.shape[-1] == HIDDEN:
            # Already a single-layer (T, D) sequence.
            return arr.astype(np.float64)
    # Per-layer keyed storage, e.g. "layer_19".
    for cand in (f"layer_{LAYER}", f"l{LAYER}", f"L{LAYER}"):
        if cand in blob:
            arr = np.asarray(blob[cand])
            if arr.ndim == 2 and arr.shape[-1] == HIDDEN:
                return arr.astype(np.float64)
    return None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    trace_dir = find_trace_dir()
    if trace_dir is None:
        print("MISSING_REGEN_INPUT", " | ".join(str(d) for d in TRACE_DIRS), file=sys.stderr)
        return 2

    base = np.load(CACHE)
    prefill = base["prefill"].astype(np.float64)
    correct = base["correct"].astype(bool)
    assert prefill.shape == (500, HIDDEN) and correct.shape == (500,)

    # Fixed prefill-DoM direction (unit), and a sanity AUROC for provenance.
    dom_prefill = prefill[correct].mean(0) - prefill[~correct].mean(0)
    dom_prefill_u = unit(dom_prefill)
    auroc_prefill = auroc(prefill @ dom_prefill_u, correct)

    trace_files = sorted(trace_dir.glob("*.npz"))
    # Accumulators: per fractional bin, sum of L19 residuals split by correctness.
    sum_pos = np.zeros((N_BINS, HIDDEN), dtype=np.float64)
    sum_neg = np.zeros((N_BINS, HIDDEN), dtype=np.float64)
    cnt_pos = np.zeros(N_BINS, dtype=np.int64)
    cnt_neg = np.zeros(N_BINS, dtype=np.int64)
    # Final-token DoM accumulator (last position of each trace).
    final_pos = np.zeros(HIDDEN, dtype=np.float64)
    final_neg = np.zeros(HIDDEN, dtype=np.float64)
    n_final_pos = n_final_neg = 0
    n_used = 0

    for fp in trace_files:
        # Map filename index -> problem index for the correctness label.
        digits = "".join(ch for ch in fp.stem if ch.isdigit())
        if not digits:
            continue
        pid = int(digits)
        if pid >= len(correct):
            continue
        try:
            blob = np.load(fp)
        except Exception:
            continue
        seq = extract_l19_sequence(blob)
        if seq is None or seq.shape[0] == 0:
            continue
        T = seq.shape[0]
        is_correct = bool(correct[pid])
        # Assign each token to a fractional bin in [0, N_BINS).
        frac = (np.arange(T) / max(T - 1, 1)) * (N_BINS - 1)
        bins = np.clip(np.round(frac).astype(int), 0, N_BINS - 1)
        for b in range(N_BINS):
            sel = seq[bins == b]
            if sel.shape[0] == 0:
                continue
            s = sel.sum(0)
            if is_correct:
                sum_pos[b] += s
                cnt_pos[b] += sel.shape[0]
            else:
                sum_neg[b] += s
                cnt_neg[b] += sel.shape[0]
        last = seq[-1]
        if is_correct:
            final_pos += last
            n_final_pos += 1
        else:
            final_neg += last
            n_final_neg += 1
        n_used += 1

    if n_used == 0:
        print("MISSING_REGEN_INPUT no-usable-traces", trace_dir, file=sys.stderr)
        return 2

    # Per-bin DoM and angle (degrees) to the fixed prefill direction.
    bin_cos = np.full(N_BINS, np.nan, dtype=np.float64)
    bin_angle = np.full(N_BINS, np.nan, dtype=np.float64)
    for b in range(N_BINS):
        if cnt_pos[b] == 0 or cnt_neg[b] == 0:
            continue
        dom_b = sum_pos[b] / cnt_pos[b] - sum_neg[b] / cnt_neg[b]
        c = float(dom_prefill_u @ unit(dom_b))
        c = max(-1.0, min(1.0, c))
        bin_cos[b] = c
        bin_angle[b] = float(np.degrees(np.arccos(c)))

    valid = ~np.isnan(bin_angle)
    idx = np.flatnonzero(valid)
    angles_valid = bin_angle[valid]

    # Monotonicity: Spearman(angle vs bin position). |rho|->1 => smooth drift.
    if len(idx) >= 3:
        rho, pval = spearmanr(idx.astype(float), angles_valid)
    else:
        rho, pval = float("nan"), float("nan")

    # Abruptness: largest single-step angle jump / total range. A value near 1
    # means the rotation happens in one step (F-3 two-signal); near 1/n_steps
    # means evenly spread (Kudo continuous).
    if len(angles_valid) >= 2:
        steps = np.abs(np.diff(angles_valid))
        ang_range = float(angles_valid.max() - angles_valid.min())
        max_step = float(steps.max())
        abruptness = float(max_step / ang_range) if ang_range > 1e-9 else float("nan")
        uniform_step = 1.0 / (len(angles_valid) - 1)
    else:
        max_step = float("nan")
        abruptness = float("nan")
        uniform_step = float("nan")

    # Final-token DoM cos to prefill (cross-check against F-3's 0.046).
    if n_final_pos > 0 and n_final_neg > 0:
        dom_final = final_pos / n_final_pos - final_neg / n_final_neg
        cos_final = float(dom_prefill_u @ unit(dom_final))
        cos_final = max(-1.0, min(1.0, cos_final))
    else:
        cos_final = float("nan")

    # Verdict heuristic: continuous if drift is strongly monotone AND no single
    # step dominates; abrupt if one step carries most of the rotation.
    if np.isnan(rho) or np.isnan(abruptness):
        verdict = "INCONCLUSIVE"
    elif abruptness >= 0.6:
        verdict = "ABRUPT_supports_F3_two_signal"
    elif abs(rho) >= 0.7 and abruptness < 0.4:
        verdict = "MONOTONIC_supports_Kudo_iterative"
    else:
        verdict = "MIXED"

    out = {
        "experiment": "P11-FE347",
        "n_traces_used": int(n_used),
        "trace_dir": str(trace_dir),
        "n_bins": N_BINS,
        "layer": LAYER,
        "auroc_prefill_dom_sanity": float(auroc_prefill),
        "bin_cos_to_prefill": [None if np.isnan(v) else float(v) for v in bin_cos],
        "bin_angle_deg": [None if np.isnan(v) else float(v) for v in bin_angle],
        "n_valid_bins": int(len(idx)),
        "spearman_angle_vs_position": None if np.isnan(rho) else float(rho),
        "spearman_pvalue": None if np.isnan(pval) else float(pval),
        "max_single_step_deg": None if np.isnan(max_step) else float(max_step),
        "abruptness_ratio": None if np.isnan(abruptness) else float(abruptness),
        "uniform_step_fraction": None if np.isnan(uniform_step) else float(uniform_step),
        "cos_final_dom_to_prefill": None if np.isnan(cos_final) else float(cos_final),
        "f3_reference_cos": F3_REF_COS,
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print("OK", verdict, "traces_used=", n_used, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())