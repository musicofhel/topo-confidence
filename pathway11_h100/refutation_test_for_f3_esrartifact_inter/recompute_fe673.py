"""P11-FE673 — Refutation test for F-3's ESR-artifact interpretation.

F-3 reports cos(prefill_DoM, final_DoM) ≈ 0.046 — the L19 prefill and
final-token difference-of-means directions are near-orthogonal, which F-3 reads
as two distinct correctness channels. McKenzie's ESR offers an alternative: a
consistency-checking circuit re-orthogonalizes the final-token state *by
default*, so 0.046 may be a generic-circuit baseline carrying no
correctness-pathway information.

This script splits the orthogonality measurement by ground-truth correctness as
the cheapest discriminator:

  1. cos_global  — reproduce the aggregate cos(prefill_DoM, final_DoM) ≈ 0.046,
                   with a bootstrap CI.
  2. Per-problem prefill/final centered-state cosine, split correct vs
     incorrect (mean, std, n + Welch t-test). If both classes sit at the same
     near-orthogonal value, the split carries no class signal.
  3. Within-class correlation of prefill-DoM vs final-DoM projections — a
     non-degenerate per-class alignment readout.

If correct and incorrect classes are statistically indistinguishable on (2)/(3),
F-3's two-channel framing collapses into a generic consistency-circuit baseline.

Needs cached Pathway 10 L19 *final-token* activations alongside the prefill
cache; if the final-token cache is absent the script reports MISSING_REGEN_INPUT.
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
from scipy.stats import ttest_ind, pearsonr

ROOT = Path("/home/musicofhel/topo-confidence")
PREFILL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
# Final-token L19 activations live in a Pathway-10 / prefill_inversion cache.
# Exact filename/key drifted across rebuilds, so probe a small candidate set.
FINAL_CACHE_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final_token.npz",
    ROOT / "pathway10/cache/m15b_final.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz",
]
FINAL_KEY_CANDIDATES = ["final", "final_token", "last_token", "final_hidden", "hidden_final"]

OUT_JSON = ROOT / "pathway11_h100/esr_orthogonality_split/results.json"

SEED = 9999
N_BOOT = 2000


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def dom(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Difference-of-means direction: mean(correct) - mean(incorrect)."""
    return X[y].mean(axis=0) - X[~y].mean(axis=0)


def locate_final() -> tuple[np.ndarray, np.ndarray | None] | None:
    """Return (final_states (500,1536), correct_or_None) or None if not found."""
    for path in FINAL_CACHE_CANDIDATES:
        if not path.exists():
            continue
        try:
            blob = np.load(path)
        except Exception:
            continue
        keys = set(blob.files)
        for key in FINAL_KEY_CANDIDATES:
            if key in keys:
                arr = blob[key]
                if arr.ndim == 2 and arr.shape == (500, 1536):
                    corr = blob["correct"] if "correct" in keys else None
                    return arr.astype(np.float64), corr
    return None


def main() -> int:
    if not PREFILL_CACHE.exists():
        print("MISSING_REGEN_INPUT", PREFILL_CACHE, file=sys.stderr)
        return 2

    pblob = np.load(PREFILL_CACHE)
    prefill = pblob["prefill"].astype(np.float64)
    correct = pblob["correct"].astype(bool)
    assert prefill.shape == (500, 1536) and correct.shape == (500,)

    found = locate_final()
    if found is None:
        print("MISSING_REGEN_INPUT", "final-token L19 activation cache", file=sys.stderr)
        return 2
    final, final_correct = found
    if final_correct is not None:
        # Consistency check: label vectors must agree across caches.
        if not np.array_equal(final_correct.astype(bool), correct):
            print("MISSING_REGEN_INPUT", "final/prefill correctness labels disagree", file=sys.stderr)
            return 2

    rng = np.random.default_rng(SEED)
    n = len(correct)

    # --- (1) Global DoM-direction orthogonality (reproduce ~0.046) ---
    prefill_dom = dom(prefill, correct)
    final_dom = dom(final, correct)
    cos_global = cosine(prefill_dom, final_dom)

    boot = np.empty(N_BOOT, dtype=np.float64)
    n_pos = int(correct.sum())
    n_neg = n - n_pos
    valid = 0
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        yb = correct[idx]
        if yb.sum() == 0 or (~yb).sum() == 0:
            boot[b] = np.nan
            continue
        boot[b] = cosine(dom(prefill[idx], yb), dom(final[idx], yb))
        valid += 1
    boot_ok = boot[np.isfinite(boot)]
    boot_ci = (
        [float(np.percentile(boot_ok, 2.5)), float(np.percentile(boot_ok, 97.5))]
        if boot_ok.size
        else [float("nan"), float("nan")]
    )

    # --- (2) Per-problem centered-state prefill/final cosine, split by class ---
    prefill_c = prefill - prefill.mean(axis=0, keepdims=True)
    final_c = final - final.mean(axis=0, keepdims=True)
    pn = np.linalg.norm(prefill_c, axis=1)
    fn = np.linalg.norm(final_c, axis=1)
    denom = pn * fn
    safe = denom > 1e-12
    per_cos = np.full(n, np.nan, dtype=np.float64)
    per_cos[safe] = np.einsum("ij,ij->i", prefill_c[safe], final_c[safe]) / denom[safe]

    cc = per_cos[correct & np.isfinite(per_cos)]
    ic = per_cos[(~correct) & np.isfinite(per_cos)]
    if cc.size and ic.size:
        t_stat, p_val = ttest_ind(cc, ic, equal_var=False)
        t_stat, p_val = float(t_stat), float(p_val)
    else:
        t_stat, p_val = float("nan"), float("nan")

    # --- (3) Within-class correlation of DoM projections ---
    pdir = prefill_dom / (np.linalg.norm(prefill_dom) + 1e-12)
    fdir = final_dom / (np.linalg.norm(final_dom) + 1e-12)
    p_proj = prefill_c @ pdir
    f_proj = final_c @ fdir

    def within_class_corr(mask: np.ndarray) -> float:
        if mask.sum() < 3:
            return float("nan")
        r, _ = pearsonr(p_proj[mask], f_proj[mask])
        return float(r)

    corr_correct = within_class_corr(correct)
    corr_incorrect = within_class_corr(~correct)

    # --- Sanity AUROCs (in-sample DoM projections) ---
    auroc_prefill = auroc(p_proj, correct)
    auroc_final = auroc(f_proj, correct)

    # --- Verdict heuristic ---
    mean_cc = float(cc.mean()) if cc.size else float("nan")
    mean_ic = float(ic.mean()) if ic.size else float("nan")
    classes_indistinguishable = bool(np.isfinite(p_val) and p_val > 0.05)
    corr_both_near_zero = bool(
        np.isfinite(corr_correct)
        and np.isfinite(corr_incorrect)
        and abs(corr_correct) < 0.15
        and abs(corr_incorrect) < 0.15
    )
    f3_collapses = classes_indistinguishable and corr_both_near_zero

    out = {
        "experiment": "P11-FE673",
        "description": "ESR-artifact refutation: cos(prefill_DoM, final_DoM) split by correctness",
        "n_total": int(n),
        "n_correct": int(n_pos),
        "n_incorrect": int(n_neg),
        "cos_global_dom": cos_global,
        "cos_global_boot_ci95": boot_ci,
        "cos_global_boot_n_valid": int(valid),
        "per_problem_state_cosine": {
            "mean_correct": mean_cc,
            "std_correct": float(cc.std()) if cc.size else float("nan"),
            "n_correct": int(cc.size),
            "mean_incorrect": mean_ic,
            "std_incorrect": float(ic.std()) if ic.size else float("nan"),
            "n_incorrect": int(ic.size),
            "welch_t": t_stat,
            "welch_p": p_val,
        },
        "within_class_dom_projection_corr": {
            "correct": corr_correct,
            "incorrect": corr_incorrect,
        },
        "sanity_auroc_in_sample": {
            "prefill_dom_projection": auroc_prefill,
            "final_dom_projection": auroc_final,
        },
        "verdict": {
            "classes_indistinguishable_on_state_cosine": classes_indistinguishable,
            "within_class_corr_both_near_zero": corr_both_near_zero,
            "f3_two_channel_framing_collapses": f3_collapses,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())