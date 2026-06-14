"""P11-FE1021 — DoM confidence vs. output answer diversity (Spearman rho).

Tests whether the L19 prefill DoM confidence score is a redundant single-pass
proxy for output-based uncertainty. For each of the 500 MATH-500 problems we
compute answer diversity across the K=8 self-consistency samples (number of
distinct final answers) and rank-correlate it against the per-problem DoM
score. A strong negative rho (|rho| > 0.8) means DoM merely restates output
consistency; a weak one (|rho| < 0.4) means DoM captures a genuinely different,
pre-generation signal — the two-signal reading of F-3.

If the K=8 caches expose only per-sample correctness (no answer strings), we
fall back to a correctness-derived diversity proxy and flag it in the output.
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
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
OUT_JSON = ROOT / "pathway11_h100/dom_answer_diversity/results.json"

N_PROBLEMS = 500
# Names that, if present in a problem NPZ, hold the per-sample final answers.
ANSWER_KEYS = (
    "answers", "answer", "final_answers", "final_answer", "pred", "preds",
    "predictions", "extracted", "extracted_answers", "responses",
)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def find_answer_key(blob) -> str | None:
    """Return the NPZ key holding per-sample answer strings, or None."""
    keys = set(blob.files)
    for name in ANSWER_KEYS:
        if name in keys:
            return name
    # Heuristic fallback: any non-bool length-K array that isn't 'correct'.
    for name in blob.files:
        if name == "correct":
            continue
        arr = blob[name]
        if arr.ndim == 1 and arr.shape[0] >= 2 and arr.dtype != np.bool_:
            return name
    return None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2
    if not K8_DIR.exists():
        print("MISSING_REGEN_INPUT", K8_DIR, file=sys.stderr)
        return 2

    correct_gt = np.load(CACHE)["correct"].astype(bool)
    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert dom_score.shape == (N_PROBLEMS,) and correct_gt.shape == (N_PROBLEMS,)

    diversity = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    frac_correct = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    used_answer_strings = False
    answer_key_seen = None
    missing = []

    for i in range(N_PROBLEMS):
        fp = K8_DIR / f"problem_{i:03d}.npz"
        if not fp.exists():
            missing.append(i)
            continue
        blob = np.load(fp, allow_pickle=True)
        corr = blob["correct"].astype(bool)
        frac_correct[i] = float(corr.mean())

        akey = find_answer_key(blob)
        if akey is not None:
            answer_key_seen = akey
            used_answer_strings = True
            ans = [str(a) for a in np.asarray(blob[akey]).ravel().tolist()]
            diversity[i] = float(len(set(ans)))
        else:
            # Correctness-only proxy: distinct outcome classes among samples.
            # All-correct collapses to the single right answer (diversity 1);
            # any incorrect contributes an extra lumped "wrong" class.
            n_correct = int(corr.sum())
            n_wrong = int((~corr).sum())
            diversity[i] = float((1 if n_correct > 0 else 0) +
                                 (1 if n_wrong > 0 else 0))

    if missing:
        print("MISSING_REGEN_INPUT", f"{len(missing)} problem NPZ files",
              missing[:10], file=sys.stderr)
        return 2

    valid = ~np.isnan(diversity)
    rho, pval = spearmanr(dom_score[valid], diversity[valid])
    rho_fc, pval_fc = spearmanr(dom_score[valid], frac_correct[valid])

    abs_rho = abs(float(rho))
    if abs_rho > 0.8:
        interpretation = "REDUNDANT: DoM is a single-pass proxy for output consistency"
    elif abs_rho < 0.4:
        interpretation = "DISTINCT: DoM captures a genuinely different signal (two-signal F-3)"
    else:
        interpretation = "PARTIAL: DoM shares moderate signal overlap with output diversity"

    out = {
        "experiment": "P11-FE1021",
        "n_problems": int(valid.sum()),
        "used_answer_strings": used_answer_strings,
        "answer_key": answer_key_seen,
        "diversity_is_proxy": (not used_answer_strings),
        "spearman_rho_dom_vs_diversity": float(rho),
        "spearman_p_dom_vs_diversity": float(pval),
        "abs_rho": abs_rho,
        "spearman_rho_dom_vs_frac_correct": float(rho_fc),
        "spearman_p_dom_vs_frac_correct": float(pval_fc),
        "mean_distinct_answers": float(np.nanmean(diversity)),
        "max_distinct_answers": float(np.nanmax(diversity)),
        "auroc_dom_vs_gt_correct": float(auroc(dom_score, correct_gt)),
        "interpretation": interpretation,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())