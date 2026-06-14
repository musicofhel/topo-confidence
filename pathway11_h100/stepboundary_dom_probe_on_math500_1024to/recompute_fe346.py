"""P11-FE346 — Step-boundary DoM probe vs prefill DoM on MATH-500 1024-tok caches.

Tests Kudo's prediction that faithful-CoT models concentrate the correctness
signal at reasoning step boundaries (newline + 'step|=|answer') rather than at
the prefill. For each model (1.5B, 7B) we:

  (a) fit a difference-of-means (DoM) direction on L19 step-boundary activations
      aggregated per problem and score OOF AUROC against K=1 correctness;
  (b) fit a DoM on the step level and score OOF AUROC against per-step
      sub-answer correctness from parsed traces;

and compare both to the F-2 prefill-DoM AUROC and to F-8 selective prediction at
coverage 0.5. The refutation #3 it directly tests: if 7B step-boundary AUROC >
7B prefill AUROC while 1.5B prefill stays strongest, F-8 needs a model-scale
reframing.

The step-boundary activation caches (m{15b,7b}_stepboundary.npz) are Stage-2
regen products — per-token L19 activations sliced to regex-matched step-boundary
positions, plus per-step parsed sub-answer correctness. If they (or the 7B
prefill cache) are absent on this machine the script reports MISSING_REGEN_INPUT.
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
CACHE_15B = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
CACHE_7B = ROOT / "pathway11_h100/prefill_inversion/cache/m7b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
STEPB_15B = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_stepboundary.npz"
STEPB_7B = ROOT / "pathway11_h100/prefill_inversion/cache/m7b_stepboundary.npz"
OUT_JSON = ROOT / "pathway11_h100/step_boundary_dom/results.json"

N_FOLDS = 5
SEED = 9999
COVERAGE = 0.5

# Per-model spec: name -> (prefill cache, step-boundary cache).
MODELS = {
    "m1.5b": (CACHE_15B, STEPB_15B),
    "m7b": (CACHE_7B, STEPB_7B),
}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def dom_oof_scores(X: np.ndarray, y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Out-of-fold difference-of-means projection scores."""
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    folds = stratified_kfold(y, k, seed)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        ytr = y[train_mask]
        if ytr.all() or (~ytr).any() is False:
            # degenerate fold — leave scores at zero
            continue
        Xtr = X[train_mask]
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        scores[test_idx] = X[test_idx] @ d_vec
    return scores


def selective_accuracy(scores: np.ndarray, correct: np.ndarray, coverage: float) -> dict:
    """Answer the top-`coverage` fraction by confidence; report answered accuracy."""
    n = len(scores)
    n_answer = max(1, int(round(coverage * n)))
    order = np.argsort(-scores)  # most confident first
    answered = order[:n_answer]
    return {
        "coverage": float(n_answer / n),
        "answered_accuracy": float(correct[answered].mean()),
        "overall_accuracy": float(correct.mean()),
        "n_answered": int(n_answer),
    }


def aggregate_per_problem(acts: np.ndarray, prob_idx: np.ndarray, n_problems: int):
    """Mean step-boundary activation per problem; mask of problems with >=1 step."""
    d = acts.shape[1]
    agg = np.zeros((n_problems, d), dtype=np.float64)
    has = np.zeros(n_problems, dtype=bool)
    for p in range(n_problems):
        sel = prob_idx == p
        if sel.any():
            agg[p] = acts[sel].mean(axis=0)
            has[p] = True
    return agg, has


def process_model(name: str, prefill_cache: Path, stepb_cache: Path) -> dict:
    pf = np.load(prefill_cache)
    prefill = pf["prefill"].astype(np.float64)
    correct = pf["correct"].astype(bool)
    n_problems = len(correct)

    sb = np.load(stepb_cache)
    stepb_acts = sb["stepb_acts"].astype(np.float64)          # (M, 1536)
    stepb_problem = sb["stepb_problem"].astype(np.int64)      # (M,) in [0, n_problems)
    stepb_correct = sb["stepb_correct"].astype(bool)          # (M,) per-step sub-answer

    res: dict = {"n_problems": int(n_problems), "n_step_boundaries": int(len(stepb_acts))}

    # --- F-2 baseline: prefill DoM AUROC + selective prediction ---
    prefill_scores = dom_oof_scores(prefill, correct, N_FOLDS, SEED)
    res["auroc_prefill_oof"] = auroc(prefill_scores, correct)
    res["selpred_prefill"] = selective_accuracy(prefill_scores, correct, COVERAGE)

    # --- (a) step-boundary DoM aggregated per problem vs K=1 correctness ---
    agg, has = aggregate_per_problem(stepb_acts, stepb_problem, n_problems)
    if has.sum() >= 2 * N_FOLDS and correct[has].any() and (~correct[has]).any():
        agg_scores_sub = dom_oof_scores(agg[has], correct[has], N_FOLDS, SEED)
        res["auroc_stepb_problem_oof"] = auroc(agg_scores_sub, correct[has])
        res["selpred_stepb_problem"] = selective_accuracy(
            agg_scores_sub, correct[has], COVERAGE
        )
        res["n_problems_with_steps"] = int(has.sum())
    else:
        res["auroc_stepb_problem_oof"] = float("nan")
        res["selpred_stepb_problem"] = None
        res["n_problems_with_steps"] = int(has.sum())

    # --- (b) step-level DoM vs per-step sub-answer correctness ---
    if stepb_correct.any() and (~stepb_correct).any():
        perstep_scores = dom_oof_scores(stepb_acts, stepb_correct, N_FOLDS, SEED)
        res["auroc_stepb_perstep_oof"] = auroc(perstep_scores, stepb_correct)
    else:
        res["auroc_stepb_perstep_oof"] = float("nan")

    return res


def main() -> int:
    required = [DOM_NPZ]
    for prefill_cache, stepb_cache in MODELS.values():
        required += [prefill_cache, stepb_cache]
    missing = [p for p in required if not p.exists()]
    if missing:
        for p in missing:
            print("MISSING_REGEN_INPUT", p, file=sys.stderr)
        return 2

    out: dict = {"experiment": "P11-FE346", "models": {}}
    for name, (prefill_cache, stepb_cache) in MODELS.items():
        out["models"][name] = process_model(name, prefill_cache, stepb_cache)

    # --- refutation #3 verdict ---
    m15 = out["models"].get("m1.5b", {})
    m7 = out["models"].get("m7b", {})
    a15_pre = m15.get("auroc_prefill_oof", float("nan"))
    a15_sb = m15.get("auroc_stepb_problem_oof", float("nan"))
    a7_pre = m7.get("auroc_prefill_oof", float("nan"))
    a7_sb = m7.get("auroc_stepb_problem_oof", float("nan"))

    scale_reframe = bool(
        np.isfinite(a7_sb) and np.isfinite(a7_pre) and np.isfinite(a15_pre)
        and np.isfinite(a15_sb)
        and a7_sb > a7_pre and a15_pre >= a15_sb
    )
    out["verdict"] = {
        "refutation_3_scale_reframe": scale_reframe,
        "delta_7b_stepb_minus_prefill": (
            float(a7_sb - a7_pre) if np.isfinite(a7_sb) and np.isfinite(a7_pre) else None
        ),
        "delta_1.5b_prefill_minus_stepb": (
            float(a15_pre - a15_sb) if np.isfinite(a15_pre) and np.isfinite(a15_sb) else None
        ),
        "note": (
            "scale_reframe True => 7B concentrates correctness signal at step "
            "boundaries while 1.5B prefill stays strongest; F-8 would need a "
            "model-scale reframing per Kudo."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())