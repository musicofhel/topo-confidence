"""P11-FE356 — rStar-Math back-propagated Q-values vs prefill L19 DoM.

Tests H-5 (prefill encodes familiarity vs decomposability) using rStar-Math's
own difficulty proxy. For each MATH-500 problem we have K=8 cached
self-consistency trajectories. rStar-Math Eq. 2 back-propagates the terminal
reward q(s_d) ∈ {0,1} (generation correctness) to every node on the rollout via
the incremental-mean update

    Q(s_i) ← Q(s_i) + (q(s_d) − Q(s_i)) / N(s_i).

Our K=8 rollouts are independent *linear* trajectories (no shared MCTS tree), so
each step is visited once and its converged Q equals the terminal reward of the
generation it belongs to. We segment each generation into steps on the "\n\n"
delimiter (when generation text is cached) so that mean-Q per problem is the
*step-weighted* mean terminal reward — this is what can differ from the raw
correct-rate (the "Q-shape"). When no text is cached we fall back to the
unweighted terminal-reward mean and flag step_shape_available=False.

Decision logic for the rationale:
  - dom_pct ~ mean_q strongly      → prefill DoM tracks decomposability/difficulty
  - dom_pct ~ correct_rate but the partial corr of dom_pct~mean_q | correct_rate
    is ~0                          → prefill DoM is familiarity (raw success), not
                                     Q-shape.
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
from scipy.stats import spearmanr, pearsonr, rankdata

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
OUT_JSON = ROOT / "pathway11_h100/rstar_q_values/results.json"

N_PROBLEMS = 500
STEP_DELIM = "\n\n"
# Candidate NPZ keys that may hold per-generation completion text.
TEXT_KEYS = ("text", "texts", "responses", "generations", "completions",
             "samples", "solutions", "outputs")


def auroc(scores, labels):
    pos = scores[labels]; neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def partial_corr(a, b, c):
    """Pearson corr of a and b after linearly residualizing both on c."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    A = np.column_stack([c, np.ones_like(c)])
    ra = a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    rb = b - A @ np.linalg.lstsq(A, b, rcond=None)[0]
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(pearsonr(ra, rb)[0])


def _extract_texts(blob):
    """Return a list of 8 generation strings if any text key is present, else None."""
    for k in TEXT_KEYS:
        if k in blob.files:
            arr = blob[k]
            try:
                vals = [str(x) for x in np.atleast_1d(arr).ravel().tolist()]
            except Exception:
                continue
            if len(vals) >= 1:
                return vals
    return None


def backprop_mean_q(correct8, texts):
    """rStar-Math Eq. 2 back-prop on linear trajectories.

    Each generation g has terminal reward q_g = 1.0 if correct else 0.0. Every
    step on g's path converges to Q = q_g (single visit). mean-Q is the mean
    over all steps; step counts come from "\n\n" segmentation when text exists,
    else each generation contributes a single unit (unweighted terminal mean).
    Returns (mean_q, step_weighted: bool, total_steps).
    """
    q_g = correct8.astype(np.float64)
    if texts is not None and len(texts) == len(q_g):
        step_qs = []
        total = 0
        for g, t in enumerate(q_g):
            n_steps = max(1, len([s for s in texts[g].split(STEP_DELIM) if s.strip()]))
            step_qs.extend([t] * n_steps)
            total += n_steps
        return float(np.mean(step_qs)), True, int(total)
    return float(np.mean(q_g)), False, int(len(q_g))


def main() -> int:
    for p in (CACHE, DOM_NPZ):
        if not p.exists():
            print("MISSING_REGEN_INPUT", p, file=sys.stderr); return 2
    if not K8_DIR.exists():
        print("MISSING_REGEN_INPUT", K8_DIR, file=sys.stderr); return 2

    correct_main = np.load(CACHE)["correct"].astype(bool)
    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    if dom_score.shape[0] != N_PROBLEMS or correct_main.shape[0] != N_PROBLEMS:
        print("MISSING_REGEN_INPUT shape mismatch", file=sys.stderr); return 2

    mean_q = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    correct_rate = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    found = np.zeros(N_PROBLEMS, dtype=bool)
    step_weighted_any = False
    total_steps = 0

    for i in range(N_PROBLEMS):
        f = K8_DIR / f"problem_{i:03d}.npz"
        if not f.exists():
            continue
        blob = np.load(f, allow_pickle=True)
        if "correct" not in blob.files:
            continue
        c8 = np.atleast_1d(blob["correct"]).astype(bool).ravel()
        if c8.size == 0:
            continue
        texts = _extract_texts(blob)
        mq, sw, nsteps = backprop_mean_q(c8, texts)
        mean_q[i] = mq
        correct_rate[i] = float(c8.mean())
        found[i] = True
        step_weighted_any = step_weighted_any or sw
        total_steps += nsteps

    n_found = int(found.sum())
    if n_found == 0:
        print("MISSING_REGEN_INPUT no per-problem K=8 caches readable", file=sys.stderr)
        return 2

    # DoM percentile over the full problem set, then restrict to found problems.
    dom_pct_full = (rankdata(dom_score) - 1.0) / (N_PROBLEMS - 1.0)
    idx = np.flatnonzero(found)
    dp = dom_pct_full[idx]
    mq = mean_q[idx]
    cr = correct_rate[idx]
    cm = correct_main[idx]

    sp_q, sp_q_p = spearmanr(dp, mq)
    pe_q, pe_q_p = pearsonr(dp, mq)
    sp_cr, sp_cr_p = spearmanr(dp, cr)
    pe_cr, pe_cr_p = pearsonr(dp, cr)

    # Familiarity-vs-decomposability discriminator: does DoM track Q-shape
    # beyond the raw correct-rate? With step-weighting mean_q can deviate from
    # correct_rate; without it the two coincide and this partial is undefined-ish.
    partial_q_given_cr = partial_corr(dp, mq, cr)
    q_shape_var = float(np.var(mq - cr))

    out = {
        "experiment": "P11-FE356",
        "n_problems_found": n_found,
        "step_shape_available": bool(step_weighted_any),
        "total_steps_segmented": int(total_steps),
        "step_delim": "\\n\\n",
        "spearman_dom_pct_mean_q": float(sp_q),
        "spearman_dom_pct_mean_q_p": float(sp_q_p),
        "pearson_dom_pct_mean_q": float(pe_q),
        "pearson_dom_pct_mean_q_p": float(pe_q_p),
        "spearman_dom_pct_correct_rate": float(sp_cr),
        "spearman_dom_pct_correct_rate_p": float(sp_cr_p),
        "pearson_dom_pct_correct_rate": float(pe_cr),
        "pearson_dom_pct_correct_rate_p": float(pe_cr_p),
        "partial_corr_dom_pct_mean_q_given_correct_rate": float(partial_q_given_cr),
        "q_shape_variance_mean_q_minus_correct_rate": q_shape_var,
        "auroc_dom_score_main_correct": float(auroc(dom_score[idx], cm)),
        "mean_q_mean": float(np.mean(mq)),
        "mean_q_std": float(np.std(mq)),
        "correct_rate_mean": float(np.mean(cr)),
        "note": (
            "mean_q is rStar-Math Eq.2 back-prop on linear K=8 trajectories; "
            "Q-shape (mean_q != correct_rate) only exists when generation text "
            "is cached for \\n\\n step-weighting. If step_shape_available is "
            "False, mean_q == correct_rate and partial_corr is uninformative."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())