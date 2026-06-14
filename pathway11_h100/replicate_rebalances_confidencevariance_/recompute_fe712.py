"""P11-FE712 — ReBalance confidence-variance overthinking/underthinking labels
vs our F-7 A/B/C/D buckets, on cached MATH-500 Stage 2 traces (Qwen-2.5-1.5B).

ReBalance constructs a behavioural axis (overthinking O vs underthinking U)
from step-wise confidence dynamics *without* correctness labels. For each
reasoning step we compute the step confidence

    c_s = exp(mean(log p_max))                 # mean over the step's tokens

then the windowed variance Var(c_s) over a sliding window W=2, and classify
each step O / U / moderate by global quantile thresholds on that variance.
The problem-level label is the most-frequent step bucket per problem.

We cross-tabulate that label-free O/U/moderate partition against our F-7
A/B/C/D buckets (here reconstructed as DoM-score quartiles: A = top quartile
... D = bottom quartile) and report the contingency table plus a chi-square /
Cramér's-V alignment. If the O partition lines up with the D bucket, the
D-bucket signature (F-7) has a label-free confidence-dynamics explanation and
the supervised correctness DoM is approximating something simpler.

Inputs required (all under topo-confidence/):
  - per-problem Stage-2 confidence traces (token/step log p_max)
  - phase2_prefill_dom.npz (DoM scores, for A/B/C/D buckets)
  - m15b_prefill.npz (correctness labels, for sanity reporting)
If the Stage-2 trace cache is absent the script prints MISSING_REGEN_INPUT
and returns 2 (it is a token-level cache, not part of the prefill NPZs).
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
# Stage-2 reasoning traces: per-problem step log-prob arrays. Candidate layouts
# are probed in _load_traces(); none are part of the documented prefill schema.
TRACE_DIR = ROOT / "pathway11_h100/data/stage2_traces"
TRACE_NPZ = ROOT / "pathway11_h100/data/stage2_traces.npz"
OUT_JSON = ROOT / "pathway11_h100/rebalance_overthinking/results.json"

N_PROBLEMS = 500
W = 2                       # variance window (consecutive steps)
SEED = 9999

# Candidate per-step log-prob keys (mean log p_max already aggregated per step).
STEP_LOGP_KEYS = ("step_logp_mean", "step_logp", "logp_step", "step_mean_logp")
# Candidate per-token log p_max keys (we mean them per step ourselves if the
# trace also carries step boundaries).
TOKEN_LOGP_KEYS = ("logp_max", "log_pmax", "token_logp_max", "logp")
STEP_BOUND_KEYS = ("step_starts", "step_boundaries", "step_offsets")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def _first_key(blob, keys):
    for k in keys:
        if k in blob:
            return k
    return None


def _step_logp_from_blob(blob) -> np.ndarray | None:
    """Return a 1-D per-step mean(log p_max) array for one problem, or None."""
    sk = _first_key(blob, STEP_LOGP_KEYS)
    if sk is not None:
        arr = np.asarray(blob[sk], dtype=np.float64).ravel()
        return arr if arr.size else None
    tk = _first_key(blob, TOKEN_LOGP_KEYS)
    if tk is None:
        return None
    tok = np.asarray(blob[tk], dtype=np.float64).ravel()
    if tok.size == 0:
        return None
    bk = _first_key(blob, STEP_BOUND_KEYS)
    if bk is None:
        # No step segmentation: treat each token as its own step.
        return tok
    bounds = np.asarray(blob[bk], dtype=np.int64).ravel()
    bounds = bounds[(bounds >= 0) & (bounds < tok.size)]
    bounds = np.unique(np.concatenate([[0], bounds, [tok.size]]))
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            out.append(float(tok[a:b].mean()))
    return np.asarray(out, dtype=np.float64) if out else None


def _load_traces() -> dict[int, np.ndarray]:
    """Map problem index -> per-step mean(log p_max) array. Empty if no cache."""
    traces: dict[int, np.ndarray] = {}
    if TRACE_DIR.exists():
        for i in range(N_PROBLEMS):
            f = TRACE_DIR / f"problem_{i:03d}.npz"
            if not f.exists():
                continue
            with np.load(f) as blob:
                arr = _step_logp_from_blob(blob)
            if arr is not None and arr.size:
                traces[i] = arr
        if traces:
            return traces
    if TRACE_NPZ.exists():
        with np.load(TRACE_NPZ, allow_pickle=True) as blob:
            # Either a 2-D padded array + lengths, or an object array of traces.
            sk = _first_key(blob, STEP_LOGP_KEYS) or _first_key(blob, TOKEN_LOGP_KEYS)
            if sk is not None:
                raw = blob[sk]
                lens = blob["lengths"].astype(int) if "lengths" in blob else None
                if raw.dtype == object:
                    for i, a in enumerate(raw):
                        a = np.asarray(a, dtype=np.float64).ravel()
                        if a.size:
                            traces[i] = a
                else:
                    raw = np.asarray(raw, dtype=np.float64)
                    for i in range(raw.shape[0]):
                        row = raw[i]
                        if lens is not None:
                            row = row[: lens[i]]
                        row = row[np.isfinite(row)]
                        if row.size:
                            traces[i] = row
    return traces


def windowed_var(c: np.ndarray, w: int) -> np.ndarray:
    """Variance over each sliding window of length w (population variance)."""
    if c.size < w:
        return np.array([float(np.var(c))]) if c.size else np.array([])
    return np.array([float(np.var(c[i : i + w])) for i in range(c.size - w + 1)])


def cramers_v(table: np.ndarray) -> tuple[float, float, float]:
    """Cramér's V, chi-square stat, p-value for a contingency table."""
    try:
        from scipy.stats import chi2_contingency
    except Exception:
        return float("nan"), float("nan"), float("nan")
    sub = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    if sub.shape[0] < 2 or sub.shape[1] < 2:
        return float("nan"), float("nan"), float("nan")
    chi2, p, _, _ = chi2_contingency(sub)
    n = sub.sum()
    k = min(sub.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if n > 0 and k > 0 else float("nan")
    return v, float(chi2), float(p)


def main() -> int:
    for required in (CACHE, DOM_NPZ):
        if not required.exists():
            print("MISSING_REGEN_INPUT", required, file=sys.stderr)
            return 2

    traces = _load_traces()
    if not traces:
        print("MISSING_REGEN_INPUT", TRACE_DIR, "/", TRACE_NPZ, file=sys.stderr)
        return 2

    correct = np.load(CACHE)["correct"].astype(bool)
    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert correct.shape == (N_PROBLEMS,) and dom_score.shape == (N_PROBLEMS,)

    probs = sorted(traces)

    # --- step-wise confidence + windowed variance, pooled across problems ---
    per_problem_vars: dict[int, np.ndarray] = {}
    per_problem_conf: dict[int, float] = {}
    all_vars = []
    for i in probs:
        logp = traces[i]
        c = np.exp(logp)                       # c_s = exp(mean(log p_max))
        v = windowed_var(c, W)
        if v.size == 0:
            continue
        per_problem_vars[i] = v
        per_problem_conf[i] = float(c.mean())
        all_vars.append(v)
    if not per_problem_vars:
        print("MISSING_REGEN_INPUT", "no usable step traces", file=sys.stderr)
        return 2
    probs = sorted(per_problem_vars)
    pooled = np.concatenate(all_vars)

    # Global quantile thresholds on the windowed variance: high-variance steps
    # are the "overthinking" wandering signature (O), low-variance steps the
    # "underthinking" premature-collapse signature (U), the rest moderate.
    q_lo, q_hi = float(np.quantile(pooled, 1.0 / 3.0)), float(np.quantile(pooled, 2.0 / 3.0))

    def step_buckets(v: np.ndarray) -> np.ndarray:
        b = np.full(v.shape, 1, dtype=np.int64)   # 1 = moderate
        b[v >= q_hi] = 2                           # 2 = O (overthinking)
        b[v <= q_lo] = 0                           # 0 = U (underthinking)
        return b

    OU_LABELS = {0: "U", 1: "moderate", 2: "O"}
    prob_ou = {}        # problem -> 0/1/2 (most-frequent step bucket)
    for i in probs:
        b = step_buckets(per_problem_vars[i])
        counts = np.bincount(b, minlength=3)
        prob_ou[i] = int(np.argmax(counts))

    # --- F-7 A/B/C/D buckets reconstructed as DoM-score quartiles --------
    # A = highest-DoM quartile (most confident/correct) ... D = lowest.
    sub_dom = dom_score[probs]
    edges = np.quantile(sub_dom, [0.25, 0.50, 0.75])
    # rank-from-top so larger DoM -> bucket A(0)
    abcd = np.full(len(probs), 3, dtype=np.int64)         # D
    abcd[sub_dom > edges[0]] = 2                           # C
    abcd[sub_dom > edges[1]] = 1                           # B
    abcd[sub_dom > edges[2]] = 0                           # A
    ABCD_LABELS = {0: "A", 1: "B", 2: "C", 3: "D"}
    prob_abcd = {i: int(abcd[j]) for j, i in enumerate(probs)}

    # --- cross-tabulation: rows = O/U/moderate (0,1,2), cols = A,B,C,D ----
    table = np.zeros((3, 4), dtype=np.int64)
    for i in probs:
        table[prob_ou[i], prob_abcd[i]] += 1

    contingency = {
        OU_LABELS[r]: {ABCD_LABELS[c]: int(table[r, c]) for c in range(4)}
        for r in range(3)
    }

    v, chi2, pval = cramers_v(table)

    # Targeted alignment: does the O row concentrate in D, and U in A?
    o_row = table[2]
    u_row = table[0]
    o_frac_in_D = float(o_row[3] / o_row.sum()) if o_row.sum() else float("nan")
    u_frac_in_A = float(u_row[0] / u_row.sum()) if u_row.sum() else float("nan")
    # Base rates for comparison.
    base_D = float((abcd == 3).mean())
    base_A = float((abcd == 0).mean())

    # Sanity: AUROC of (negative) mean step confidence as a correctness proxy.
    conf_vec = np.array([per_problem_conf[i] for i in probs], dtype=np.float64)
    corr_vec = correct[probs]
    auroc_conf = auroc(conf_vec, corr_vec)
    auroc_dom = auroc(dom_score, correct)

    out = {
        "experiment": "P11-FE712",
        "n_problems_with_traces": len(probs),
        "window_W": W,
        "var_quantile_thresholds": {"q33": q_lo, "q67": q_hi},
        "ou_bucket_counts": {OU_LABELS[r]: int(table[r].sum()) for r in range(3)},
        "abcd_bucket_counts": {ABCD_LABELS[c]: int(table[:, c].sum()) for c in range(4)},
        "contingency_OU_x_ABCD": contingency,
        "cramers_v": v,
        "chi2": chi2,
        "chi2_pvalue": pval,
        "O_fraction_in_D": o_frac_in_D,
        "D_base_rate": base_D,
        "U_fraction_in_A": u_frac_in_A,
        "A_base_rate": base_A,
        "auroc_mean_step_confidence": auroc_conf,
        "auroc_dom_reference": auroc_dom,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print("OK", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())