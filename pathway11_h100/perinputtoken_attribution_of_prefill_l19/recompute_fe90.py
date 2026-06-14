"""P11-FE90 — Per-input-token attribution of the prefill L19 DoM projection.

Cheap, gradient-only baseline for refutation #1 in the FE90 brief: is the
F-2 prefill L19 DoM signal a *global* geometric direction, or is it carried by
a small set of input tokens?

For the DoM read-out f(h) = h · DoM the gradient wrt the hidden state is just
DoM, so grad×input on a token's L19 hidden state h_t is h_t ⊙ DoM, which sums
over dims to the per-token contribution a_t = h_t · DoM. Under the cheap
linear-decomposition assumption (prefill ≈ mean_t h_t, i.e. ignore cross-token
attention mixing — that full DecompX-style propagation is deferred to FE74),
the problem-level projection is mean_t a_t, so {a_t} is a faithful per-input-
token attribution of the DoM score.

We report, per problem:
  (a) attribution entropy over tokens (nats, and normalized by log T),
  (b) top-5% token mass fraction,
and across the cohort:
  (c) D-bucket (bottom-quartile DoM score) vs A-bucket (top-quartile) entropy
      distributions, with a Mann-Whitney U test, and
  (d) example-level highlights of the highest-|a_t| tokens for the 5 highest-
      and 5 lowest-DoM-score problems.

Requires a per-token L19 cache (hidden states + token strings) that is NOT the
single-vector main cache; if it is absent the script reports MISSING_REGEN_INPUT
and exits 2 (the per-token dump needs a model forward pass, done elsewhere).
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
# Per-token L19 dump: one npz per problem with 'hidden' (T,1536) and 'tokens' (T,).
PER_TOKEN_DIR = ROOT / "pathway11_h100/prefill_inversion/cache/per_token"
OUT_JSON = ROOT / "pathway11_h100/token_attribution/results.json"

N_PROBLEMS = 500
TOP_FRAC = 0.05
N_HIGHLIGHTS = 5
N_TOP_TOKENS = 8
EPS = 1e-12


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def entropy_nats(weights: np.ndarray) -> float:
    """Shannon entropy (nats) of a non-negative attribution-mass vector."""
    total = weights.sum()
    if total <= EPS:
        return 0.0
    p = weights / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def top_mass_fraction(weights: np.ndarray, frac: float) -> float:
    total = weights.sum()
    if total <= EPS or len(weights) == 0:
        return float("nan")
    k = max(1, int(np.ceil(frac * len(weights))))
    top = np.sort(weights)[::-1][:k]
    return float(top.sum() / total)


def as_str(tok) -> str:
    if isinstance(tok, bytes):
        return tok.decode("utf-8", "replace")
    return str(tok)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2
    if not PER_TOKEN_DIR.is_dir():
        print("MISSING_REGEN_INPUT", PER_TOKEN_DIR, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (N_PROBLEMS, 1536) and y.shape == (N_PROBLEMS,)
    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert dom_score.shape == (N_PROBLEMS,)

    # Global DoM direction (mean-of-means difference), unit-normalized.
    dom = X[y].mean(axis=0) - X[~y].mean(axis=0)
    nrm = np.linalg.norm(dom)
    if nrm < EPS:
        print("MISSING_REGEN_INPUT", "degenerate DoM", file=sys.stderr)
        return 2
    dom = dom / nrm

    per_token_files = sorted(PER_TOKEN_DIR.glob("problem_*.npz"))
    if not per_token_files:
        print("MISSING_REGEN_INPUT", PER_TOKEN_DIR, file=sys.stderr)
        return 2

    entropies = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    norm_entropies = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    top5_mass = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    seq_tokens = np.zeros(N_PROBLEMS, dtype=np.int64)
    attribution_by_problem: dict[int, tuple[np.ndarray, list[str]]] = {}

    n_loaded = 0
    for f in per_token_files:
        try:
            pid = int(f.stem.split("_")[-1])
        except ValueError:
            continue
        if not (0 <= pid < N_PROBLEMS):
            continue
        d = np.load(f, allow_pickle=True)
        if "hidden" not in d or "tokens" not in d:
            continue
        H = d["hidden"].astype(np.float64)
        toks = [as_str(t) for t in d["tokens"]]
        if H.ndim != 2 or H.shape[1] != 1536 or H.shape[0] != len(toks) or H.shape[0] == 0:
            continue

        # grad×input per token for f(h)=h·DoM reduces to a_t = h_t · DoM.
        a = H @ dom
        mag = np.abs(a)
        entropies[pid] = entropy_nats(mag)
        norm_entropies[pid] = entropies[pid] / np.log(len(toks)) if len(toks) > 1 else 0.0
        top5_mass[pid] = top_mass_fraction(mag, TOP_FRAC)
        seq_tokens[pid] = len(toks)
        attribution_by_problem[pid] = (a, toks)
        n_loaded += 1

    if n_loaded == 0:
        print("MISSING_REGEN_INPUT", "no usable per-token files", file=sys.stderr)
        return 2

    valid = ~np.isnan(entropies)

    # (c) D-bucket (bottom-quartile DoM) vs A-bucket (top-quartile DoM) entropy.
    order = np.argsort(dom_score)
    q = N_PROBLEMS // 4
    d_idx = np.array([i for i in order[:q] if valid[i]])
    a_idx = np.array([i for i in order[::-1][:q] if valid[i]])

    def bucket_stats(idx: np.ndarray) -> dict:
        if len(idx) == 0:
            return {"n": 0, "mean_entropy": None, "std_entropy": None,
                    "mean_norm_entropy": None, "mean_top5_mass": None}
        return {
            "n": int(len(idx)),
            "mean_entropy": float(np.mean(entropies[idx])),
            "std_entropy": float(np.std(entropies[idx])),
            "mean_norm_entropy": float(np.mean(norm_entropies[idx])),
            "mean_top5_mass": float(np.mean(top5_mass[idx])),
        }

    mw_u = mw_p = None
    if len(d_idx) > 0 and len(a_idx) > 0:
        try:
            from scipy.stats import mannwhitneyu
            u, p = mannwhitneyu(entropies[d_idx], entropies[a_idx], alternative="two-sided")
            mw_u, mw_p = float(u), float(p)
        except Exception:
            pass

    # (d) Example highlights for the 5 highest- and 5 lowest-DoM-score problems
    #     that have per-token attributions available.
    def highlight(pids: list[int]) -> list[dict]:
        out = []
        for pid in pids:
            if pid not in attribution_by_problem:
                continue
            a, toks = attribution_by_problem[pid]
            top = np.argsort(np.abs(a))[::-1][:N_TOP_TOKENS]
            out.append({
                "problem": int(pid),
                "dom_score": float(dom_score[pid]),
                "correct": bool(y[pid]),
                "entropy": float(entropies[pid]),
                "top5_mass": float(top5_mass[pid]),
                "top_tokens": [
                    {"token": toks[t], "attribution": float(a[t]), "pos": int(t)}
                    for t in top
                ],
            })
        return out

    high_pids = [int(i) for i in order[::-1] if i in attribution_by_problem][:N_HIGHLIGHTS]
    low_pids = [int(i) for i in order if i in attribution_by_problem][:N_HIGHLIGHTS]

    out = {
        "experiment": "P11-FE90",
        "description": "per-input-token grad×input attribution of prefill L19 DoM projection",
        "n_problems_with_attribution": int(n_loaded),
        "auroc_dom_score": float(auroc(dom_score, y)),
        "cohort": {
            "mean_entropy": float(np.nanmean(entropies)),
            "median_entropy": float(np.nanmedian(entropies)),
            "mean_norm_entropy": float(np.nanmean(norm_entropies)),
            "mean_top5_mass": float(np.nanmean(top5_mass)),
            "median_top5_mass": float(np.nanmedian(top5_mass)),
            "mean_seq_tokens": float(np.mean(seq_tokens[valid])),
        },
        "buckets": {
            "definition": "A-bucket = top-quartile DoM score, D-bucket = bottom-quartile DoM score",
            "A_bucket": bucket_stats(a_idx),
            "D_bucket": bucket_stats(d_idx),
            "mannwhitney_u_entropy": mw_u,
            "mannwhitney_p_entropy": mw_p,
        },
        "highlights": {
            "most_correct_predicted": highlight(high_pids),
            "most_incorrect_predicted": highlight(low_pids),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())