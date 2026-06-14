"""P11-FE183 — Output-probability vs hidden-state-geometry alignment.

Tests whether F-3's prefill/final DoM orthogonality (cos = 0.046) is downstream
of a shared output-likelihood signal. Computes Pearson and Spearman correlations
of three output-prob aggregations (mean-token-log-prob, min-token-log-prob,
product-token-log-prob = sum-of-log-probs) against the prefill DoM score and the
final-token DoM score over the 500 MATH-500 problems.

Decision rule: if BOTH DoM scores correlate strongly (|r| > 0.6) with the SAME
output-prob aggregation, the "two distinct geometric mechanisms" framing weakens
(H-13 head-level attribution and H-17 RoPE de-rotation lose motivation).
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
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
FINAL_DOM_JSON = ROOT / "scratch/pathway10_temporal_and_verifier_results.json"
LOGPROB_CANDIDATES = [
    ROOT / "pathway11_h100/data/token_logprobs.npz",
    ROOT / "pathway11_h100/prefill_gated_compute/token_logprobs.npz",
    ROOT / "pathway11_h100/data/k8_selfconsistency/token_logprobs.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/token_logprobs.npz",
]
OUT_JSON = ROOT / "pathway11_h100/output_prob_geometry/results.json"

N = 500
STRONG_THRESHOLD = 0.6


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def _find_500_array(obj, hints):
    """Recursively locate a length-500 numeric array whose key matches a hint."""
    found = []

    def walk(o, key=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, str(k))
        elif isinstance(o, list):
            if len(o) == N and all(isinstance(x, (int, float)) for x in o):
                lk = key.lower()
                score = sum(h in lk for h in hints)
                found.append((score, np.asarray(o, dtype=np.float64)))
            else:
                for x in o:
                    walk(x, key)

    walk(obj)
    if not found:
        return None
    found.sort(key=lambda t: t[0], reverse=True)
    if found[0][0] == 0:
        return None
    return found[0][1]


def load_final_dom():
    """Final-token L19 DoM score, length 500. From DOM_NPZ key or scratch JSON."""
    if DOM_NPZ.exists():
        blob = np.load(DOM_NPZ)
        for key in ("final_score", "final_token_score", "final_dom_score", "final"):
            if key in blob.files and blob[key].shape == (N,):
                return blob[key].astype(np.float64)
    if FINAL_DOM_JSON.exists():
        data = json.loads(FINAL_DOM_JSON.read_text())
        arr = _find_500_array(data, ("final", "dom"))
        if arr is not None:
            return arr
    return None


def load_logprob_aggregations():
    """Return dict mean/min/product (sum-of-log) aggregations, length 500, or None."""
    for path in LOGPROB_CANDIDATES:
        if not path.exists():
            continue
        blob = np.load(path, allow_pickle=True)
        files = set(blob.files)

        # Precomputed aggregations.
        agg = {}
        keymap = {
            "mean_token_log_prob": ("mean_logprob", "mean_token_logprob", "mean_logp"),
            "min_token_log_prob": ("min_logprob", "min_token_logprob", "min_logp"),
            "product_token_log_prob": ("sum_logprob", "sum_token_logprob", "total_logprob", "logprob_sum"),
        }
        for out_key, candidates in keymap.items():
            for c in candidates:
                if c in files and blob[c].shape == (N,):
                    agg[out_key] = blob[c].astype(np.float64)
                    break
        if len(agg) == 3:
            return agg

        # Ragged per-token log-probs (object array of per-problem token arrays).
        for tk in ("token_logprobs", "logprobs", "token_logp"):
            if tk in files:
                obj = blob[tk]
                if len(obj) == N:
                    means = np.empty(N, dtype=np.float64)
                    mins = np.empty(N, dtype=np.float64)
                    sums = np.empty(N, dtype=np.float64)
                    for i in range(N):
                        toks = np.asarray(obj[i], dtype=np.float64).ravel()
                        if toks.size == 0:
                            means[i] = mins[i] = sums[i] = np.nan
                        else:
                            means[i] = toks.mean()
                            mins[i] = toks.min()
                            sums[i] = toks.sum()
                    return {
                        "mean_token_log_prob": means,
                        "min_token_log_prob": mins,
                        "product_token_log_prob": sums,
                    }
    return None


def corr_pair(a: np.ndarray, b: np.ndarray) -> dict:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(mask.sum())}
    pr = float(pearsonr(a[mask], b[mask])[0])
    sr = float(spearmanr(a[mask], b[mask])[0])
    return {"pearson": pr, "spearman": sr, "n": int(mask.sum())}


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    correct = cache["correct"].astype(bool)
    assert correct.shape == (N,)

    prefill_dom = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert prefill_dom.shape == (N,)

    final_dom = load_final_dom()
    if final_dom is None:
        print("MISSING_REGEN_INPUT", "final-token DoM score", file=sys.stderr)
        return 2

    logprobs = load_logprob_aggregations()
    if logprobs is None:
        print("MISSING_REGEN_INPUT", "token log-probs", file=sys.stderr)
        return 2

    dom_scores = {"prefill_dom": prefill_dom, "final_dom": final_dom}

    correlations = {}
    for agg_name, agg_vals in logprobs.items():
        correlations[agg_name] = {}
        for dom_name, dom_vals in dom_scores.items():
            correlations[agg_name][dom_name] = corr_pair(agg_vals, dom_vals)

    # Shared-signal verdict: does any aggregation correlate strongly (|r|>0.6 on
    # both pearson and spearman) with BOTH DoM scores?
    shared = []
    for agg_name, by_dom in correlations.items():
        pre = by_dom["prefill_dom"]
        fin = by_dom["final_dom"]
        strong_pre = abs(pre["pearson"]) > STRONG_THRESHOLD and abs(pre["spearman"]) > STRONG_THRESHOLD
        strong_fin = abs(fin["pearson"]) > STRONG_THRESHOLD and abs(fin["spearman"]) > STRONG_THRESHOLD
        if strong_pre and strong_fin:
            shared.append(agg_name)

    out = {
        "experiment": "P11-FE183",
        "n_problems": N,
        "strong_threshold": STRONG_THRESHOLD,
        "cos_prefill_final_dom_reference": 0.046,
        "auroc_prefill_dom": auroc(prefill_dom, correct),
        "auroc_final_dom": auroc(final_dom, correct),
        "correlations": correlations,
        "shared_output_prob_aggregations": shared,
        "two_mechanisms_framing_weakens": bool(shared),
        "verdict": (
            "WEAKENS: both DoM scores reduce to a shared output-prob signal via "
            + ", ".join(shared)
            if shared
            else "HOLDS: no single output-prob aggregation strongly explains both DoM directions"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())