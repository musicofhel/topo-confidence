"""P11-FE714 — ReBalance unsupervised steering vector vs the supervised correctness DoM.

ReBalance constructs an over/under-thinking steering direction
v = (mu^O - mu^U) / ||mu^O - mu^U|| from L19 hidden states, *without* using
correctness labels. This experiment tests whether that unsupervised axis is
aligned with:
  - the supervised prefill correctness DoM (F-2): mu_correct - mu_incorrect,
  - the final-token DoM (F-3, the rotation-revealing partner of prefill DoM),
  - the CoE principal direction (PC1 of the centered L19 covariance).

The cached cache only carries L19 *prefill* states + correctness + seq_len, so
the over/under-thinking grouping is proxied behaviourally by reasoning length:
the longest-tertile traces are treated as the over-thinking pool (O) and the
shortest-tertile as the under-thinking pool (U). This is label-free w.r.t.
correctness — exactly the property H-12 cares about.

Verdict logic (on |cos(v, prefill_DoM_F2)|):
  >= 0.30  -> ReBalance approximates the supervised correctness direction
             unsupervised: H-12 CONFIRMED, F-2's labelled-data need weakened.
  <  0.10  -> over/under-thinking is a NEW axis distinct from correctness: a
             second behavioural lever to compose with H-1 steering.
  else     -> AMBIGUOUS.
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
# Optional final-token cache (rotation partner of the prefill DoM, F-3). The
# documented schema only guarantees prefill states, so this is best-effort.
FINAL_NPZ = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz"
OUT_JSON = ROOT / "pathway11_h100/rebalance_steering/results.json"

SEED = 9999


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def cos(a: np.ndarray, b: np.ndarray):
    if a is None or b is None:
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return None
    return float(np.dot(a, b) / (na * nb))


def rebalance_vector(X: np.ndarray, seq_len: np.ndarray) -> np.ndarray:
    """Label-free over/under-thinking axis via length tertiles.

    O = longest-third traces (over-thinking), U = shortest-third (under-thinking).
    """
    order = np.argsort(seq_len, kind="stable")
    third = len(order) // 3
    u_idx = order[:third]            # shortest -> under-thinking
    o_idx = order[len(order) - third:]  # longest -> over-thinking
    mu_o = X[o_idx].mean(axis=0)
    mu_u = X[u_idx].mean(axis=0)
    return unit(mu_o - mu_u)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    seq_len = blob["seq_len"].astype(np.float64)
    assert X.shape == (500, 1536) and y.shape == (500,)

    # ReBalance unsupervised steering vector (length-tertile proxy for O/U).
    v = rebalance_vector(X, seq_len)

    # Supervised prefill correctness DoM (F-2).
    prefill_dom = unit(X[y].mean(axis=0) - X[~y].mean(axis=0))

    # Sanity: cached DoM scores should rank-agree with our refit DoM projection.
    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    proj = X @ prefill_dom
    auroc_refit_dom = auroc(proj, y)
    auroc_cached_dom = auroc(dom_score, y)

    # CoE principal direction: PC1 of the centered L19 covariance.
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc / max(len(Xc) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    coe_pc1 = unit(evecs[:, -1])

    # Final-token DoM (F-3 rotation partner) — best-effort, may be absent.
    final_dom = None
    final_source = None
    if FINAL_NPZ.exists():
        fblob = np.load(FINAL_NPZ)
        key = next(
            (k for k in ("final", "prefill_final", "final_token", "hidden")
             if k in fblob.files),
            None,
        )
        if key is not None:
            Xf = fblob[key].astype(np.float64)
            if Xf.shape == X.shape:
                final_dom = unit(Xf[y].mean(axis=0) - Xf[~y].mean(axis=0))
                final_source = f"{FINAL_NPZ.name}:{key}"

    cos_v_prefill_dom = cos(v, prefill_dom)
    cos_v_final_dom = cos(v, final_dom)
    cos_v_coe = cos(v, coe_pc1)
    cos_prefill_coe = cos(prefill_dom, coe_pc1)

    # Verdict on |cos(v, prefill_DoM_F2)|.
    if cos_v_prefill_dom is None:
        verdict = "INCONCLUSIVE_NO_VECTOR"
    else:
        a = abs(cos_v_prefill_dom)
        if a >= 0.30:
            verdict = "H12_CONFIRMED_rebalance_approximates_correctness_DoM"
        elif a < 0.10:
            verdict = "NEW_AXIS_overunder_thinking_distinct_from_correctness"
        else:
            verdict = "AMBIGUOUS"

    out = {
        "experiment": "P11-FE714",
        "n": int(len(y)),
        "ou_grouping": "length_tertile_proxy",
        "cos_v_prefill_dom_F2": cos_v_prefill_dom,
        "abs_cos_v_prefill_dom_F2": (
            None if cos_v_prefill_dom is None else abs(cos_v_prefill_dom)
        ),
        "cos_v_final_token_dom_F3": cos_v_final_dom,
        "cos_v_coe_pc1": cos_v_coe,
        "cos_prefill_dom_coe_pc1": cos_prefill_coe,
        "final_token_dom_source": final_source,
        "final_token_dom_available": final_dom is not None,
        "auroc_refit_prefill_dom": auroc_refit_dom,
        "auroc_cached_prefill_dom": auroc_cached_dom,
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())