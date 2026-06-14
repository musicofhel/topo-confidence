"""P11-FE238 — GRIDE depth-axis intrinsic-dimension sweep vs the F-2 DoM layer.

Cheng et al. report a depth-axis intrinsic-dimension (ID) peak across transformer
LMs that coincides with abstract syntactic/semantic processing; F-1 reports a
time-axis breathing peak via participation ratio. This script runs the GRIDE
(Generalized Ratios ID Estimator, Denti et al.) over the cached Qwen-2.5-1.5B
prefill point cloud at every available layer, locates the depth-ID-peak window,
and asks whether L19 (the F-2 DoM layer) sits inside it.

It then computes a per-problem *local* GRIDE-ID at the peak layer (and at L19)
and tests it as a fully label-free correctness predictor, comparing AUROC against
the F-2 supervised DoM ceiling (0.7731). This probes Refutation 1 (depth vs time
axis split) and Refutation 3 (ID carries signal where PH was null).

Layer activations: if an all-layers prefill cache is present we sweep 0-28;
otherwise we degrade gracefully to the single cached L19 slice and say so.
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
from scipy.optimize import minimize_scalar
from scipy.special import betaln
from sklearn.neighbors import NearestNeighbors

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
ALL_LAYERS_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_all_layers.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/gride_id/results.json"

L19 = 19
N1 = 2          # GRIDE inner neighbor order
N2 = 4          # GRIDE outer neighbor order (n2 = 2 * n1)
F2_SUPERVISED = 0.7731


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def signed_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC taking the better of the score and its negation (label-free probe)."""
    a = auroc(scores, labels)
    if np.isnan(a):
        return a
    return max(a, 1.0 - a)


def gride_ratios(X: np.ndarray, n1: int, n2: int) -> np.ndarray:
    """mu_i = r_{i,n2} / r_{i,n1}, the GRIDE generalized distance ratios (>1)."""
    nbrs = NearestNeighbors(n_neighbors=n2 + 1).fit(X)
    dist, _ = nbrs.kneighbors(X)  # column 0 is the self-distance (0)
    r1 = dist[:, n1]
    r2 = dist[:, n2]
    good = (r1 > 0) & np.isfinite(r1) & np.isfinite(r2)
    mu = np.full(X.shape[0], np.nan, dtype=np.float64)
    mu[good] = r2[good] / r1[good]
    mu[good] = np.maximum(mu[good], 1.0 + 1e-12)
    return mu


def gride_nll(d: float, logmu: np.ndarray, n1: int, n2: int) -> float:
    """Negative log-likelihood of the GRIDE ratio distribution for dimension d."""
    if d <= 0:
        return np.inf
    # log(mu^d - 1) computed stably as d*logmu + log1p(-exp(-d*logmu))
    a = d * logmu
    log_pow_m1 = a + np.log1p(-np.exp(-a))
    ll = (
        np.log(d)
        - betaln(n2 - n1, n1)
        + (n2 - n1 - 1) * log_pow_m1
        - (d * (n2 - 1) + 1.0) * logmu
    )
    return -float(np.sum(ll))


def gride_id_global(mu: np.ndarray, n1: int, n2: int) -> float:
    """Global GRIDE MLE for the intrinsic dimension over all valid ratios."""
    valid = np.isfinite(mu) & (mu > 1.0)
    logmu = np.log(mu[valid])
    if logmu.size < 10:
        return float("nan")
    res = minimize_scalar(
        gride_nll, bounds=(0.1, 500.0), method="bounded", args=(logmu, n1, n2)
    )
    return float(res.x) if res.success else float("nan")


def gride_id_local(mu: np.ndarray, n1: int, n2: int) -> np.ndarray:
    """Per-point local ID estimate d_i = log(n2/n1) / log(mu_i).

    From r_n ~ n^{1/d}: mu_i = (n2/n1)^{1/d_i}, so larger mu => lower local ID.
    """
    out = np.full(mu.shape[0], np.nan, dtype=np.float64)
    valid = np.isfinite(mu) & (mu > 1.0)
    out[valid] = np.log(n2 / n1) / np.log(mu[valid])
    return out


def load_layer_activations():
    """Return (layers, dict[layer]->X(500,1536), full_sweep_bool) or None if missing."""
    if ALL_LAYERS_CACHE.exists():
        blob = np.load(ALL_LAYERS_CACHE)
        key = next(
            (k for k in ("prefill_all", "hidden_all", "prefill", "states") if k in blob),
            None,
        )
        if key is not None and blob[key].ndim == 3:
            arr = np.asarray(blob[key], dtype=np.float64)
            shape = arr.shape
            try:
                samp_ax = shape.index(500)
            except ValueError:
                samp_ax = int(np.argmax(shape))
            feat_candidates = [i for i in range(3) if i != samp_ax and shape[i] == 1536]
            feat_ax = feat_candidates[0] if feat_candidates else (
                [i for i in range(3) if i != samp_ax][-1]
            )
            layer_ax = [i for i in range(3) if i not in (samp_ax, feat_ax)][0]
            arr = np.moveaxis(arr, (layer_ax, samp_ax, feat_ax), (0, 1, 2))
            n_layers = arr.shape[0]
            layers = list(range(n_layers))
            return layers, {l: arr[l] for l in layers}, True

    if CACHE.exists():
        blob = np.load(CACHE)
        X = blob["prefill"].astype(np.float64)
        return [L19], {L19: X}, False

    return None


def main() -> int:
    loaded = load_layer_activations()
    if loaded is None:
        print("MISSING_REGEN_INPUT", ALL_LAYERS_CACHE, "or", CACHE, file=sys.stderr)
        return 2
    layers, acts, full_sweep = loaded

    # Labels (and a DoM reference) come from the canonical L19 cache when present.
    if CACHE.exists():
        cache = np.load(CACHE)
        correct = cache["correct"].astype(bool)
    else:
        blob = np.load(ALL_LAYERS_CACHE)
        if "correct" not in blob:
            print("MISSING_REGEN_INPUT", "correct labels", file=sys.stderr)
            return 2
        correct = blob["correct"].astype(bool)

    auroc_dom_ref = float("nan")
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        if dom_score.shape[0] == correct.shape[0]:
            auroc_dom_ref = signed_auroc(dom_score, correct)

    # --- depth-axis global ID sweep ---
    per_layer = []
    global_ids = {}
    for l in layers:
        mu = gride_ratios(acts[l], N1, N2)
        gid = gride_id_global(mu, N1, N2)
        global_ids[l] = gid
        per_layer.append({"layer": int(l), "gride_id_global": gid})

    valid_layers = {l: g for l, g in global_ids.items() if np.isfinite(g)}
    if not valid_layers:
        print("GRIDE_ID_ESTIMATION_FAILED", file=sys.stderr)
        return 1
    peak_layer = int(max(valid_layers, key=valid_layers.get))
    peak_id = float(valid_layers[peak_layer])

    # depth-ID-peak window: layers within 5% of the peak ID
    window = sorted(
        int(l) for l, g in valid_layers.items() if g >= 0.95 * peak_id
    )
    l19_in_window = L19 in window

    # --- per-problem local ID at the peak layer (label-free predictor) ---
    mu_peak = gride_ratios(acts[peak_layer], N1, N2)
    localid_peak = gride_id_local(mu_peak, N1, N2)
    finite_peak = np.isfinite(localid_peak)
    auroc_peak_signed = auroc(
        np.where(finite_peak, localid_peak, np.nanmedian(localid_peak[finite_peak])),
        correct,
    )
    auroc_peak_best = signed_auroc(
        np.where(finite_peak, localid_peak, np.nanmedian(localid_peak[finite_peak])),
        correct,
    )

    # --- and at L19 specifically, if available ---
    auroc_l19_best = float("nan")
    l19_global_id = float("nan")
    if L19 in acts:
        l19_global_id = float(global_ids.get(L19, float("nan")))
        mu_l19 = gride_ratios(acts[L19], N1, N2)
        localid_l19 = gride_id_local(mu_l19, N1, N2)
        finite_l19 = np.isfinite(localid_l19)
        filled = np.where(
            finite_l19, localid_l19, np.nanmedian(localid_l19[finite_l19])
        )
        auroc_l19_best = signed_auroc(filled, correct)

    out = {
        "experiment": "P11-FE238",
        "estimator": "GRIDE",
        "n1": N1,
        "n2": N2,
        "full_depth_sweep": bool(full_sweep),
        "n_layers_swept": len(layers),
        "per_layer": per_layer,
        "peak_layer": peak_layer,
        "peak_gride_id_global": peak_id,
        "depth_id_peak_window": window,
        "l19_in_peak_window": bool(l19_in_window),
        "l19_global_gride_id": l19_global_id,
        "auroc_localid_peak_signed": float(auroc_peak_signed),
        "auroc_localid_peak_best_of_sign": float(auroc_peak_best),
        "auroc_localid_l19_best_of_sign": float(auroc_l19_best),
        "auroc_dom_reference_best_of_sign": auroc_dom_ref,
        "f2_supervised_reference": F2_SUPERVISED,
        "note": (
            "Full 0-28 depth sweep." if full_sweep else
            "Only the cached L19 slice was available; depth-axis peak comparison "
            "is degenerate (single layer). Regenerate the all-layers prefill cache "
            "to test Refutation 1 (depth-vs-time axis split)."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())