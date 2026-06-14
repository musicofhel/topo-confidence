"""P11-FE538 — D²HScore (intra-layer attention-weighted centroid dispersion +
inter-layer drift) head-to-head vs supervised L19 prefill DoM.

D²HScore is a label-free hidden-state uncertainty estimator: for each problem it
sums (a) the attention-weighted dispersion of token hidden states *within* each
layer and (b) the drift of the per-layer attention-weighted centroid *across*
layers. The paper reports 74.48 AUROC on GSM8K with Qwen1.5-7B-Instruct.

This script replicates D²HScore on our cached P11 1.5B Stage-2 per-layer NPZs for
the 500 MATH-500 problems and compares its AUROC head-to-head against:
  - F-2's supervised L19 prefill DoM (canonical 0.7731 OOF)
  - final-token L19 DoM (prior 0.7186)
  - CoE-60 (prior 0.811, loaded if a cached JSON exists)

It directly arbitrates two refutations:
  #2 label-free (D²HScore) vs supervised (DoM) — does the unsupervised score
     match the supervised direction at comparable scale?
  #4 single-layer (L19 DoM) vs all-layer (D²HScore uses every cached layer).

Per-layer per-token activations are NOT part of the committed L19 cache; if the
per-layer cache is absent the script prints MISSING_REGEN_INPUT and returns 2.
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
COE_JSON = ROOT / "pathway11_h100/results/coe60_results.json"
OUT_JSON = ROOT / "pathway11_h100/d2hscore/results.json"

# Candidate locations for the per-layer / per-token Stage-2 activations.
PERLAYER_DIRS = [
    ROOT / "pathway11_h100/data/per_layer_states",
    ROOT / "pathway11_h100/data/stage2_per_layer",
    ROOT / "pathway11_h100/prefill_inversion/cache/per_layer",
]
PERLAYER_PACKED = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_all_layers.npz",
    ROOT / "pathway11_h100/data/stage2_all_layers.npz",
]

# Prior numbers (1024-tok canonical) — comparison anchors, not recomputed here.
PRIOR_L19_DOM_OOF = 0.7731
PRIOR_FINAL_DOM = 0.7186
PRIOR_COE60 = 0.811

N_PROBLEMS = 500
_HIDDEN_KEYS = ("hidden_states", "hidden", "states", "layers", "h")
_ATTN_KEYS = ("attn_weights", "attn", "attention", "weights", "a")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def oriented_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Label-free scores have no a-priori sign; report the better orientation."""
    a = auroc(scores, labels)
    if np.isnan(a):
        return a
    return max(a, 1.0 - a)


def _pick(blob, keys):
    for k in keys:
        if k in blob.files:
            return blob[k]
    return None


def _load_problem(blob):
    """Return (hidden, attn) for one problem.

    hidden: (L, T, d) per-token per-layer, or (L, d) pooled-per-layer.
    attn:   (T,) attention weights over tokens, or None.
    """
    hidden = _pick(blob, _HIDDEN_KEYS)
    if hidden is None:
        return None, None
    hidden = np.asarray(hidden, dtype=np.float64)
    attn = _pick(blob, _ATTN_KEYS)
    if attn is not None:
        attn = np.asarray(attn, dtype=np.float64).ravel()
    return hidden, attn


def _d2h_for_problem(hidden: np.ndarray, attn) -> tuple[float, float]:
    """Return (intra_layer_dispersion, inter_layer_drift) for one problem.

    intra: mean over layers of attention-weighted token dispersion.
    inter: mean over adjacent layers of centroid drift (L2).
    """
    if hidden.ndim == 3:
        L, T, _ = hidden.shape
        if attn is None or attn.shape[0] != T:
            w = np.full(T, 1.0 / max(T, 1))
        else:
            w = np.clip(attn, 0.0, None)
            s = w.sum()
            w = w / s if s > 0 else np.full(T, 1.0 / max(T, 1))
        centroids = np.empty((L, hidden.shape[2]), dtype=np.float64)
        disp = np.empty(L, dtype=np.float64)
        for l in range(L):
            Hl = hidden[l]                      # (T, d)
            c = w @ Hl                           # (d,)
            centroids[l] = c
            sq = ((Hl - c) ** 2).sum(axis=1)     # (T,)
            disp[l] = float(w @ sq)
        intra = float(disp.mean())
    elif hidden.ndim == 2:
        # pooled-per-layer: (L, d) — no within-layer tokens, intra undefined.
        centroids = hidden
        intra = 0.0
    else:
        return float("nan"), float("nan")

    if centroids.shape[0] >= 2:
        drift = np.linalg.norm(np.diff(centroids, axis=0), axis=1)  # (L-1,)
        inter = float(drift.mean())
    else:
        inter = 0.0
    return intra, inter


def _load_perlayer():
    """Load per-problem per-layer activations from a directory or packed NPZ.

    Returns (intra[N], inter[N], n_layers, source_str) or None if unavailable.
    """
    # Directory of problem_NNN.npz files.
    for d in PERLAYER_DIRS:
        if d.is_dir():
            files = sorted(d.glob("problem_*.npz"))
            if len(files) >= N_PROBLEMS:
                intra = np.full(N_PROBLEMS, np.nan)
                inter = np.full(N_PROBLEMS, np.nan)
                n_layers = 0
                for f in files[:N_PROBLEMS]:
                    try:
                        idx = int(f.stem.split("_")[-1])
                    except ValueError:
                        continue
                    if not (0 <= idx < N_PROBLEMS):
                        continue
                    with np.load(f) as blob:
                        hidden, attn = _load_problem(blob)
                    if hidden is None:
                        continue
                    n_layers = max(n_layers, hidden.shape[0])
                    intra[idx], inter[idx] = _d2h_for_problem(hidden, attn)
                if np.isfinite(intra).any() or np.isfinite(inter).any():
                    return intra, inter, n_layers, str(d)

    # Packed NPZ: object array of per-problem (L, T, d) or dense (N, L, d).
    for p in PERLAYER_PACKED:
        if p.exists():
            with np.load(p, allow_pickle=True) as blob:
                arr = _pick(blob, _HIDDEN_KEYS)
                attn_all = _pick(blob, _ATTN_KEYS)
            if arr is None:
                continue
            intra = np.full(N_PROBLEMS, np.nan)
            inter = np.full(N_PROBLEMS, np.nan)
            n_layers = 0
            for i in range(min(N_PROBLEMS, len(arr))):
                hidden = np.asarray(arr[i], dtype=np.float64)
                attn = None
                if attn_all is not None and i < len(attn_all):
                    attn = np.asarray(attn_all[i], dtype=np.float64).ravel()
                if hidden.ndim < 2:
                    continue
                n_layers = max(n_layers, hidden.shape[0])
                intra[i], inter[i] = _d2h_for_problem(hidden, attn)
            if np.isfinite(intra).any() or np.isfinite(inter).any():
                return intra, inter, n_layers, str(p)
    return None


def _zscore(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-12:
        return np.zeros_like(x)
    return (x - m) / s


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    perlayer = _load_perlayer()
    if perlayer is None:
        print(
            "MISSING_REGEN_INPUT per-layer Stage-2 activations "
            f"(checked {[str(d) for d in PERLAYER_DIRS]} and "
            f"{[str(p) for p in PERLAYER_PACKED]})",
            file=sys.stderr,
        )
        return 2

    intra, inter, n_layers, source = perlayer

    cache = np.load(CACHE)
    correct = cache["correct"].astype(bool)
    assert correct.shape == (N_PROBLEMS,)

    # Restrict to problems with a valid D²HScore (finite components).
    valid = np.isfinite(intra) & np.isfinite(inter)
    if valid.sum() < 2:
        print("MISSING_REGEN_INPUT no valid per-layer problems", file=sys.stderr)
        return 2

    y = correct[valid]
    intra_v = intra[valid]
    inter_v = inter[valid]
    # Label-free combined score: standardize each component, sum (paper form).
    combined = _zscore(intra_v) + _zscore(inter_v)

    # Head-to-head supervised baseline on the SAME valid subset.
    dom_auroc_subset = float("nan")
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        dom_auroc_subset = auroc(dom_score[valid], y)

    coe60 = PRIOR_COE60
    if COE_JSON.exists():
        try:
            coe60 = float(json.loads(COE_JSON.read_text()).get("auroc", PRIOR_COE60))
        except (ValueError, KeyError):
            pass

    # D²HScore orientation: higher dispersion/drift => less confident => lower
    # correctness, so the raw score is expected to be anti-correlated.
    a_intra = oriented_auroc(intra_v, y)
    a_inter = oriented_auroc(inter_v, y)
    a_combined = oriented_auroc(combined, y)
    d2h_best = max(a for a in (a_intra, a_inter, a_combined) if not np.isnan(a))

    out = {
        "experiment": "P11-FE538",
        "description": "D2HScore intra/inter-layer hidden-state dispersion vs supervised L19 DoM",
        "source_cache": source,
        "n_layers": int(n_layers),
        "n_valid": int(valid.sum()),
        "n_total": int(N_PROBLEMS),
        # D²HScore (label-free, all-layer) — oriented AUROCs:
        "auroc_d2h_intra_oriented": a_intra,
        "auroc_d2h_inter_oriented": a_inter,
        "auroc_d2h_combined_oriented": a_combined,
        "auroc_d2h_best": float(d2h_best),
        # Raw (signed) AUROCs for the record:
        "auroc_d2h_intra_raw": auroc(intra_v, y),
        "auroc_d2h_inter_raw": auroc(inter_v, y),
        "auroc_d2h_combined_raw": auroc(combined, y),
        # Head-to-head supervised baselines:
        "auroc_l19_dom_subset": dom_auroc_subset,
        "prior_l19_dom_oof": PRIOR_L19_DOM_OOF,
        "prior_final_token_dom": PRIOR_FINAL_DOM,
        "prior_coe60": coe60,
        # Refutation verdicts:
        "refutation_2_labelfree_ge_supervised": bool(
            not np.isnan(dom_auroc_subset) and d2h_best >= dom_auroc_subset
        ),
        "refutation_4_alllayer_ge_singlelayer": bool(
            d2h_best >= PRIOR_L19_DOM_OOF
        ),
        "delta_d2h_minus_dom_subset": (
            float(d2h_best - dom_auroc_subset)
            if not np.isnan(dom_auroc_subset)
            else float("nan")
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())