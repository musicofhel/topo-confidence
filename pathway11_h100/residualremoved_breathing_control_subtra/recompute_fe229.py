"""P11-FE229 — Residual-removed breathing control for F-1.

Refutation #2 of the F-1 "breathing" curve. The paper's Figure 3 notes that
per-block contributions are tiny relative to the residual stream, so the
layer-wise participation-ratio (PR) curve that F-1 calls "breathing" might be
measuring slow residual-stream drift rather than genuine layer-wise
reorganization.

Control: instead of computing the PR curve on the full residual stream h[L],
recompute it on the per-block contribution delta[L] = h[L+1] - h[L] (the
residual-stream component removed). Compare three curves:
  - raw_pr[L]   : PR of the full residual stream at each layer (published F-1)
  - delta_pr[L] : PR of the block contributions (residual drift subtracted)
  - mp_null_pr  : PR of a white-Gaussian matrix of matched (n, d) — MP null
PR is scale-invariant, so the null depends only on (n, d).

Outcome A — breathing survives: delta_pr stays well below the MP null and
retains curve amplitude ⇒ F-1 is real layer-wise reorganization.
Outcome B — breathing collapses: delta_pr rises to the MP null and loses
amplitude ⇒ F-1 is a residual-stream slow-drift artifact.

Runs on the cached layer-wise P11 NPZs for Qwen-2.5-1.5B and 7B. CPU only.
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

# Layer-wise residual-stream caches: (n_samples, n_layers, dim) under key
# "hidden_states"/"hidden", or per-layer keys "layer_00".."layer_NN".
LAYER_CACHES = {
    "qwen2.5-1.5b": [
        ROOT / "pathway11_h100/breathing/m15b_layerwise.npz",
        ROOT / "pathway11_h100/prefill_inversion/cache/m15b_layerwise.npz",
        ROOT / "pathway11_h100/data/layerwise/m15b_layerwise.npz",
    ],
    "qwen2.5-7b": [
        ROOT / "pathway11_h100/breathing/m7b_layerwise.npz",
        ROOT / "pathway11_h100/prefill_inversion/cache/m7b_layerwise.npz",
        ROOT / "pathway11_h100/data/layerwise/m7b_layerwise.npz",
    ],
}

OUT_JSON = ROOT / "pathway11_h100/residual_removed_breathing/results.json"

SEED = 9999
N_NULL = 5  # white-Gaussian null replicates


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def participation_ratio(X: np.ndarray) -> float:
    """PR = (Σλ)² / Σλ² of the sample-covariance eigenvalues. Scale-invariant.

    Low PR ⇒ variance concentrated in few directions (structured); high PR ⇒
    spread across many directions (random/isotropic)."""
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    C = Xc.T @ Xc / max(n - 1, 1)
    lam = np.linalg.eigvalsh(C)
    lam = np.clip(lam, 0.0, None)
    s1 = float(lam.sum())
    s2 = float((lam * lam).sum())
    if s2 <= 0.0:
        return float("nan")
    return s1 * s1 / s2


def mp_null_pr(n: int, d: int, seed: int, reps: int = N_NULL) -> float:
    """Marchenko-Pastur null PR: white-Gaussian matrix of matched shape."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        G = rng.standard_normal((n, d))
        vals.append(participation_ratio(G))
    return float(np.mean(vals))


def load_layerwise(paths: list[Path]) -> np.ndarray | None:
    for p in paths:
        if not p.exists():
            continue
        blob = np.load(p)
        files = list(blob.files)
        if "hidden_states" in files:
            H = blob["hidden_states"]
        elif "hidden" in files:
            H = blob["hidden"]
        else:
            keys = sorted(k for k in files if k.startswith("layer"))
            if not keys:
                continue
            H = np.stack([blob[k] for k in keys], axis=1)
        H = np.asarray(H, dtype=np.float64)
        if H.ndim != 3:
            continue
        return H  # (n_samples, n_layers, dim)
    return None


def analyze_model(H: np.ndarray, seed: int) -> dict:
    """H: (n_samples, n_layers, dim). Build raw / delta / null PR curves."""
    n, n_layers, d = H.shape

    raw_pr = [participation_ratio(H[:, L, :]) for L in range(n_layers)]
    # Block contributions delta[L] = h[L+1] - h[L]; one fewer point than layers.
    delta = H[:, 1:, :] - H[:, :-1, :]
    delta_pr = [participation_ratio(delta[:, L, :]) for L in range(delta.shape[1])]

    null_pr = mp_null_pr(n, d, seed)

    raw_arr = np.asarray(raw_pr, dtype=np.float64)
    delta_arr = np.asarray(delta_pr, dtype=np.float64)

    def amplitude(a: np.ndarray) -> float:
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return float("nan")
        return float(finite.max() - finite.min())

    raw_amp = amplitude(raw_arr)
    delta_amp = amplitude(delta_arr)

    # Fraction of the raw structural gap (null - raw) that the residual-removed
    # delta curve retains. ~1 ⇒ structure survives; ~0 ⇒ collapses to MP null.
    raw_gap = null_pr - float(np.nanmean(raw_arr))
    delta_gap = null_pr - float(np.nanmean(delta_arr))
    survival = float(delta_gap / raw_gap) if abs(raw_gap) > 1e-9 else float("nan")

    # Amplitude retention of the breathing curve after residual removal.
    amp_retention = float(delta_amp / raw_amp) if raw_amp > 1e-9 else float("nan")

    return {
        "n_samples": int(n),
        "n_layers": int(n_layers),
        "dim": int(d),
        "raw_pr_curve": [float(x) for x in raw_arr],
        "delta_pr_curve": [float(x) for x in delta_arr],
        "mp_null_pr": null_pr,
        "raw_pr_mean": float(np.nanmean(raw_arr)),
        "delta_pr_mean": float(np.nanmean(delta_arr)),
        "raw_breathing_amplitude": raw_amp,
        "delta_breathing_amplitude": delta_amp,
        "structure_survival_fraction": survival,
        "amplitude_retention_fraction": amp_retention,
    }


def classify(model_results: dict) -> str:
    """Outcome A (real) vs B (artifact), pooled across available models.

    Survival ⇒ delta curve stays structured (low PR vs MP null) AND retains
    breathing amplitude. Collapse ⇒ delta rises to the null and flattens."""
    survivals, retentions = [], []
    for r in model_results.values():
        if np.isfinite(r["structure_survival_fraction"]):
            survivals.append(r["structure_survival_fraction"])
        if np.isfinite(r["amplitude_retention_fraction"]):
            retentions.append(r["amplitude_retention_fraction"])
    if not survivals:
        return "INCONCLUSIVE"
    surv = float(np.mean(survivals))
    ret = float(np.mean(retentions)) if retentions else 0.0
    if surv >= 0.5 and ret >= 0.5:
        return "A_BREATHING_SURVIVES"  # F-1 is real layer-wise reorganization
    if surv <= 0.2 or ret <= 0.2:
        return "B_COLLAPSES_TO_MP_NULL"  # F-1 is residual-drift artifact
    return "PARTIAL"


def main() -> int:
    model_results: dict[str, dict] = {}
    missing: list[str] = []

    for model, paths in LAYER_CACHES.items():
        H = load_layerwise(paths)
        if H is None:
            missing.append(model)
            continue
        model_results[model] = analyze_model(H, SEED)

    if not model_results:
        # No layer-wise cache available for any model — cannot build a curve.
        print("MISSING_REGEN_INPUT", file=sys.stderr)
        for model in missing:
            for p in LAYER_CACHES[model]:
                print("  tried:", p, file=sys.stderr)
        return 2

    out = {
        "experiment": "P11-FE229",
        "description": "Residual-removed breathing control for F-1",
        "seed": SEED,
        "n_null_replicates": N_NULL,
        "models": model_results,
        "models_missing": missing,
        "outcome": classify(model_results),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print("wrote", OUT_JSON)
    print("outcome:", out["outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())