"""FE119 — Layer-sweep cos(prefill_DoM, final_DoM) across all 29 layers.

Tests whether F-3's orthogonality (cos=0.046 at L19) is a transient in a
continuous rotation or a stable geometric fact. Bootstrap 1000x CIs.

Output: pathway11_h100/results/fe119_layer_sweep_cos.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
DATA_DIR = ROOT / "pathway8_layerwise/data/math500"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe119_layer_sweep_cos.json"

N_PROBLEMS = 500
HIDDEN_DIM = 1536
N_LAYERS = 29
N_BOOTSTRAP = 1000
SEED = 9999


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main() -> int:
    print("Loading all 29 layers x 2 positions (prefill + final) from 500 NPZs...")
    X_prefill = np.zeros((N_LAYERS, N_PROBLEMS, HIDDEN_DIM), dtype=np.float32)
    X_final = np.zeros((N_LAYERS, N_PROBLEMS, HIDDEN_DIM), dtype=np.float32)
    correct = np.zeros(N_PROBLEMS, dtype=bool)

    for i in range(N_PROBLEMS):
        fp = DATA_DIR / f"problem_{i:03d}.npz"
        if not fp.exists():
            raise SystemExit(f"missing: {fp}")
        with np.load(fp, allow_pickle=True) as d:
            s = d["states"]  # (29, T, 1536)
            X_prefill[:, i, :] = s[:, 0, :].astype(np.float32)
            X_final[:, i, :] = s[:, -1, :].astype(np.float32)
            correct[i] = bool(d["correct"])
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{N_PROBLEMS}")

    print(f"Correct: {correct.sum()}/{N_PROBLEMS}")
    print(f"Memory: X_prefill + X_final = {(X_prefill.nbytes + X_final.nbytes) / 1e6:.0f} MB")

    # Per-layer cos(prefill_DoM, final_DoM)
    cos_profile = np.zeros(N_LAYERS, dtype=np.float64)
    for L in range(N_LAYERS):
        Xp = X_prefill[L] - X_prefill[L].mean(axis=0)
        Xf = X_final[L] - X_final[L].mean(axis=0)
        dom_p = Xp[correct].mean(0) - Xp[~correct].mean(0)
        dom_f = Xf[correct].mean(0) - Xf[~correct].mean(0)
        cos_profile[L] = cosine_sim(dom_p, dom_f)

    print("\ncos(prefill_DoM, final_DoM) per layer:")
    for L in range(N_LAYERS):
        marker = " <-- L19" if L == 19 else ""
        print(f"  L{L:02d}: {cos_profile[L]:.4f}{marker}")

    # Bootstrap CIs
    print(f"\nBootstrapping {N_BOOTSTRAP} resamples...")
    rng = np.random.default_rng(SEED)
    boot_cos = np.zeros((N_BOOTSTRAP, N_LAYERS), dtype=np.float64)

    for b in range(N_BOOTSTRAP):
        idx = rng.choice(N_PROBLEMS, N_PROBLEMS, replace=True)
        correct_b = correct[idx]
        if not correct_b.any() or correct_b.all():
            boot_cos[b, :] = np.nan
            continue
        for L in range(N_LAYERS):
            Xp = X_prefill[L, idx] - X_prefill[L, idx].mean(axis=0)
            Xf = X_final[L, idx] - X_final[L, idx].mean(axis=0)
            dom_p = Xp[correct_b].mean(0) - Xp[~correct_b].mean(0)
            dom_f = Xf[correct_b].mean(0) - Xf[~correct_b].mean(0)
            boot_cos[b, L] = cosine_sim(dom_p, dom_f)
        if (b + 1) % 250 == 0:
            print(f"  Bootstrap {b + 1}/{N_BOOTSTRAP}")

    ci_lo = np.nanpercentile(boot_cos, 2.5, axis=0)
    ci_hi = np.nanpercentile(boot_cos, 97.5, axis=0)

    # Analysis
    min_cos_layer = int(np.argmin(np.abs(cos_profile)))
    max_cos_layer = int(np.argmax(cos_profile))
    passes_through_high = bool(np.any(np.abs(cos_profile) > 0.8))

    out = {
        "experiment": "FE119",
        "description": "Layer-sweep cos(prefill_DoM, final_DoM) across all 29 layers",
        "n": N_PROBLEMS,
        "n_correct": int(correct.sum()),
        "n_layers": N_LAYERS,
        "cos_per_layer": cos_profile.tolist(),
        "ci_lower_per_layer": ci_lo.tolist(),
        "ci_upper_per_layer": ci_hi.tolist(),
        "cos_at_L19": float(cos_profile[19]),
        "cos_at_L19_reference": 0.046,
        "min_abs_cos_layer": min_cos_layer,
        "min_abs_cos_value": float(cos_profile[min_cos_layer]),
        "max_cos_layer": max_cos_layer,
        "max_cos_value": float(cos_profile[max_cos_layer]),
        "passes_through_high_alignment": bool(passes_through_high),
        "cross_check": {
            "L19_cos_near_0_046": bool(abs(cos_profile[19] - 0.046) < 0.05),
        },
        "interpretation": (
            "If cos passes through ±1 at some layer, F-3 orthogonality at L19 "
            "is a transient in a continuous rotation. If cos is low across all "
            "layers, prefill and final DoM directions are genuinely independent "
            "computational channels."
        ),
        "n_bootstrap": N_BOOTSTRAP,
        "fold_structure": "N/A (no OOF, full-sample DoM per layer)",
        "meta": {
            "data_dir": str(DATA_DIR),
            "seed": SEED,
            "method": "per_layer_cosine_prefill_final_dom_with_bootstrap",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
