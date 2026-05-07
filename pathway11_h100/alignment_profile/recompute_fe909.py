"""FE909 — Per-layer alignment score profile.

Computes cos(mean_diff_layer, DoM_L19) across all 29 layers to measure
where the correctness direction emerges in the residual stream.
Bootstrap 95% CIs via 2000 resamples.

Output: pathway11_h100/results/fe909_alignment_profile.json
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
DATA_DIR = ROOT / "pathway8_layerwise/data/math500"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe909_alignment_profile.json"

N_PROBLEMS = 500
HIDDEN_DIM = 1536
N_LAYERS = 29
N_BOOTSTRAP = 2000
SEED = 9999


def load_per_layer_prefill() -> tuple[np.ndarray, np.ndarray]:
    """Returns prefill (N_LAYERS, N_PROBLEMS, HIDDEN_DIM) and correct (N_PROBLEMS,).

    Copied from recompute_fe145.py — do NOT import (fragile sys.path + side effects).
    """
    X = np.zeros((N_LAYERS, N_PROBLEMS, HIDDEN_DIM), dtype=np.float32)
    y = np.zeros(N_PROBLEMS, dtype=bool)
    for i in range(N_PROBLEMS):
        fp = DATA_DIR / f"problem_{i:03d}.npz"
        if not fp.exists():
            raise SystemExit(f"missing per-problem NPZ: {fp}")
        with np.load(fp, allow_pickle=True) as d:
            s = d["states"]  # (29, T, 1536)
            X[:, i, :] = s[:, 0, :].astype(np.float32)
            y[i] = bool(d["correct"])
    return X, y


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main() -> int:
    print("Loading per-layer prefill activations (500 files, ~3 min)...")
    X, correct = load_per_layer_prefill()
    n = N_PROBLEMS

    # L19 DoM direction
    X19 = X[19]
    X19_c = X19 - X19.mean(axis=0)
    dom_l19 = X19_c[correct].mean(0) - X19_c[~correct].mean(0)
    dom_l19_norm = dom_l19 / (np.linalg.norm(dom_l19) + 1e-30)

    # Per-layer alignment
    alignment = np.zeros(N_LAYERS, dtype=np.float64)
    for layer in range(N_LAYERS):
        Xl = X[layer]
        Xl_c = Xl - Xl.mean(axis=0)
        diff_l = Xl_c[correct].mean(0) - Xl_c[~correct].mean(0)
        alignment[layer] = cosine_sim(diff_l, dom_l19_norm)

    # Bootstrap CIs
    rng = np.random.default_rng(SEED)
    boot_alignment = np.zeros((N_BOOTSTRAP, N_LAYERS), dtype=np.float64)
    for b in range(N_BOOTSTRAP):
        idx = rng.choice(n, n, replace=True)
        correct_b = correct[idx]
        if not correct_b.any() or correct_b.all():
            boot_alignment[b, :] = np.nan
            continue
        for layer in range(N_LAYERS):
            Xl = X[layer, idx]
            Xl_c = Xl - Xl.mean(axis=0)
            diff_l = Xl_c[correct_b].mean(0) - Xl_c[correct_b == False].mean(0)
            boot_alignment[b, layer] = cosine_sim(diff_l, dom_l19_norm)
        if (b + 1) % 500 == 0:
            print(f"  Bootstrap {b + 1}/{N_BOOTSTRAP}")

    ci_lo = np.nanpercentile(boot_alignment, 2.5, axis=0)
    ci_hi = np.nanpercentile(boot_alignment, 97.5, axis=0)

    # Peak analysis
    peak_layer = int(np.argmax(alignment))
    peak_value = float(alignment[peak_layer])

    # Half-maximum width
    half_max = peak_value / 2
    above_half = alignment >= half_max
    width_at_half = int(above_half.sum())
    first_above = int(np.argmax(above_half))
    last_above = int(N_LAYERS - 1 - np.argmax(above_half[::-1]))

    out = {
        "experiment": "FE909",
        "description": "Per-layer alignment score profile: cos(layer_diff, DoM_L19)",
        "n": n,
        "n_layers": N_LAYERS,
        "hidden_dim": HIDDEN_DIM,
        "alignment_per_layer": alignment.tolist(),
        "ci_lower_per_layer": ci_lo.tolist(),
        "ci_upper_per_layer": ci_hi.tolist(),
        "peak_layer": peak_layer,
        "peak_alignment": peak_value,
        "half_max_width": width_at_half,
        "half_max_range": [first_above, last_above],
        "is_l19_peak": peak_layer == 19,
        "alignment_l19": float(alignment[19]),
        "n_bootstrap": N_BOOTSTRAP,
        "interpretation": (
            "Peak width <= 4 layers => sharply localized (training imprint); "
            "width > 8 => broad multi-layer computation"
        ),
        "meta": {
            "data_dir": str(DATA_DIR),
            "seed": SEED,
            "method": "cosine_alignment_with_L19_DoM_per_layer",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nAlignment profile:")
    for layer in range(N_LAYERS):
        marker = " <-- peak" if layer == peak_layer else ""
        print(f"  L{layer:02d}: {alignment[layer]:.4f} [{ci_lo[layer]:.4f}, {ci_hi[layer]:.4f}]{marker}")
    print(f"\npeak_layer={peak_layer} alignment={peak_value:.4f}")
    print(f"half_max_width={width_at_half} range=[{first_above}, {last_above}]")
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
