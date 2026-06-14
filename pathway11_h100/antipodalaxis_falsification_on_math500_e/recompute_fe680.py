"""P11-FE680 — Antipodal-axis falsification on MATH-500.

F-4 (asymmetric collapse) predicts that the antipodal symmetry assumed by
Spherical Steering's vMF gate (mu_H = -mu_T, which gates Eq. 17) breaks on
reasoning data. This script tests that prediction on the cached Qwen-2.5-1.5B
L19 residual-stream activations.

For each position we compute, on the unit hypersphere (vMF regime):
  mu_T = normalize( mean_i normalize(x_i)  over correct  examples )
  mu_H = normalize( mean_i normalize(x_i)  over incorrect examples )
and report cos(mu_T, -mu_H) == -cos(mu_T, mu_H). A perfectly antipodal pair
gives +1; drift toward 0 (or below) refutes the paper's gating assumption for
the math-reasoning regime.

Available cache: pathway11_h100/prefill_inversion/cache/m15b_prefill.npz holds
the L19 prefill (last prompt token) hidden states (500, 1536). Last-token and
intermediate generation-position caches are not present on local hardware, so
those positions are reported as skipped (MISSING) rather than fabricated. The
prefill position alone is a clean test of the antipodal premise. A
correctness-balanced bootstrap supplies a CI and a label-permutation null
gives the chance level of the antipodal cosine.
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
# Optional position caches — present only if a future re-extract dumps them.
LASTTOK_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_lasttoken.npz"
MIDGEN_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_midgen.npz"
OUT_JSON = ROOT / "pathway11_h100/antipodal_falsification/results.json"

SEED = 9999
N_BOOT = 2000
N_PERM = 2000
EPS = 1e-12


def unit_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < EPS, 1.0, norms)
    return X / norms


def mean_direction(X: np.ndarray) -> np.ndarray:
    """Mean of the unit-normalized rows, renormalized to the sphere."""
    m = unit_rows(X).mean(axis=0)
    n = float(np.linalg.norm(m))
    if n < EPS:
        return m
    return m / n


def antipodal_cos(X: np.ndarray, y: np.ndarray) -> dict:
    """cos(mu_T, mu_H) and the antipodal score cos(mu_T, -mu_H)."""
    mu_T = mean_direction(X[y])
    mu_H = mean_direction(X[~y])
    cos_TH = float(mu_T @ mu_H)
    return {
        "cos_muT_muH": cos_TH,
        "cos_muT_neg_muH": -cos_TH,
        "resultant_len_T": float(np.linalg.norm(unit_rows(X[y]).mean(axis=0))),
        "resultant_len_H": float(np.linalg.norm(unit_rows(X[~y]).mean(axis=0))),
        "n_T": int(y.sum()),
        "n_H": int((~y).sum()),
    }


def analyze_position(name: str, X: np.ndarray, y: np.ndarray) -> dict:
    assert X.ndim == 2 and X.shape[0] == y.shape[0]
    base = antipodal_cos(X, y)

    rng = np.random.default_rng(SEED)
    pos_idx = np.flatnonzero(y)
    neg_idx = np.flatnonzero(~y)

    # Correctness-balanced bootstrap CI on the antipodal cosine.
    boot = np.empty(N_BOOT, dtype=np.float64)
    for b in range(N_BOOT):
        bp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        mt = mean_direction(X[bp])
        mh = mean_direction(X[bn])
        boot[b] = -float(mt @ mh)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    # Label-permutation null: shuffle correctness, recompute antipodal cosine.
    perm = np.empty(N_PERM, dtype=np.float64)
    for p in range(N_PERM):
        yp = rng.permutation(y)
        mt = mean_direction(X[yp])
        mh = mean_direction(X[~yp])
        perm[p] = -float(mt @ mh)

    obs = base["cos_muT_neg_muH"]
    # Two-sided: how often does the null reach the observed deviation from +1?
    null_dev = np.abs(perm - 1.0)
    obs_dev = abs(obs - 1.0)
    p_value = float((null_dev >= obs_dev).mean())

    return {
        **base,
        "antipodal_ci95": [float(ci_lo), float(ci_hi)],
        "antipodal_null_mean": float(perm.mean()),
        "antipodal_null_std": float(perm.std()),
        "antipodal_perm_pvalue_vs_unity": p_value,
        # Refutes the antipodal assumption when +1 is excluded from the CI.
        "unity_in_ci95": bool(ci_lo <= 1.0 <= ci_hi),
    }


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X_prefill = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X_prefill.shape == (500, 1536) and y.shape == (500,)

    positions: dict[str, dict] = {}
    skipped: list[str] = []

    positions["prefill_last_prompt_tok"] = analyze_position(
        "prefill_last_prompt_tok", X_prefill, y
    )

    for name, path in (
        ("last_generation_tok", LASTTOK_CACHE),
        ("mid_generation_tok", MIDGEN_CACHE),
    ):
        if not path.exists():
            skipped.append(name)
            continue
        b = np.load(path)
        key = "prefill" if "prefill" in b.files else b.files[0]
        Xp = b[key].astype(np.float64)
        yp = b["correct"].astype(bool) if "correct" in b.files else y
        positions[name] = analyze_position(name, Xp, yp)

    pf = positions["prefill_last_prompt_tok"]
    out = {
        "experiment": "P11-FE680",
        "description": "Antipodal-axis falsification (mu_H == -mu_T) on MATH-500 L19 activations",
        "model": "Qwen-2.5-1.5B",
        "n_boot": N_BOOT,
        "n_perm": N_PERM,
        "seed": SEED,
        "positions": positions,
        "skipped_positions": skipped,
        "skipped_reason": "no cached NPZ for this position on local hardware",
        # Headline: does the math-regime prefill axis satisfy antipodal symmetry?
        "prefill_antipodal_cos": pf["cos_muT_neg_muH"],
        "prefill_antipodal_ci95": pf["antipodal_ci95"],
        "antipodal_holds_at_prefill": pf["unity_in_ci95"],
        "verdict": (
            "ANTIPODAL_HOLDS" if pf["unity_in_ci95"] else "ANTIPODAL_REFUTED"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(
        f"prefill cos(mu_T,-mu_H)={pf['cos_muT_neg_muH']:.4f} "
        f"CI95={pf['antipodal_ci95']} verdict={out['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())