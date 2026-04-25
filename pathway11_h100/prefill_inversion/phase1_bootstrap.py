#!/usr/bin/env python3
"""Experiment 3 / Phase 1: Bootstrap validation of 7B prefill PR inversion.

(a) Bootstrap 1000× for each of 7B and 1.5B: resample n problems w/ replacement,
    split by correctness, compute PR per group, record PR(correct)/PR(incorrect).
    95% CI on the ratio. Also record PR_correct and PR_incorrect CIs separately.

(b) Label-balance control: random subsample of the larger group to the size of the
    smaller group. For 7B: 366 correct / 134 incorrect → subsample correct to 134.
    For 1.5B: 243 correct / 257 incorrect → subsample incorrect to 243.
    Repeat 100× and check if the inversion survives balancing.

Caches prefill arrays to disk since loading is slow (~5 min).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (OUT_DIR, load_7b, load_15b, participation_ratio)

CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 1000
N_BALANCE = 100
SEED = 9999


def cached_load(name: str, loader):
    """Cache the (prefill, final_tok, correct, seq_len) per model."""
    fp = CACHE_DIR / f"{name}.npz"
    if fp.exists():
        d = np.load(fp)
        return d["prefill"], d["final_tok"], d["correct"].astype(bool), d["seq_len"], []
    pf, ft, c, sl, miss = loader()
    np.savez_compressed(fp, prefill=pf.astype(np.float16), final_tok=ft.astype(np.float16),
                        correct=c, seq_len=sl)
    return pf, ft, c, sl, miss


def bootstrap_pr(H: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> dict:
    """Bootstrap the (correct PR, incorrect PR, ratio) distributions."""
    rng = np.random.default_rng(seed)
    n = len(y)
    pr_c = np.zeros(n_boot)
    pr_i = np.zeros(n_boot)
    ratios = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if y_b.sum() < 5 or (~y_b).sum() < 5:
            pr_c[b] = np.nan; pr_i[b] = np.nan; ratios[b] = np.nan
            continue
        H_b = H[idx]
        pc = participation_ratio(H_b[y_b])
        pi = participation_ratio(H_b[~y_b])
        pr_c[b] = pc; pr_i[b] = pi
        ratios[b] = pc / max(pi, 1e-30)
    return dict(
        pr_correct=dict(
            mean=float(np.nanmean(pr_c)),
            median=float(np.nanmedian(pr_c)),
            ci95=[float(np.nanpercentile(pr_c, 2.5)),
                  float(np.nanpercentile(pr_c, 97.5))],
        ),
        pr_incorrect=dict(
            mean=float(np.nanmean(pr_i)),
            median=float(np.nanmedian(pr_i)),
            ci95=[float(np.nanpercentile(pr_i, 2.5)),
                  float(np.nanpercentile(pr_i, 97.5))],
        ),
        ratio=dict(
            mean=float(np.nanmean(ratios)),
            median=float(np.nanmedian(ratios)),
            ci95=[float(np.nanpercentile(ratios, 2.5)),
                  float(np.nanpercentile(ratios, 97.5))],
            pct_ratio_above_1=float(np.mean(ratios > 1)),
            n_boot_valid=int(np.sum(~np.isnan(ratios))),
        ),
    )


def balance_control(H: np.ndarray, y: np.ndarray, n_draws: int, seed: int) -> dict:
    """Subsample the majority class down to match minority class. Compute PR per draw."""
    rng = np.random.default_rng(seed)
    n_pos = int(y.sum()); n_neg = int((~y).sum())
    keep = min(n_pos, n_neg)
    majority_label = True if n_pos > n_neg else False
    minority_mask = ~y if majority_label else y
    majority_idxs = np.where(y if majority_label else ~y)[0]
    minority_idxs = np.where(minority_mask)[0]

    pr_min = np.zeros(n_draws)
    pr_maj = np.zeros(n_draws)
    ratios = np.zeros(n_draws)
    for b in range(n_draws):
        samp = rng.choice(majority_idxs, size=keep, replace=False)
        # minority is always kept in full (already size=keep)
        pmin = participation_ratio(H[minority_idxs])
        pmaj = participation_ratio(H[samp])
        pr_min[b] = pmin
        pr_maj[b] = pmaj
        # Ratio in the "correct/incorrect" convention so it's directly comparable:
        if majority_label:  # majority is correct
            ratios[b] = pmaj / max(pmin, 1e-30)
        else:
            ratios[b] = pmin / max(pmaj, 1e-30)
    return dict(
        n_min=int(keep), n_maj_subsampled=int(keep),
        minority_is_correct=not bool(majority_label),
        pr_minority_mean=float(pr_min.mean()),
        pr_majority_subsampled_mean=float(pr_maj.mean()),
        ratio_correct_over_incorrect_mean=float(ratios.mean()),
        ratio_correct_over_incorrect_ci95=[float(np.percentile(ratios, 2.5)),
                                           float(np.percentile(ratios, 97.5))],
        pct_ratio_above_1=float(np.mean(ratios > 1)),
    )


def analyze_model(name: str, H: np.ndarray, y: np.ndarray,
                  n_boot: int = N_BOOT, n_balance: int = N_BALANCE):
    print(f"\n=== {name} ===")
    print(f"  n={len(y)}, correct={y.sum()} ({y.mean():.3f}), incorrect={(~y).sum()}")
    t0 = time.time()
    # Point estimates
    pr_c = participation_ratio(H[y])
    pr_i = participation_ratio(H[~y])
    ratio = pr_c / pr_i
    print(f"  point: PR_correct={pr_c:.3f}, PR_incorrect={pr_i:.3f}, ratio={ratio:.3f}")
    print(f"  bootstrap {n_boot}x ...")
    boot = bootstrap_pr(H, y, n_boot, SEED)
    print(f"    PR_correct: mean={boot['pr_correct']['mean']:.3f} CI95={boot['pr_correct']['ci95']}")
    print(f"    PR_incorrect: mean={boot['pr_incorrect']['mean']:.3f} CI95={boot['pr_incorrect']['ci95']}")
    print(f"    ratio: mean={boot['ratio']['mean']:.3f} CI95={boot['ratio']['ci95']} "
          f"P(ratio>1)={boot['ratio']['pct_ratio_above_1']:.3f}")
    print(f"  balance control {n_balance}x ...")
    bal = balance_control(H, y, n_balance, SEED)
    print(f"    balanced: PR_min={bal['pr_minority_mean']:.3f} "
          f"PR_maj_subsamp={bal['pr_majority_subsampled_mean']:.3f}")
    print(f"    ratio correct/incorrect: {bal['ratio_correct_over_incorrect_mean']:.3f} "
          f"CI95={bal['ratio_correct_over_incorrect_ci95']} "
          f"P(>1)={bal['pct_ratio_above_1']:.3f}")
    print(f"  {time.time()-t0:.1f}s")
    return dict(
        n=int(len(y)), n_correct=int(y.sum()), n_incorrect=int((~y).sum()),
        point=dict(pr_correct=float(pr_c), pr_incorrect=float(pr_i), ratio=float(ratio)),
        bootstrap=boot, balance_control=bal,
    )


def main():
    print("=" * 70)
    print("Exp 3 / Phase 1: Bootstrap validation of 7B prefill PR inversion")
    print("=" * 70)

    print("\nLoading 7B prefill...")
    pf7, ft7, c7, sl7, _ = cached_load("m7b_prefill", load_7b)
    print(f"  7B shape: {pf7.shape}")
    print("Loading 1.5B prefill...")
    pf15, ft15, c15, sl15, _ = cached_load("m15b_prefill", load_15b)
    print(f"  1.5B shape: {pf15.shape}")

    results = {}
    for name, H, y in [("7B_prefill", pf7, c7), ("1.5B_prefill", pf15, c15),
                       ("7B_final_token", ft7, c7), ("1.5B_final_token", ft15, c15)]:
        results[name] = analyze_model(name, H, y)

    with open(OUT_DIR / "phase1_bootstrap.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_DIR/'phase1_bootstrap.json'}")


if __name__ == "__main__":
    main()
