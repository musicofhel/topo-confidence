#!/usr/bin/env python3
"""A1: Build mean-pooled (and prefill/last-token) caches from raw per-problem NPZs.

Outputs:
    cache/math500_1p5b.npz  — X_mean, X_prefill, X_last, y, mean_logprob, n_gen_tokens, d2h_attn_entropy, subjects
    cache/math500_7b.npz    — same schema (d2h_attn_entropy is all-zeros on 7B)
    cache/bbh_1p5b.npz      — same schema + subset array
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from lib import (
    CACHE_DIR, DATA_1P5B, DATA_7B, DATA_BBH, BBH_SUBSETS,
    N_LAYERS, HIDDEN_1P5B, HIDDEN_7B,
    ensure_dirs, load_manifest, parse_subject, timer,
)


def build_math500_cache(data_dir: Path, hidden_dim: int, name: str):
    manifest = load_manifest(data_dir)
    n = manifest["n_problems"]
    assert n == 500

    X_mean = np.zeros((n, N_LAYERS, hidden_dim), dtype=np.float32)
    X_prefill = np.zeros((n, N_LAYERS, hidden_dim), dtype=np.float32)
    X_last = np.zeros((n, N_LAYERS, hidden_dim), dtype=np.float32)
    y = np.zeros(n, dtype=bool)
    mean_logprob = np.zeros(n, dtype=np.float64)
    n_gen_tokens = np.zeros(n, dtype=np.int32)
    d2h_attn_entropy = np.zeros((n, 28), dtype=np.float64)
    subjects = []

    t0 = time.time()
    for i, prob in enumerate(manifest["problems"]):
        fp = data_dir / f"problem_{i:03d}.npz"
        with np.load(fp, allow_pickle=True) as d:
            s = d["states"]  # (29, T, H) float16
            X_mean[i] = s.mean(axis=1).astype(np.float32)
            X_prefill[i] = s[:, 0, :].astype(np.float32)
            X_last[i] = s[:, -1, :].astype(np.float32)
            y[i] = bool(d["correct"])
            mean_logprob[i] = float(d["mean_logprob"])
            d2h_attn_entropy[i] = d["d2h_attn_entropy"]
        n_gen_tokens[i] = prob["n_gen_tokens"]
        subjects.append(parse_subject(prob["unique_id"]))

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"  {name}: {i+1}/{n} ({rate:.1f} prob/s, ETA {eta:.0f}s)")

    out_path = CACHE_DIR / f"{name}.npz"
    np.savez(
        out_path,
        X_mean=X_mean,
        X_prefill=X_prefill,
        X_last=X_last,
        y=y,
        mean_logprob=mean_logprob,
        n_gen_tokens=n_gen_tokens,
        d2h_attn_entropy=d2h_attn_entropy,
        subjects=np.array(subjects),
        accuracy=manifest["accuracy"],
        n_correct=manifest["n_correct"],
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"  → {out_path} ({size_mb:.0f} MB)")
    print(f"  accuracy={y.mean():.4f} (manifest: {manifest['accuracy']})")


def build_bbh_cache(data_dir: Path, hidden_dim: int, name: str):
    manifest = load_manifest(data_dir)
    problems_by_subset = {}
    for p in manifest["problems"]:
        problems_by_subset.setdefault(p["subset"], []).append(p)

    all_X_mean = []
    all_X_prefill = []
    all_X_last = []
    all_y = []
    all_logprob = []
    all_tokens = []
    all_entropy = []
    all_subsets = []

    for subset in BBH_SUBSETS:
        subset_dir = data_dir / subset
        n = len(list(subset_dir.glob("problem_*.npz")))

        t0 = time.time()
        for i in range(n):
            fp = subset_dir / f"problem_{i:03d}.npz"
            if not fp.exists():
                print(f"  WARNING: {fp} missing, skipping")
                continue
            with np.load(fp, allow_pickle=True) as d:
                s = d["states"]
                all_X_mean.append(s.mean(axis=1).astype(np.float32))
                all_X_prefill.append(s[:, 0, :].astype(np.float32))
                all_X_last.append(s[:, -1, :].astype(np.float32))
                all_y.append(bool(d["correct"]))
                all_logprob.append(float(d["mean_logprob"]))
                all_entropy.append(d["d2h_attn_entropy"])
            all_tokens.append(0)
            all_subsets.append(subset)

        elapsed = time.time() - t0
        count = sum(1 for s in all_subsets if s == subset)
        acc = np.mean([all_y[j] for j, s in enumerate(all_subsets) if s == subset])
        print(f"  {subset}: {count} problems, acc={acc:.3f} ({elapsed:.1f}s)")

    out_path = CACHE_DIR / f"{name}.npz"
    np.savez(
        out_path,
        X_mean=np.stack(all_X_mean),
        X_prefill=np.stack(all_X_prefill),
        X_last=np.stack(all_X_last),
        y=np.array(all_y),
        mean_logprob=np.array(all_logprob),
        n_gen_tokens=np.array(all_tokens, dtype=np.int32),
        d2h_attn_entropy=np.stack(all_entropy),
        subsets=np.array(all_subsets),
    )
    size_mb = out_path.stat().st_size / 1e6
    total_acc = np.mean(all_y)
    print(f"  → {out_path} ({size_mb:.0f} MB), n={len(all_y)}, acc={total_acc:.3f}")


def main():
    ensure_dirs()

    with timer("A1: 1.5B MATH-500 cache"):
        build_math500_cache(DATA_1P5B, HIDDEN_1P5B, "math500_1p5b")

    with timer("A1: 7B MATH-500 cache"):
        build_math500_cache(DATA_7B, HIDDEN_7B, "math500_7b")

    with timer("A1: BBH 1.5B cache"):
        build_bbh_cache(DATA_BBH, HIDDEN_1P5B, "bbh_1p5b")

    print("\n[A1] All caches built.")


if __name__ == "__main__":
    main()
