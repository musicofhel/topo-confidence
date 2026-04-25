#!/usr/bin/env python3
"""Experiment 3 / Phase 3: mechanistic analysis.

(a) MATH-500 difficulty levels. MATH-500 ships with "level" field (1–5). Load from
    HuggingFace HuggingFaceH4/MATH-500 (or cache locally). For each model:
      * prefill PR per (level, correctness) cell
      * n per cell
    Report whether the 7B inversion holds across levels or is concentrated at specific levels.

(b) K-means clustering (k ∈ {5, 6, 8, 10}) on 7B prefill activations. Check cluster
    alignment with correctness and difficulty level (adjusted Rand index).

Outputs phase3_mechanistic.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, participation_ratio

CACHE_DIR = OUT_DIR / "cache"


def load_cached(name):
    fp = CACHE_DIR / f"{name}.npz"
    d = np.load(fp)
    return d["prefill"].astype(np.float32), d["final_tok"].astype(np.float32), \
           d["correct"].astype(bool), d["seq_len"]


def load_math500_levels():
    """Return (n_problems,) int array of difficulty levels, or all-zeros if unavailable."""
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        levels = []
        for item in ds:
            lv = item.get("level", "Level 0")
            # Levels in MATH-500 are strings like "Level 1" through "Level 5"
            if isinstance(lv, str):
                parts = lv.replace("Level", "").strip()
                try:
                    levels.append(int(parts))
                except ValueError:
                    levels.append(0)
            else:
                levels.append(int(lv) if lv else 0)
        return np.array(levels, dtype=int)
    except Exception as e:
        print(f"  WARNING: could not load MATH-500 levels: {e}")
        return np.zeros(500, dtype=int)


def pr_per_difficulty(H, c, levels):
    results = {}
    for lvl in sorted(set(int(l) for l in levels)):
        if lvl == 0:
            continue
        mask_lvl = levels == lvl
        mask_c = mask_lvl & c
        mask_i = mask_lvl & ~c
        n_c = int(mask_c.sum()); n_i = int(mask_i.sum())
        entry = dict(n_level=int(mask_lvl.sum()), n_correct=n_c, n_incorrect=n_i)
        if n_c >= 5:
            entry["pr_correct"] = float(participation_ratio(H[mask_c]))
        if n_i >= 5:
            entry["pr_incorrect"] = float(participation_ratio(H[mask_i]))
        if "pr_correct" in entry and "pr_incorrect" in entry:
            entry["ratio"] = entry["pr_correct"] / entry["pr_incorrect"]
        results[f"level_{lvl}"] = entry
    return results


def kmeans_alignment(H, c, levels, k_values=(5, 6, 8, 10)):
    # PCA to 64-d for tractable k-means on 3584-d activations
    pca = PCA(n_components=min(64, H.shape[1] - 1), random_state=9999)
    Hp = pca.fit_transform(H)
    var_explained = float(pca.explained_variance_ratio_.sum())
    results = {"pca_components": Hp.shape[1], "pca_var_explained": var_explained}
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=9999, n_init=10)
        labels = km.fit_predict(Hp)
        # Distribution of correctness per cluster
        per_cluster = []
        for ci in range(k):
            m = labels == ci
            per_cluster.append(dict(
                n=int(m.sum()),
                acc=float(c[m].mean()) if m.any() else 0.0,
                level_mean=float(levels[m].mean()) if m.any() and levels.any() else 0.0,
            ))
        ari_correct = float(adjusted_rand_score(c.astype(int), labels))
        ari_level = float(adjusted_rand_score(levels, labels)) if levels.any() else None
        # Silhouette-like: variance of per-cluster accuracy (higher = more correctness-sorted)
        per_acc = np.array([pc["acc"] for pc in per_cluster])
        acc_variance = float(per_acc.var())
        results[f"k{k}"] = dict(
            ari_correct=ari_correct, ari_level=ari_level,
            acc_variance_across_clusters=acc_variance,
            per_cluster=per_cluster,
        )
    return results


def main():
    print("=" * 70)
    print("Exp 3 / Phase 3: mechanistic analysis")
    print("=" * 70)
    pf7, ft7, c7, sl7 = load_cached("m7b_prefill")
    pf15, ft15, c15, sl15 = load_cached("m15b_prefill")

    print("\nLoading MATH-500 difficulty levels...")
    levels = load_math500_levels()
    have_levels = levels.any()
    print(f"  have levels: {have_levels}; dist = {dict(zip(*np.unique(levels, return_counts=True)))}")

    out = {"have_levels": bool(have_levels)}
    if have_levels:
        print("\n(a) PR per (level, correctness) — 7B ...")
        pr_lvl_7b_prefill = pr_per_difficulty(pf7, c7, levels)
        for lvl, e in pr_lvl_7b_prefill.items():
            if "ratio" in e:
                print(f"  {lvl}: n={e['n_level']} (c={e['n_correct']}, i={e['n_incorrect']}) "
                      f"PR_c={e['pr_correct']:.3f} PR_i={e['pr_incorrect']:.3f} ratio={e['ratio']:.3f}")
            else:
                print(f"  {lvl}: n={e['n_level']} (too few in one group)")
        pr_lvl_7b_ft = pr_per_difficulty(ft7, c7, levels)

        print("\n(a') PR per (level, correctness) — 1.5B ...")
        pr_lvl_15b_prefill = pr_per_difficulty(pf15, c15, levels)
        for lvl, e in pr_lvl_15b_prefill.items():
            if "ratio" in e:
                print(f"  {lvl}: n={e['n_level']} (c={e['n_correct']}, i={e['n_incorrect']}) "
                      f"PR_c={e['pr_correct']:.3f} PR_i={e['pr_incorrect']:.3f} ratio={e['ratio']:.3f}")

        out["pr_per_level"] = dict(
            model_7b_prefill=pr_lvl_7b_prefill,
            model_7b_finaltok=pr_lvl_7b_ft,
            model_15b_prefill=pr_lvl_15b_prefill,
        )
        out["level_distribution"] = {int(lvl): int(cnt)
                                     for lvl, cnt in zip(*np.unique(levels, return_counts=True))}

    print("\n(b) K-means clustering on 7B prefill ...")
    km7 = kmeans_alignment(pf7, c7, levels)
    print(f"  PCA: {km7['pca_components']} components, {km7['pca_var_explained']:.3f} var")
    for k in [5, 6, 8, 10]:
        e = km7[f"k{k}"]
        print(f"  k={k}: ARI(correct)={e['ari_correct']:.4f} "
              f"ARI(level)={e['ari_level']} "
              f"acc_var_across_clusters={e['acc_variance_across_clusters']:.4f}")
        # Print top 3 most correctness-sorted clusters
        pc = sorted(e["per_cluster"], key=lambda x: x["acc"], reverse=True)[:3]
        pc_bot = sorted(e["per_cluster"], key=lambda x: x["acc"])[:3]
        print(f"    top-acc clusters: {[(p['n'], round(p['acc'],3), round(p['level_mean'],2)) for p in pc]}")
        print(f"    bot-acc clusters: {[(p['n'], round(p['acc'],3), round(p['level_mean'],2)) for p in pc_bot]}")

    print("\n(b') K-means on 1.5B prefill ...")
    km15 = kmeans_alignment(pf15, c15, levels)
    for k in [5, 6, 8, 10]:
        e = km15[f"k{k}"]
        print(f"  k={k}: ARI(correct)={e['ari_correct']:.4f} ARI(level)={e['ari_level']}")

    out["kmeans_7b"] = km7
    out["kmeans_15b"] = km15

    with open(OUT_DIR / "phase3_mechanistic.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved: {OUT_DIR/'phase3_mechanistic.json'}")


if __name__ == "__main__":
    main()
