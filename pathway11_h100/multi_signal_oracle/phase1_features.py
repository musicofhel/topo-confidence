#!/usr/bin/env python3
"""Exp 2b / Phase 1: build per-problem feature matrix + 3-class oracle target.

Features (per problem, 500 total):
  - prefill_dom    : OOF DoM score at L19 prefill-end (float; already in phase2_prefill_dom.npz)
  - finaltok_dom   : OOF DoM score at L19 final-token (float; already in phase2_prefill_dom.npz)
  - prefill_lpr    : per-problem "local PR" = PR of 20 nearest neighbors on prefill activations
  - finaltok_lpr   : per-problem "local PR" on final-token activations
  - seq_len        : greedy K=1 sampled T=0.7 generation length (int)
  - mean_logprob   : mean token logprob for K=1 sample (float; bonus)

Oracle 3-class target (from Exp 2 Phase 4 buckets):
  0 = use K=1  (bucket A: always-right;   and bucket D: pathological K=1-right-but-K=8-wrong — MUST route to K=1!)
  1 = use K=8  (bucket B: recoverable; K=8 fixes it)
  2 = refuse   (bucket C: never-right; neither K=1 nor K=8 correct)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE_NPZ = ROOT / "pathway11_h100/prefill_inversion/cache/m7b_prefill.npz"
PHASE2_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
PHASE1_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase1_majority_vote.npz"
OUT_DIR = ROOT / "pathway11_h100/multi_signal_oracle"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_LOCAL = 20  # neighborhood size for local PR


def participation_ratio_np(X: np.ndarray) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    s2 = s ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum() + 1e-30)


def local_pr(H: np.ndarray, k: int = K_LOCAL) -> np.ndarray:
    """For each row, compute the PR of its k nearest neighbors (including itself)."""
    H64 = H.astype(np.float64, copy=False)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(H64)
    _, idx = nn.kneighbors(H64)  # (n, k)
    lpr = np.zeros(len(H64))
    for i in range(len(H64)):
        neighbors = H64[idx[i]]
        lpr[i] = participation_ratio_np(neighbors)
    return lpr


def bucket_labels(k1_correct: np.ndarray, k8_correct: np.ndarray) -> np.ndarray:
    """Return per-problem bucket label: 'A','B','C','D'."""
    lab = np.empty(len(k1_correct), dtype="U1")
    for i in range(len(k1_correct)):
        if k1_correct[i] and k8_correct[i]:
            lab[i] = "A"  # always right
        elif (not k1_correct[i]) and k8_correct[i]:
            lab[i] = "B"  # recoverable
        elif (not k1_correct[i]) and (not k8_correct[i]):
            lab[i] = "C"  # never right
        else:
            lab[i] = "D"  # pathological: K=1 right, K=8 wrong
    return lab


def bucket_to_3class(bkt: np.ndarray) -> np.ndarray:
    """A or D -> 0 (use K=1); B -> 1 (use K=8); C -> 2 (refuse)."""
    out = np.zeros(len(bkt), dtype=int)
    out[(bkt == "A") | (bkt == "D")] = 0
    out[bkt == "B"] = 1
    out[bkt == "C"] = 2
    return out


def main():
    print("=" * 70)
    print("Exp 2b / Phase 1: build features + 3-class oracle target")
    print("=" * 70)

    # Activations
    print("\nLoading 7B prefill cache ...")
    cache = np.load(CACHE_NPZ)
    H_pre = cache["prefill"].astype(np.float32)
    H_fin = cache["final_tok"].astype(np.float32)
    correct_legacy = cache["correct"].astype(bool)
    print(f"  prefill: {H_pre.shape}, final_tok: {H_fin.shape}, n={len(H_pre)}")

    # Phase 2 features
    print("Loading Phase 2 OOF DoM + seq_len ...")
    p2 = np.load(PHASE2_NPZ, allow_pickle=True)
    prefill_dom = p2["prefill_score"].astype(np.float32)
    finaltok_dom = p2["final_token_score"].astype(np.float32)
    seq_len = p2["seq_len"].astype(np.int64)
    mean_logprob = p2["mean_logprob"].astype(np.float32)
    k1_correct = p2["correct_k1"].astype(bool)
    print(f"  prefill_dom AUROC-ish mean={prefill_dom.mean():.3f} std={prefill_dom.std():.3f}")
    print(f"  seq_len mean={seq_len.mean():.0f} std={seq_len.std():.0f}")

    # K=8 majority outcomes
    print("Loading Phase 1 majority-vote outcomes ...")
    p1 = np.load(PHASE1_NPZ, allow_pickle=True)
    k8_correct = p1["k8_majority_correct"].astype(bool)
    k1_sample0_correct = p1["k1_sample0_correct"].astype(bool)
    print(f"  K=1 greedy acc = {k1_correct.mean():.3f} "
          f"(K=1 sample[0] sampled acc = {k1_sample0_correct.mean():.3f})")
    print(f"  K=8 majority acc = {k8_correct.mean():.3f}")

    # 3-class oracle target
    bkt = bucket_labels(k1_correct, k8_correct)
    y3 = bucket_to_3class(bkt)
    n_by = dict(zip(*np.unique(bkt, return_counts=True)))
    print(f"\nBucket counts: {dict(n_by)}")
    print(f"3-class counts: K=1 (A+D)={int((y3==0).sum())}, "
          f"K=8 (B)={int((y3==1).sum())}, "
          f"refuse (C)={int((y3==2).sum())}")

    # Local PR
    print(f"\nComputing per-problem local PR (k={K_LOCAL} neighbors) ...")
    print("  prefill activations ...")
    prefill_lpr = local_pr(H_pre, K_LOCAL)
    print(f"    mean={prefill_lpr.mean():.2f} std={prefill_lpr.std():.2f} "
          f"range=[{prefill_lpr.min():.2f}, {prefill_lpr.max():.2f}]")
    print("  final-token activations ...")
    finaltok_lpr = local_pr(H_fin, K_LOCAL)
    print(f"    mean={finaltok_lpr.mean():.2f} std={finaltok_lpr.std():.2f} "
          f"range=[{finaltok_lpr.min():.2f}, {finaltok_lpr.max():.2f}]")

    # Per-bucket summary
    print("\nPer-bucket feature means:")
    for b in ["A", "B", "C", "D"]:
        m = bkt == b
        print(f"  {b} (n={int(m.sum())}): prefill_dom={prefill_dom[m].mean():+.3f} "
              f"prefill_lpr={prefill_lpr[m].mean():.2f} "
              f"finaltok_dom={finaltok_dom[m].mean():+.3f} "
              f"finaltok_lpr={finaltok_lpr[m].mean():.2f} "
              f"seq_len={seq_len[m].mean():.0f}")

    # Save feature matrix
    out_path = OUT_DIR / "features.npz"
    np.savez_compressed(
        out_path,
        prefill_dom=prefill_dom,
        finaltok_dom=finaltok_dom,
        prefill_lpr=prefill_lpr,
        finaltok_lpr=finaltok_lpr,
        seq_len=seq_len,
        mean_logprob=mean_logprob,
        k1_correct=k1_correct,
        k8_correct=k8_correct,
        bucket=bkt,
        y3=y3,
    )
    print(f"\nSaved: {out_path}")

    # Small json summary for quick eyeballing
    per_bucket = {}
    for b in ["A", "B", "C", "D"]:
        m = bkt == b
        per_bucket[b] = dict(
            n=int(m.sum()),
            prefill_dom_mean=float(prefill_dom[m].mean()),
            prefill_lpr_mean=float(prefill_lpr[m].mean()),
            finaltok_dom_mean=float(finaltok_dom[m].mean()),
            finaltok_lpr_mean=float(finaltok_lpr[m].mean()),
            seq_len_mean=float(seq_len[m].mean()),
        )
    summary = dict(
        n_problems=int(len(H_pre)),
        k_local=K_LOCAL,
        bucket_counts={b: int((bkt == b).sum()) for b in "ABCD"},
        y3_counts={"K1_AD": int((y3 == 0).sum()),
                   "K8_B": int((y3 == 1).sum()),
                   "refuse_C": int((y3 == 2).sum())},
        per_bucket_feature_means=per_bucket,
    )
    with open(OUT_DIR / "phase1_features.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {OUT_DIR/'phase1_features.json'}")


if __name__ == "__main__":
    main()
