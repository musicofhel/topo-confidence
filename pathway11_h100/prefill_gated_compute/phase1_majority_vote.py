#!/usr/bin/env python3
"""Experiment 2 / Phase 1: real majority-vote accuracy at K=1, 2, 4, 8.

For each problem (0..499):
  1. Load the K=8 cached samples from pathway11_h100/data/k8_selfconsistency/problem_XXX.npz.
     (texts: 8 object strs; correct: (8,) bool from math-verify at extraction time.)
  2. Extract \\boxed{...} from each sample text.
  3. Cluster the 8 answers into equivalence classes:
     - first by exact extracted-string equality (fast path),
     - then merge clusters whose representatives are math-verify equivalent.
     The "correct" cluster (if any) is well-defined: all samples whose `correct` flag
     is True land in the same equivalence class (they all verify equal to gt).
  4. Majority vote = argmax cluster size. Ties broken by picking the cluster that
     contains the first-index sample (arbitrary, deterministic).
     Majority correct ⇔ the winning cluster is the "ground-truth" cluster
     ⇔ any member of the winning cluster has cached correct=True.
  5. For K=8 → single majority vote. For K∈{1,2,4} average over n_subsample=50
     random subsets (seeded) using the same clustering logic.

Writes phase1_majority_vote.npz:
    problem_indices: (n_problems,) int
    gt_answers: (n_problems,) object[str]
    per_sample_boxed: (n_problems, 8) object[str]   # raw extracted strings
    per_sample_correct: (n_problems, 8) bool        # cached from stage 3
    k1_correct_per_problem: (n_problems,) bool      # sample 0 correct? (deterministic: K=1 = first sample)
    k1_mean_acc: float                               # averaged over 50 subsamples of K=1 ∀ problem
    k2_mean_acc: float
    k4_mean_acc: float
    k8_majority_correct: (n_problems,) bool
    k8_mean_acc: float
    per_k_accuracy: {K: accuracy}
    per_problem_majority_correct_at_K: {K: (n_problems,) bool (K=8 deterministic, otherwise expected fraction over subsamples)}
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

try:
    from math_verify import parse, verify  # type: ignore
    HAS_MV = True
except Exception:
    HAS_MV = False

ROOT = Path("/home/musicofhel/topo-confidence")
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
OUT_DIR = ROOT / "pathway11_h100/prefill_gated_compute"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PROBLEMS = 500
K_FULL = 8
K_VALUES = [1, 2, 4, 8]
N_SUBSAMPLE = 50
SEED = 9999


def extract_boxed(text: str) -> str:
    """Extract \\boxed{...} content with balanced braces; fallback to last number."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        import re
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return nums[-1] if nums else ""
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1].strip()


def mv_equiv(a: str, b: str) -> bool:
    """True if two LaTeX strings are math-equivalent per math-verify."""
    if a == b:
        return True
    if not HAS_MV:
        return False
    try:
        pa = parse(f"${a}$")
        pb = parse(f"${b}$")
        return bool(verify(pa, pb))
    except Exception:
        return False


def cluster_samples(boxed: list[str], correct_flags: np.ndarray) -> list[list[int]]:
    """Cluster sample indices into math-equivalence classes.

    Fast path: string equality. Slow path: pairwise math-verify for representatives.

    Guarantees: all samples with correct_flags[k]=True end up in the same cluster
    (they're all equivalent to ground truth, and math-verify is transitive-ish
    enough for this use).
    """
    n = len(boxed)
    # Fast-path buckets by exact string.
    fast_buckets: dict[str, list[int]] = {}
    for i, s in enumerate(boxed):
        fast_buckets.setdefault(s, []).append(i)

    # Now merge fast buckets whose representatives are math-equivalent.
    reps = list(fast_buckets.keys())
    parent = {r: r for r in reps}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # Only run math-verify pairs that weren't already string-equal.
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            if find(reps[i]) == find(reps[j]):
                continue
            if mv_equiv(reps[i], reps[j]):
                union(reps[i], reps[j])

    # Collect final clusters.
    clusters: dict[str, list[int]] = {}
    for r, idxs in fast_buckets.items():
        root = find(r)
        clusters.setdefault(root, []).extend(idxs)

    cluster_list = list(clusters.values())
    # Sanity: all correct samples should be in ONE cluster (they all verify to GT).
    if correct_flags.any():
        correct_idxs = set(int(i) for i in np.where(correct_flags)[0])
        found_in = [ci for ci, c in enumerate(cluster_list) if set(c) & correct_idxs]
        if len(found_in) > 1:
            # Merge them — math-verify pairwise may be non-transitive for weird inputs;
            # correctness-equivalence trumps.
            merged = []
            for ci in sorted(found_in, reverse=True):
                merged.extend(cluster_list.pop(ci))
            cluster_list.append(merged)

    # Stable ordering: by smallest index in each cluster.
    cluster_list.sort(key=lambda c: min(c))
    return cluster_list


def majority_correct(subsample_indices: list[int],
                     clusters: list[list[int]],
                     correct_flags: np.ndarray) -> bool:
    """Majority-vote correctness over a subsample.

    A cluster 'vote count' is the number of subsample members it contains.
    Winner is argmax; ties broken by whichever cluster contains the smallest
    subsample index (deterministic, doesn't prefer correct or incorrect).
    Winner is correct iff any of its members has correct_flags=True.
    """
    sub = set(subsample_indices)
    votes: list[tuple[int, int, list[int]]] = []
    for cl in clusters:
        members_in_sub = [i for i in cl if i in sub]
        if not members_in_sub:
            continue
        votes.append((len(members_in_sub), min(members_in_sub), cl))
    if not votes:
        return False
    votes.sort(key=lambda v: (-v[0], v[1]))  # max votes first, tie break by smallest index
    winner_cluster = votes[0][2]
    return bool(correct_flags[winner_cluster].any())


def main():
    print("=" * 70)
    print("Experiment 2 / Phase 1: Majority-vote accuracy at K=1,2,4,8")
    print("=" * 70)

    rng = random.Random(SEED)

    # Load MATH-500 ground-truth answers (for reference; not needed for majority vote)
    # We only need the cached per-sample `correct` flags.
    per_problem_boxed: list[list[str]] = []
    per_problem_correct: list[np.ndarray] = []
    per_problem_clusters: list[list[list[int]]] = []
    missing = []

    for i in range(N_PROBLEMS):
        fp = K8_DIR / f"problem_{i:03d}.npz"
        if not fp.exists():
            missing.append(i)
            continue
        with np.load(fp, allow_pickle=True) as d:
            texts = d["texts"]
            correct = np.asarray(d["correct"]).astype(bool)
        boxed = [extract_boxed(str(t)) for t in texts]
        clusters = cluster_samples(boxed, correct)
        per_problem_boxed.append(boxed)
        per_problem_correct.append(correct)
        per_problem_clusters.append(clusters)
        if (i + 1) % 50 == 0:
            print(f"  clustered {i+1}/{N_PROBLEMS}")

    n_ok = len(per_problem_boxed)
    print(f"  loaded {n_ok}/{N_PROBLEMS} problems (missing: {missing[:10]}{'...' if len(missing)>10 else ''})")

    # Per-K accuracy.
    k8_maj_correct = np.zeros(n_ok, dtype=bool)
    per_k_acc = {}
    per_problem_mean_maj_at_K: dict[int, np.ndarray] = {}

    for K in K_VALUES:
        acc_accum = np.zeros(n_ok)
        if K == K_FULL:
            n_draws = 1
            for p in range(n_ok):
                sub = list(range(K_FULL))
                c = majority_correct(sub, per_problem_clusters[p], per_problem_correct[p])
                acc_accum[p] = float(c)
                k8_maj_correct[p] = c
        else:
            n_draws = N_SUBSAMPLE
            for _ in range(N_SUBSAMPLE):
                for p in range(n_ok):
                    sub = rng.sample(range(K_FULL), K)
                    c = majority_correct(sub, per_problem_clusters[p], per_problem_correct[p])
                    acc_accum[p] += float(c)
            acc_accum /= N_SUBSAMPLE

        per_problem_mean_maj_at_K[K] = acc_accum
        overall = float(acc_accum.mean())
        per_k_acc[K] = overall
        print(f"  K={K}: overall acc = {overall:.4f}  (averaged over {n_draws} draw{'s' if n_draws!=1 else ''})")

    # Extra: K=1 deterministic (sample 0) for use as "K=1 greedy" proxy if needed.
    # But Stage 2 has its own K=1 greedy labels; those are the real ones. This is just the
    # K=8 pool's sample 0.
    k1_sample0_correct = np.array(
        [per_problem_correct[p][0] for p in range(n_ok)], dtype=bool
    )
    print(f"  sample-0 correct (one-of-eight K=1 realization): {k1_sample0_correct.mean():.4f}")

    # Save.
    out_path = OUT_DIR / "phase1_majority_vote.npz"
    # (dtype=object) for per-sample boxed strings (variable length)
    per_sample_boxed_arr = np.empty((n_ok, K_FULL), dtype=object)
    per_sample_correct_arr = np.zeros((n_ok, K_FULL), dtype=bool)
    for p in range(n_ok):
        for k in range(K_FULL):
            per_sample_boxed_arr[p, k] = per_problem_boxed[p][k]
            per_sample_correct_arr[p, k] = per_problem_correct[p][k]

    np.savez_compressed(
        out_path,
        problem_indices=np.array([i for i in range(N_PROBLEMS) if i not in missing], dtype=int),
        per_sample_boxed=per_sample_boxed_arr,
        per_sample_correct=per_sample_correct_arr,
        k1_sample0_correct=k1_sample0_correct,
        k8_majority_correct=k8_maj_correct,
        mean_maj_at_K1=per_problem_mean_maj_at_K[1],
        mean_maj_at_K2=per_problem_mean_maj_at_K[2],
        mean_maj_at_K4=per_problem_mean_maj_at_K[4],
        mean_maj_at_K8=per_problem_mean_maj_at_K[8],
    )
    print(f"\nSaved: {out_path}")

    summary = {
        "n_problems": n_ok,
        "missing": missing,
        "per_k_accuracy": per_k_acc,
        "k1_sample0_accuracy": float(k1_sample0_correct.mean()),
        "k8_majority_accuracy": float(k8_maj_correct.mean()),
        "seed": SEED,
        "n_subsample_per_K": N_SUBSAMPLE,
    }
    with open(OUT_DIR / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {OUT_DIR/'phase1_summary.json'}")


if __name__ == "__main__":
    main()
