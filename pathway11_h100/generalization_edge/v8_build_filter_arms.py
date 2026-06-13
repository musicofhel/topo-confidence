#!/usr/bin/env python3
"""v8_build_filter_arms.py — SPEC v8 Phase 2 local trace curation (CPU).

Consumes the teacher generation (pod_results/v8_teacher_{key}.json.gz; key
frozen at G1) and the train pool (pod_payload/v8_math_train_4000.jsonl.gz),
grades every teacher trace with the pinned grader, fits the frozen free gate on
the Math-7B MATH-500 DEV cell (true labels live in DEV, never in train) at the
precision-0.90 operating point, applies it frozen to the train traces, and emits
the four matched-N filter arms:

  (a) GT-filter   : keep traces whose final answer is correct (train labels).
  (b) Gate-filter : keep traces whose teacher free-gate score >= the dev
                    precision-0.90 threshold (the H-L3 zero-label pattern).
  (c) Unfiltered  : random N from the full pool.
  (d) Length-match: random N from the full pool, length distribution matched
                    to arm (b)'s keep-set (10 quantile bins) — the step-length
                    confound made into a control.

N = min(|a keep|, |b keep|); every arm is subsampled to N (seed 0). Writes
pod_payload/v8_traces_{a,b,c,d}.jsonl.gz (rows {idx, problem, text}) plus
results/v8_filter_arms_report.json (filter confusion matrix vs train labels,
N, per-arm length distributions).
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
POD = HERE / "pod_results"
PAYLOAD = HERE / "pod_payload"
RESULTS = HERE / "results"

sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/musicofhel/topo-confidence")
from pathway8_layerwise.extract_math500 import (  # noqa: E402
    _extract_boxed, check_correct)

TEACHER_KEY = "math7b"          # frozen at G1 (H-N CONFIRMED -> Qwen2.5-Math-7B)
DEV_KEY = "math7b"              # gate fit on the same model's MATH-500 cell
PRECISION_TARGET = 0.90
SEED = 0


def load_gz(path: Path):
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def load_jsonl(path: Path):
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh]


def fit_gate(X, y):
    sc = StandardScaler().fit(X)
    est = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)
    return lambda Z: est.predict_proba(sc.transform(Z))[:, 1]


def precision_threshold(scores, labels, target):
    """Smallest (most inclusive) score threshold whose kept-set precision
    >= target. Returns (threshold, achieved_precision, achievable)."""
    order = np.argsort(-scores)
    best = None
    for t in scores[order]:
        keep = scores >= t
        if keep.sum() == 0:
            continue
        prec = float(labels[keep].mean())
        if prec >= target:
            best = (float(t), prec)
    if best is not None:
        return best[0], best[1], True
    # unreachable target: fall back to the highest-precision single threshold
    t = float(scores[order][0])
    return t, float(labels[scores >= t].mean()), False


def subsample(idx_pool, n, rng):
    idx_pool = np.asarray(idx_pool)
    if len(idx_pool) <= n:
        return idx_pool
    return rng.choice(idx_pool, size=n, replace=False)


def length_matched(target_lengths, pool_idx, pool_lengths, n, rng):
    """Pick n indices from pool whose n_gen distribution matches
    target_lengths via 10 quantile bins (seed-0 deterministic)."""
    edges = np.quantile(target_lengths, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    tgt_counts, _ = np.histogram(target_lengths, bins=edges)
    # scale target bin counts to total n (they already sum to len(target))
    pool_lengths = np.asarray(pool_lengths)
    pool_idx = np.asarray(pool_idx)
    chosen, shortfall = [], 0
    for b in range(10):
        want = int(round(tgt_counts[b] * n / tgt_counts.sum()))
        in_bin = pool_idx[(pool_lengths >= edges[b]) &
                          (pool_lengths < edges[b + 1])]
        if len(in_bin) <= want:
            chosen.extend(in_bin.tolist())
            shortfall += want - len(in_bin)
        else:
            chosen.extend(rng.choice(in_bin, size=want,
                                     replace=False).tolist())
    if shortfall > 0:  # top up randomly from unused pool to hit N
        remaining = np.setdiff1d(pool_idx, np.asarray(chosen))
        if len(remaining):
            chosen.extend(rng.choice(remaining,
                                     size=min(shortfall, len(remaining)),
                                     replace=False).tolist())
    return np.asarray(chosen[:n]), shortfall


def write_arm(name, positions, teacher, prob_by_idx, report):
    rows = []
    for p in positions:
        ix = teacher["idx"][p]
        rows.append({"idx": ix, "problem": prob_by_idx[ix],
                     "text": teacher["text"][p]})
    out = PAYLOAD / f"v8_traces_{name}.jsonl.gz"
    with gzip.open(out, "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    lens = [int(teacher["n_gen"][p]) for p in positions]
    report["arms"][name] = {
        "n": len(rows), "file": out.name,
        "len_mean": float(np.mean(lens)), "len_median": float(np.median(lens)),
        "len_p10": float(np.percentile(lens, 10)),
        "len_p90": float(np.percentile(lens, 90))}
    print(f"  arm {name}: N={len(rows)} len(mean/med)="
          f"{np.mean(lens):.0f}/{np.median(lens):.0f} -> {out.name}")


def main():
    rng = np.random.default_rng(SEED)
    teacher = load_gz(POD / f"v8_teacher_{TEACHER_KEY}.json.gz")
    n_tr = len(teacher["idx"])
    print(f"teacher {teacher['model_name']}: {n_tr} traces", flush=True)

    train = load_jsonl(PAYLOAD / "v8_math_train_4000.jsonl.gz")
    prob_by_idx = {r["idx"]: r["problem"] for r in train}
    gold_by_idx = {r["idx"]: _extract_boxed(r["solution"]) for r in train}

    # ---- grade teacher traces (GT labels for arm a + confusion matrix) ----
    correct = np.zeros(n_tr, dtype=bool)
    for i, (ix, text) in enumerate(zip(teacher["idx"], teacher["text"])):
        correct[i] = bool(check_correct(text, gold_by_idx[ix]))
        if (i + 1) % 500 == 0:
            print(f"  graded {i+1}/{n_tr} (acc so far "
                  f"{correct[:i+1].mean():.3f})", flush=True)
    print(f"teacher train accuracy: {correct.mean():.4f}", flush=True)

    # ---- fit the frozen free gate on the Math-7B MATH-500 DEV cell ----
    dev = load_gz(POD / f"v8_frontier_{DEV_KEY}.json.gz")
    y_dev = np.array([bool(check_correct(t, a))
                      for t, a in zip(dev["text"], dev["answer"])])
    Xd = np.column_stack([np.array(dev["n_gen"], float),
                          np.nan_to_num(np.array(dev["mean_lp"], float))])
    scorer = fit_gate(Xd, y_dev)
    s_dev = scorer(Xd)
    thr, prec_at_thr, reachable = precision_threshold(s_dev, y_dev,
                                                      PRECISION_TARGET)

    # apply frozen to train traces
    Xt = np.column_stack([np.array(teacher["n_gen"], float),
                          np.nan_to_num(np.array(teacher["mean_lp"], float))])
    s_tr = scorer(Xt)
    gate_keep = s_tr >= thr

    # ---- confusion matrix of the zero-label gate vs train truth ----
    tp = int((gate_keep & correct).sum())
    fp = int((gate_keep & ~correct).sum())
    fn = int((~gate_keep & correct).sum())
    tn = int((~gate_keep & ~correct).sum())
    gate_prec = tp / max(tp + fp, 1)
    gate_rec = tp / max(tp + fn, 1)

    n_a = int(correct.sum())
    n_b = int(gate_keep.sum())
    N = min(n_a, n_b)
    print(f"gate dev-precision target {PRECISION_TARGET} -> thr {thr:.4f} "
          f"(dev prec {prec_at_thr:.3f}, reachable={reachable})", flush=True)
    print(f"|GT-correct|={n_a}  |gate-keep|={n_b}  -> matched N={N}", flush=True)
    print(f"gate vs train-truth: precision {gate_prec:.3f} recall "
          f"{gate_rec:.3f}  [tp{tp} fp{fp} fn{fn} tn{tn}]", flush=True)

    report = {"spec": "v8 Phase 2 filter arms (EXP-91)",
              "teacher": teacher["model_name"], "n_train": n_tr,
              "teacher_train_acc": float(correct.mean()),
              "gate_threshold": thr, "gate_dev_precision_at_thr": prec_at_thr,
              "gate_precision_reachable": reachable,
              "keep_sizes": {"gt_correct": n_a, "gate_keep": n_b},
              "matched_N": N,
              "gate_vs_truth": {"precision": gate_prec, "recall": gate_rec,
                                "tp": tp, "fp": fp, "fn": fn, "tn": tn},
              "arms": {}}

    # ---- build the four arms at matched N ----
    a_pos = subsample(np.where(correct)[0], N, rng)
    b_pos = subsample(np.where(gate_keep)[0], N, rng)
    c_pos = subsample(np.arange(n_tr), N, rng)
    b_lengths = np.array(teacher["n_gen"], float)[np.where(gate_keep)[0]]
    d_pos, shortfall = length_matched(
        b_lengths, np.arange(n_tr), np.array(teacher["n_gen"], float), N, rng)
    report["arm_d_length_shortfall"] = int(shortfall)

    write_arm("a", a_pos, teacher, prob_by_idx, report)
    write_arm("b", b_pos, teacher, prob_by_idx, report)
    write_arm("c", c_pos, teacher, prob_by_idx, report)
    write_arm("d", d_pos, teacher, prob_by_idx, report)

    RESULTS.mkdir(exist_ok=True)
    f = RESULTS / "v8_filter_arms_report.json"
    f.write_text(json.dumps(report, indent=1, default=float))
    print(f"-> {f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
