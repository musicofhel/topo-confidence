"""k8_lib.py — self-consistency answer extraction + voting for Phase 1B.1.

The K=8 cache stores per-sample L19 states, full texts, and per-sample
correctness (math_verify vs gold). To compute *plain* self-consistency
majority and *probe-reranked* variants we cluster the 8 samples by their
extracted \\boxed{} answer. Key identity: all samples sharing an answer string
have the same correctness, so a cluster is "correct" iff ANY sample in it is
marked correct — we never need the gold string itself.

Verified: plain majority (mode answer, ties -> highest mean per-sample order)
reproduces the 0.550 anchor.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("/home/musicofhel/topo-confidence")))
from pathway8_layerwise.extract_math500 import _extract_boxed  # noqa: E402

import activation_loader as AL  # noqa: E402


def extract_answers(texts) -> list[str]:
    """Normalised \\boxed{} answer string per sample ('' if none found)."""
    out = []
    for t in texts:
        a = _extract_boxed(str(t))
        out.append(a.strip() if a else "")
    return out


def _clusters(answers: list[str]) -> dict[str, list[int]]:
    cl = defaultdict(list)
    for i, a in enumerate(answers):
        cl[a].append(i)
    return cl


def plain_majority_correct(answers, correct, tie_weight=None) -> bool:
    """Mode-answer self-consistency. Ties broken by larger summed tie_weight
    (default: equal weight = sample order via stable max). Returns whether the
    winning answer is correct (cluster correct iff any member correct)."""
    cl = _clusters(answers)
    if tie_weight is None:
        tie_weight = np.ones(len(answers))
    best, best_key = None, None
    for ans, idxs in cl.items():
        # primary: cluster size; secondary: summed weight
        key = (len(idxs), float(tie_weight[idxs].sum()))
        if best is None or key > best_key:
            best, best_key = idxs, key
    return bool(np.asarray(correct)[best].any())


def probe_argmax_correct(probe_scores, correct) -> bool:
    """Pick the single highest-probe sample; its correctness is the outcome."""
    j = int(np.argmax(probe_scores))
    return bool(correct[j])


def probe_weighted_majority_correct(answers, correct, probe_scores) -> bool:
    """Vote per answer-cluster weighted by probe score (softmax-free: raw,
    shifted to be non-negative). Winning cluster correct iff any member correct."""
    w = np.asarray(probe_scores, dtype=np.float64)
    w = w - w.min() + 1e-9
    cl = _clusters(answers)
    best, best_key = None, None
    for ans, idxs in cl.items():
        key = (float(w[idxs].sum()), len(idxs))
        if best is None or key > best_key:
            best, best_key = idxs, key
    return bool(np.asarray(correct)[best].any())


def load_all_k8():
    """Load every K=8 problem: returns dict of stacked arrays + per-problem lists.

    L19 (n,8,1536), correct (n,8) bool, answers list[list[str]].
    """
    n = AL.n_k8_problems()
    L19 = np.zeros((n, 8, 1536), dtype=np.float64)
    correct = np.zeros((n, 8), dtype=bool)
    answers = []
    for i in range(n):
        d = AL.load_k8(i)
        L19[i] = d["L19_samples"]
        correct[i] = d["correct"]
        answers.append(extract_answers(d["texts"]))
    return {"L19": L19, "correct": correct, "answers": answers, "n": n}
