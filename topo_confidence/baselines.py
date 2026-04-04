"""Baseline confidence estimators for comparison."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax


def output_entropy(logits_list: list[np.ndarray]) -> np.ndarray:
    """Average per-token entropy over generated tokens.

    Args:
        logits_list: list of (gen_len, vocab_size) arrays, one per problem.

    Returns:
        (n_problems,) array of mean output entropy values.
        Lower entropy = more confident.
    """
    entropies = []
    for logits in logits_list:
        if len(logits) == 0:
            entropies.append(0.0)
            continue
        probs = softmax(logits, axis=-1)
        token_entropy = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
        entropies.append(float(np.mean(token_entropy)))
    return np.array(entropies)


def max_token_probability(logits_list: list[np.ndarray]) -> np.ndarray:
    """Average max token probability over generated tokens.

    Args:
        logits_list: list of (gen_len, vocab_size) arrays.

    Returns:
        (n_problems,) array. Higher = more confident.
    """
    scores = []
    for logits in logits_list:
        if len(logits) == 0:
            scores.append(1.0)
            continue
        probs = softmax(logits, axis=-1)
        max_probs = np.max(probs, axis=-1)
        scores.append(float(np.mean(max_probs)))
    return np.array(scores)


def first_token_probability(logits_list: list[np.ndarray]) -> np.ndarray:
    """Probability of the first generated token (greedy).

    A simple proxy: if the model is confident about how to start
    its answer, it's more likely to be correct overall.

    Args:
        logits_list: list of (gen_len, vocab_size) arrays.

    Returns:
        (n_problems,) array. Higher = more confident.
    """
    scores = []
    for logits in logits_list:
        if len(logits) == 0:
            scores.append(1.0)
            continue
        probs = softmax(logits[0], axis=-1)
        scores.append(float(np.max(probs)))
    return np.array(scores)
