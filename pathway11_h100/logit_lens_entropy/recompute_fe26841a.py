"""FE26841a — Logit-lens entropy from L19 prefill activations.

Loads embed_tokens.weight (unembed via tied weights), projects L19 prefill
activations to logits, computes per-sample entropy, uses as correctness
predictor via 5-fold OOF AUROC.

Output: pathway11_h100/results/fe26841a_logit_lens_entropy.json
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
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from safetensors import safe_open

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe26841a_logit_lens_entropy.json"

SAFETENSORS_PATH = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors"

SEED = 9999
N_FOLDS = 5
CHUNK_SIZE = 50


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def main() -> int:
    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float32)
    correct = cache["correct"].astype(bool)
    n = X.shape[0]

    dom_data = np.load(DOM_NPZ)
    dom_scores = dom_data["prefill_score"].astype(np.float32)

    print("Loading embed_tokens.weight via safetensors...")
    f = safe_open(str(SAFETENSORS_PATH), framework="pt")
    import torch
    W_unembed = f.get_tensor("model.embed_tokens.weight").float().numpy()  # (151936, 1536)
    vocab_size = W_unembed.shape[0]
    print(f"Unembed shape: {W_unembed.shape}")

    # Compute entropy in chunks to limit memory
    entropies = np.zeros(n, dtype=np.float64)
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        logits = X[start:end] @ W_unembed.T  # (chunk, vocab_size)
        # Numerically stable softmax
        logits_max = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        # Entropy: -sum(p * log(p + eps))
        H = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        entropies[start:end] = H
        print(f"  Processed {end}/{n} samples")

    del W_unembed

    # Stats by class
    ent_correct = entropies[correct]
    ent_incorrect = entropies[~correct]

    # 5-fold OOF AUROC using entropy as predictor (lower entropy => more confident)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_scores = np.zeros(n, dtype=np.float64)
    for train_idx, test_idx in skf.split(X, correct):
        # Entropy is a single feature; use logistic regression
        lr = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
        lr.fit(entropies[train_idx].reshape(-1, 1), correct[train_idx])
        oof_scores[test_idx] = lr.predict_proba(entropies[test_idx].reshape(-1, 1))[:, 1]

    entropy_auroc = auroc(oof_scores, correct)

    # Raw entropy as score (negate: lower entropy => higher confidence)
    raw_auroc = auroc(-entropies, correct)

    # Spearman correlation with DoM scores
    rho, pval = spearmanr(entropies, dom_scores)

    out = {
        "experiment": "FE26841a",
        "description": "Logit-lens entropy from L19 prefill activations for correctness prediction",
        "n": n,
        "vocab_size": vocab_size,
        "entropy_auroc_oof_logreg": float(entropy_auroc),
        "entropy_auroc_raw_negated": float(raw_auroc),
        "entropy_mean_correct": float(ent_correct.mean()),
        "entropy_std_correct": float(ent_correct.std()),
        "entropy_mean_incorrect": float(ent_incorrect.mean()),
        "entropy_std_incorrect": float(ent_incorrect.std()),
        "entropy_class_gap": float(ent_incorrect.mean() - ent_correct.mean()),
        "spearman_entropy_vs_dom": float(rho),
        "spearman_pvalue": float(pval),
        "dom_auroc_reference": 0.7731,
        "fold_structure": "StratifiedKFold(n_splits=5, shuffle=True, random_state=9999)",
        "meta": {
            "cache": str(CACHE),
            "dom_npz": str(DOM_NPZ),
            "safetensors": str(SAFETENSORS_PATH),
            "seed": SEED,
            "method": "logit_lens_entropy_via_tied_embed_tokens",
            "chunk_size": CHUNK_SIZE,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Entropy AUROC (OOF logreg): {entropy_auroc:.4f}")
    print(f"Entropy AUROC (raw negated): {raw_auroc:.4f}")
    print(f"Entropy correct: {ent_correct.mean():.2f} ± {ent_correct.std():.2f}")
    print(f"Entropy incorrect: {ent_incorrect.mean():.2f} ± {ent_incorrect.std():.2f}")
    print(f"Spearman(entropy, DoM): {rho:.4f} (p={pval:.4e})")
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
