#!/usr/bin/env python3
"""Experiment 2 / Phase 2: out-of-fold prefill DoM confidence score.

Load position-0 (= last-prefill-token) L19 activations from Stage 2:
    pathway8_layerwise/data/math500/problem_XXX.npz
      states: (29, n_gen, 1536) float16; use states[STEERING_LAYER, 0, :]
      correct: K=1 greedy correctness (bool)

Labels = Stage 2 K=1 greedy correct (expected ≈ 243/500 = 48.6%).

5-fold StratifiedKFold CV:
  For each fold:
    - On train split: mu_correct = mean(h | correct=T), mu_incorrect = mean(h | correct=F)
      DoM direction v = mu_correct - mu_incorrect (NOT normalized — we report projection
      magnitude, not cosine, to keep it interpretable).
    - Score held-out problems as s = (h - mu_center) · v_unit, where
      mu_center = (mu_correct + mu_incorrect)/2 and v_unit = v / ||v||.
      (Center-then-project gives a signed scalar with a natural "0 = boundary" threshold,
      though for the compute-gating experiment we care about quantiles/order, not absolute.)

Also collect side signals for baselines in Phase 3:
  - final-token DoM: states[STEERING_LAYER, -1, :]  (last generated token), OOF via same CV
  - sequence length: n_gen (per problem)
  - mean_logprob: cached scalar

Outputs phase2_prefill_dom.npz:
    problem_indices: (n_problems,) int
    correct_k1: (n_problems,) bool
    prefill_score: (n_problems,) float32  # OOF
    final_token_score: (n_problems,) float32  # OOF
    seq_len: (n_problems,) int
    mean_logprob: (n_problems,) float32
    per_fold_auroc: {"prefill": [5], "final": [5]}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
STAGE2_DIR = ROOT / "pathway8_layerwise/data/math500"
OUT_DIR = ROOT / "pathway11_h100/prefill_gated_compute"

N_PROBLEMS = 500
STEERING_LAYER = 19
SEED = 9999
N_FOLDS = 5


def load_stage2() -> tuple:
    """Load prefill-end and final-token L19 + correctness + side signals."""
    prefill = np.zeros((N_PROBLEMS, 1536), dtype=np.float32)
    final_tok = np.zeros((N_PROBLEMS, 1536), dtype=np.float32)
    correct = np.zeros(N_PROBLEMS, dtype=bool)
    seq_len = np.zeros(N_PROBLEMS, dtype=int)
    mean_lp = np.zeros(N_PROBLEMS, dtype=np.float32)
    missing = []

    for i in range(N_PROBLEMS):
        fp = STAGE2_DIR / f"problem_{i:03d}.npz"
        if not fp.exists():
            missing.append(i)
            continue
        with np.load(fp, allow_pickle=True) as d:
            s = d["states"]  # (29, n_gen, 1536)
            prefill[i] = s[STEERING_LAYER, 0, :].astype(np.float32)
            final_tok[i] = s[STEERING_LAYER, -1, :].astype(np.float32)
            correct[i] = bool(d["correct"])
            seq_len[i] = s.shape[1]
            try:
                mean_lp[i] = float(d["mean_logprob"])
            except Exception:
                mean_lp[i] = np.nan

    return prefill, final_tok, correct, seq_len, mean_lp, missing


def oof_dom_scores(H: np.ndarray, y: np.ndarray, seed: int = SEED) -> tuple:
    """5-fold StratifiedKFold out-of-fold DoM projection score.

    Returns (scores, per_fold_auroc).
    """
    scores = np.zeros(len(y), dtype=np.float32)
    per_fold = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(skf.split(H, y)):
        mu_c = H[tr][y[tr]].mean(axis=0)
        mu_i = H[tr][~y[tr]].mean(axis=0)
        v = mu_c - mu_i
        v_unit = v / (np.linalg.norm(v) + 1e-12)
        mu_center = 0.5 * (mu_c + mu_i)
        s_te = (H[te] - mu_center) @ v_unit
        scores[te] = s_te
        per_fold.append(float(roc_auc_score(y[te], s_te)))
    return scores, per_fold


def main():
    print("=" * 70)
    print("Experiment 2 / Phase 2: OOF prefill DoM score")
    print("=" * 70)

    prefill, final_tok, correct, seq_len, mean_lp, missing = load_stage2()
    if missing:
        print(f"  missing Stage 2 files: {missing[:10]}{'...' if len(missing)>10 else ''}")
        mask = np.array([i not in missing for i in range(N_PROBLEMS)])
        prefill = prefill[mask]; final_tok = final_tok[mask]
        correct = correct[mask]; seq_len = seq_len[mask]; mean_lp = mean_lp[mask]
        problem_indices = np.where(mask)[0]
    else:
        problem_indices = np.arange(N_PROBLEMS)

    n = len(correct)
    print(f"  loaded {n} problems | K=1 greedy accuracy = {correct.mean():.4f} ({correct.sum()}/{n})")

    prefill_score, prefill_folds = oof_dom_scores(prefill, correct)
    final_score, final_folds = oof_dom_scores(final_tok, correct)

    prefill_auroc = float(roc_auc_score(correct, prefill_score))
    final_auroc = float(roc_auc_score(correct, final_score))

    print(f"  prefill L19 DoM: OOF AUROC = {prefill_auroc:.4f}  (per-fold {['%.3f'%x for x in prefill_folds]})")
    print(f"  final-tok L19 DoM: OOF AUROC = {final_auroc:.4f}  (per-fold {['%.3f'%x for x in final_folds]})")
    print(f"  seq_len: min={seq_len.min()} median={int(np.median(seq_len))} max={seq_len.max()}")
    # seq_len correlation with correctness (note: longer sequences might be harder problems or just verbose)
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(seq_len, correct.astype(int))
        print(f"  spearman(seq_len, correct) = {rho:.3f} (p={p:.2e})")
    except Exception:
        pass

    out_path = OUT_DIR / "phase2_prefill_dom.npz"
    np.savez_compressed(
        out_path,
        problem_indices=problem_indices,
        correct_k1=correct,
        prefill_score=prefill_score,
        final_token_score=final_score,
        seq_len=seq_len,
        mean_logprob=mean_lp,
    )
    print(f"\nSaved: {out_path}")

    summary = {
        "n_problems": int(n),
        "k1_greedy_accuracy": float(correct.mean()),
        "prefill_dom_auroc_oof": prefill_auroc,
        "final_token_dom_auroc_oof": final_auroc,
        "per_fold_auroc": {"prefill": prefill_folds, "final": final_folds},
        "seed": SEED,
    }
    with open(OUT_DIR / "phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {OUT_DIR/'phase2_summary.json'}")


if __name__ == "__main__":
    main()
