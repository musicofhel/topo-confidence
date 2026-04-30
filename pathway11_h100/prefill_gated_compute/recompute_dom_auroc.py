#!/usr/bin/env python3
"""Tier-1 regen for the prefill / final-token L19 DoM AUROC headline numbers.

Re-runs the smallest meaningful step that produces the headline numbers in
results.json — given a cached intermediate. Two modes:

  --mode=cache   (default, ms):
      Load phase2_prefill_dom.npz (OOF DoM scores + correctness labels),
      recompute roc_auc_score(correct_k1, prefill_score) and ditto for
      final-token. Catches results.json drift from the cache.

  --mode=full    (seconds-to-minutes, requires per-problem NPZs):
      Reload L19 prefill + final-token activations from
      pathway8_layerwise/data/math500/problem_*.npz, redo 5-fold OOF
      DoM, then AUROC. Catches everything cache-mode catches plus
      wrong fold split / wrong layer / bad DoM math / bad labels.

Stdout protocol (consumed by validate_claims.py):
    prefill_oof=<float>
    final_oof=<float>
Exit 0 on success; nonzero on missing intermediates.

Usage:
    python recompute_dom_auroc.py                       # cache mode
    python recompute_dom_auroc.py --mode=full           # full regen
    python recompute_dom_auroc.py --key=prefill_oof     # print just one key
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
STAGE2_DIR = ROOT / "pathway8_layerwise/data/math500"
N_PROBLEMS = 500
STEERING_LAYER = 19
SEED = 9999
N_FOLDS = 5


def from_cache() -> tuple[float, float]:
    if not CACHE_PATH.exists():
        sys.exit(f"MISSING_REGEN_INPUT: {CACHE_PATH} not found — run phase2_prefill_dom.py first")
    with np.load(CACHE_PATH) as d:
        y = d["correct_k1"]
        prefill_oof = float(roc_auc_score(y, d["prefill_score"]))
        final_oof = float(roc_auc_score(y, d["final_token_score"]))
    return prefill_oof, final_oof


def from_full() -> tuple[float, float]:
    if not STAGE2_DIR.exists():
        sys.exit(f"MISSING_REGEN_INPUT: {STAGE2_DIR} not found — Stage 2 NPZs required for --mode=full")
    prefill = np.zeros((N_PROBLEMS, 1536), dtype=np.float32)
    final_tok = np.zeros((N_PROBLEMS, 1536), dtype=np.float32)
    correct = np.zeros(N_PROBLEMS, dtype=bool)
    mask = np.zeros(N_PROBLEMS, dtype=bool)
    for i in range(N_PROBLEMS):
        fp = STAGE2_DIR / f"problem_{i:03d}.npz"
        if not fp.exists():
            continue
        with np.load(fp, allow_pickle=True) as d:
            s = d["states"]
            prefill[i] = s[STEERING_LAYER, 0, :].astype(np.float32)
            final_tok[i] = s[STEERING_LAYER, -1, :].astype(np.float32)
            correct[i] = bool(d["correct"])
            mask[i] = True
    if not mask.all():
        sys.exit(f"MISSING_REGEN_INPUT: {(~mask).sum()} of {N_PROBLEMS} stage-2 NPZs missing")

    def oof(H: np.ndarray, y: np.ndarray) -> float:
        scores = np.zeros(len(y), dtype=np.float32)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for tr, te in skf.split(H, y):
            mu_c = H[tr][y[tr]].mean(axis=0)
            mu_i = H[tr][~y[tr]].mean(axis=0)
            v = mu_c - mu_i
            v_unit = v / (np.linalg.norm(v) + 1e-12)
            mu_center = 0.5 * (mu_c + mu_i)
            scores[te] = (H[te] - mu_center) @ v_unit
        return float(roc_auc_score(y, scores))

    return oof(prefill, correct), oof(final_tok, correct)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cache", "full"], default="cache")
    ap.add_argument("--key", choices=["prefill_oof", "final_oof", "both"], default="both")
    args = ap.parse_args()

    prefill_oof, final_oof = (from_cache() if args.mode == "cache" else from_full())

    if args.key in ("prefill_oof", "both"):
        print(f"prefill_oof={prefill_oof:.10f}")
    if args.key in ("final_oof", "both"):
        print(f"final_oof={final_oof:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
