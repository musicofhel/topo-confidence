"""P11-FE177 — Register-token probe sweep across Qwen-2.5-1.5B prompt positions.

Brinkmann et al. (2604.xxxxx) show transformers offload sub-path storage to
semantically empty *register* tokens. F-3's prefix/final orthogonality
(cos=0.046) hints there may be additional positions carrying redundant
correctness signal. This script trains F-2's logistic-regression correctness
probe at *every* labelled token position in the MATH-500 prompt — instruction
tokens (Question:, Answer:, \\boxed{), separators (\\n, ',', '.'), padding,
prefill, and final — and reports the OOF AUROC of each. Any low-information
position whose probe AUROC matches or exceeds the prefill reference (0.7731)
is flagged as acting as a register-token working-memory slot.

Requires a per-position activation cache (one (500, 1536) array per named
position) which is not part of the headline schema. If absent, the script
exits cleanly with MISSING_REGEN_INPUT so the caller can regenerate it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
# Per-position activation cache: NPZ with a "correct" (500,) bool key and one
# (500, 1536) float array per named token position (e.g. "pos_question",
# "pos_answer", "pos_boxed", "pos_newline", "pos_comma", "pos_period",
# "pos_pad", "pos_prefill", "pos_final").
PER_TOKEN_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_per_token_positions.npz"
# Headline prefill cache — used only to cross-check the prefill column / labels.
MAIN_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/register_token_sweep/results.json"

N_FOLDS = 5
SEED = 9999
PREFILL_REF = 0.7731  # F-2 prefill L19 DoM OOF AUROC (1.5B), the bar to match.
MATCH_TOL = 0.005     # within this of the reference counts as "matches".

# Positions considered "low-information" in Brinkmann et al.'s register sense:
# separators, padding, and instruction-scaffold tokens (not prefill/final).
LOW_INFO_TOKENS = {
    "pos_question", "pos_answer", "pos_boxed",
    "pos_newline", "pos_comma", "pos_period", "pos_pad",
}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def oof_probe_auroc(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> float:
    """F-2 probe: standardized logistic regression, 5-fold OOF probabilities."""
    n = len(y)
    oof = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        scaler = StandardScaler().fit(X[train_mask])
        Xtr = scaler.transform(X[train_mask])
        Xte = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, y[train_mask])
        oof[test_idx] = clf.predict_proba(Xte)[:, 1]
    return auroc(oof, y)


def main() -> int:
    if not PER_TOKEN_CACHE.exists():
        print("MISSING_REGEN_INPUT", PER_TOKEN_CACHE, file=sys.stderr)
        return 2

    blob = np.load(PER_TOKEN_CACHE)
    if "correct" not in blob.files:
        print("MISSING_REGEN_INPUT", PER_TOKEN_CACHE, "(no 'correct' key)", file=sys.stderr)
        return 2
    y = blob["correct"].astype(bool)
    if y.shape != (500,):
        print("MISSING_REGEN_INPUT", PER_TOKEN_CACHE, "(bad 'correct' shape)", file=sys.stderr)
        return 2

    # Cross-check labels against the headline cache when it is present.
    if MAIN_CACHE.exists():
        main_y = np.load(MAIN_CACHE)["correct"].astype(bool)
        if main_y.shape == y.shape and not np.array_equal(main_y, y):
            print("WARN: per-token labels differ from headline cache", file=sys.stderr)

    # Every (500, 1536) array is a candidate token position.
    positions = [
        k for k in blob.files
        if k not in ("correct", "seq_len")
        and getattr(blob[k], "ndim", 0) == 2
        and blob[k].shape == (500, 1536)
    ]
    if not positions:
        print("MISSING_REGEN_INPUT", PER_TOKEN_CACHE, "(no position arrays)", file=sys.stderr)
        return 2

    folds = stratified_kfold(y, N_FOLDS, SEED)

    per_position = {}
    for name in positions:
        X = blob[name].astype(np.float64)
        a = oof_probe_auroc(X, y, folds)
        per_position[name] = {
            "auroc_oof": a,
            "delta_vs_prefill": float(a - PREFILL_REF),
            "low_information": name in LOW_INFO_TOKENS,
        }

    # A register-token candidate is a low-information position whose probe AUROC
    # matches (within tolerance) or exceeds the prefill reference.
    register_candidates = sorted(
        (
            name for name, d in per_position.items()
            if d["low_information"] and d["auroc_oof"] >= PREFILL_REF - MATCH_TOL
        ),
        key=lambda n: per_position[n]["auroc_oof"],
        reverse=True,
    )

    matches_or_exceeds = sorted(
        (name for name, d in per_position.items()
         if d["auroc_oof"] >= PREFILL_REF - MATCH_TOL),
        key=lambda n: per_position[n]["auroc_oof"],
        reverse=True,
    )

    best_name = max(per_position, key=lambda n: per_position[n]["auroc_oof"])

    out = {
        "experiment": "P11-FE177",
        "description": "Register-token probe sweep across prompt positions",
        "prefill_reference_auroc": PREFILL_REF,
        "match_tolerance": MATCH_TOL,
        "n_positions": len(positions),
        "per_position": per_position,
        "best_position": best_name,
        "best_auroc": per_position[best_name]["auroc_oof"],
        "positions_matching_or_exceeding_prefill": matches_or_exceeds,
        "register_token_candidates": register_candidates,
        "register_token_found": bool(register_candidates),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())