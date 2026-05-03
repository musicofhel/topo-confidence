"""FE115 — Song-Zhong pos/ctx decomposition on L19 (Qwen-2.5-1.5B).

Direct refutation test for F-3. Decomposes hidden states into
    s_i[t] = μ + pos_t + ctx_i + resid_i[t]
and asks whether the prefill/final orthogonality (raw cos = 0.046) is
genuinely a "two-circuits" geometric distinction or a positional artifact.

If F-3 is structural, the cosine between prefill-DoM and final-DoM stays
small after removing μ + pos_t + ctx_i. If F-3 is positional, the cosine
jumps once pos_t is subtracted.

Method:
  Pass 1: streaming accumulators for μ (global) and pos_t (per-position).
  Pass 2: ctx_i per problem = mean_t(s_i[t]) − μ.
  Pass 3: residuals at prefill (t=0) and final (t=T_i−1) positions only.
  Probes: 5-fold stratified OOF DoM on prefill_resid / final_resid.

Reference: Song & Zhong (2310.04861), "Uncovering Hidden Geometry in
Transformers via Disentangling Position and Context".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/home/musicofhel/topo-confidence")
NPZ_DIR = ROOT / "pathway8_layerwise/data/math500"
OUT_JSON = ROOT / "pathway11_h100/song_zhong/results.json"

LAYER = 19
N_FOLDS = 5
SEED = 9999


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


def oof_dom_auroc(X: np.ndarray, y: np.ndarray, k: int, seed: int) -> float:
    folds = stratified_kfold(y, k, seed)
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train = np.ones(n, dtype=bool); train[test_idx] = False
        d = X[train & y].mean(axis=0) - X[train & ~y].mean(axis=0)
        scores[test_idx] = X[test_idx] @ d
    return auroc(scores, y)


def main() -> int:
    files = sorted(NPZ_DIR.glob("problem_*.npz"))
    if len(files) != 500:
        print(f"MISSING_REGEN_INPUT expected 500 files in {NPZ_DIR}, got {len(files)}",
              file=sys.stderr)
        return 2

    # ---- Pass 0: load lengths & labels ----
    lengths = np.zeros(500, dtype=np.int64)
    correct = np.zeros(500, dtype=bool)
    for i, f in enumerate(files):
        d = np.load(f)
        lengths[i] = d["states"].shape[1]
        correct[i] = bool(d["correct"])
    T_max = int(lengths.max())
    hidden = 1536
    print(f"n=500, T_min={lengths.min()}, T_median={int(np.median(lengths))}, T_max={T_max}")
    print(f"correct: {correct.sum()}/500")

    # ---- Pass 1: streaming μ and pos_t accumulators ----
    pos_sum = np.zeros((T_max, hidden), dtype=np.float64)
    pos_count = np.zeros(T_max, dtype=np.int64)
    global_sum = np.zeros(hidden, dtype=np.float64)
    global_count = 0

    for i, f in enumerate(files):
        s = np.load(f)["states"][LAYER].astype(np.float64)  # (T_i, 1536)
        T_i = s.shape[0]
        pos_sum[:T_i] += s
        pos_count[:T_i] += 1
        global_sum += s.sum(axis=0)
        global_count += T_i

    mu = global_sum / global_count                        # (1536,)
    # pos_t = mean_t − μ; well-defined for each t with pos_count[t] > 0
    pos_t = np.zeros((T_max, hidden), dtype=np.float64)
    valid = pos_count > 0
    pos_t[valid] = pos_sum[valid] / pos_count[valid, None] - mu

    # ---- Pass 2 & 3: collect prefill + final, raw and residualized ----
    prefill_raw = np.zeros((500, hidden), dtype=np.float64)
    final_raw = np.zeros((500, hidden), dtype=np.float64)
    prefill_resid = np.zeros((500, hidden), dtype=np.float64)
    final_resid = np.zeros((500, hidden), dtype=np.float64)

    for i, f in enumerate(files):
        s = np.load(f)["states"][LAYER].astype(np.float64)  # (T_i, 1536)
        T_i = s.shape[0]
        ctx_i = s.mean(axis=0) - mu                         # (1536,)

        s0 = s[0]
        sT = s[T_i - 1]

        prefill_raw[i] = s0
        final_raw[i] = sT
        # resid = s − μ − pos_t − ctx_i
        prefill_resid[i] = s0 - mu - pos_t[0] - ctx_i
        final_resid[i] = sT - mu - pos_t[T_i - 1] - ctx_i

    # ---- DoM directions ----
    def dom(X: np.ndarray) -> np.ndarray:
        return X[correct].mean(axis=0) - X[~correct].mean(axis=0)

    d_prefill_raw = dom(prefill_raw)
    d_final_raw = dom(final_raw)
    d_prefill_resid = dom(prefill_resid)
    d_final_resid = dom(final_resid)

    def cos(a, b) -> float:
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float((a @ b) / (na * nb))

    cos_raw = cos(d_prefill_raw, d_final_raw)
    cos_resid = cos(d_prefill_resid, d_final_resid)

    # ---- 5-fold OOF DoM AUROC ----
    auroc_prefill_raw = oof_dom_auroc(prefill_raw, correct, N_FOLDS, SEED)
    auroc_final_raw = oof_dom_auroc(final_raw, correct, N_FOLDS, SEED)
    auroc_prefill_resid = oof_dom_auroc(prefill_resid, correct, N_FOLDS, SEED)
    auroc_final_resid = oof_dom_auroc(final_resid, correct, N_FOLDS, SEED)

    F3_COS_THRESHOLD = 0.30
    out = {
        "n": 500,
        "layer": LAYER,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "cos_prefill_final_raw": cos_raw,
        "cos_prefill_final_resid": cos_resid,
        "auroc_prefill_raw": auroc_prefill_raw,
        "auroc_final_raw": auroc_final_raw,
        "auroc_prefill_resid": auroc_prefill_resid,
        "auroc_final_resid": auroc_final_resid,
        "auroc_prefill_drop": auroc_prefill_raw - auroc_prefill_resid,
        "auroc_final_drop": auroc_final_raw - auroc_final_resid,
        "f3_cos_threshold": F3_COS_THRESHOLD,
        "f3_orthogonality_holds": cos_resid < F3_COS_THRESHOLD,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    for k in ("cos_prefill_final_raw", "cos_prefill_final_resid",
              "auroc_prefill_raw", "auroc_final_raw",
              "auroc_prefill_resid", "auroc_final_resid",
              "auroc_prefill_drop", "auroc_final_drop"):
        print(f"{k}={out[k]:.10f}")
    print(f"f3_orthogonality_holds={int(out['f3_orthogonality_holds'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
