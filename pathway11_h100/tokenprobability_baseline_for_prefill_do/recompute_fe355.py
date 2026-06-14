"""P11-FE355 — Token-probability baseline for prefill DoM (Pawitan & Holmes framing).

From cached K=1 MATH-500 greedy generations, derive a per-problem mean log-prob of
the predicted answer span ("token-prob"). Three measurements, all OOF 5-fold:

  1. token-prob alone: single-feature logistic regression AUROC vs first-pass
     correctness (a monotone single feature, so this equals the raw token-prob AUROC).
  2. prefill DoM AUROC: OOF difference-of-means direction on the 1536-d L19 prefill
     hidden state (reproduces F-2 ≈ 0.7731).
  3. residual prefill-DoM: per-fold, partial token-prob out of every prefill column
     (regress each column on token-prob using train coeffs, take residuals), refit the
     DoM direction on the residualized train fold, score the residualized test fold.

If the residual-after-token-prob AUROC stays > 0.7, prefill DoM carries information not
explained by token-level probability; if it collapses toward 0.5, F-2 is largely a
logprob re-skin.

Token-prob source: a cached per-problem mean answer-span log-prob from the K=1 greedy
run. This script scans a list of plausible cache locations/keys and prints
MISSING_REGEN_INPUT if none resolve to a (500,) array.
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

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/token_prob_baseline/results.json"

N = 500
N_FOLDS = 5
SEED = 9999

# Candidate (path, key) pairs for an NPZ holding a (500,) mean-logprob array.
LOGPROB_NPZ_CANDIDATES = [
    (ROOT / "pathway11_h100/prefill_gated_compute/phase1_logprobs.npz",
     ("mean_logprob", "answer_logprob", "logprob", "mean_answer_logprob")),
    (ROOT / "pathway11_h100/prefill_gated_compute/results.npz",
     ("mean_logprob", "answer_logprob", "logprob")),
    (ROOT / "pathway11_h100/data/k1_greedy_logprobs.npz",
     ("mean_logprob", "answer_logprob", "logprob")),
    (ROOT / "pathway11_h100/data/k1_greedy/logprobs.npz",
     ("mean_logprob", "answer_logprob", "logprob")),
]

# Candidate JSON files: either a list of 500 records, or a dict with a 500-length list.
LOGPROB_JSON_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_gated_compute/results.json",
    ROOT / "pathway11_h100/prefill_gated_compute/phase1_logprobs.json",
    ROOT / "pathway11_h100/data/k1_greedy_logprobs.json",
]
JSON_KEYS = ("mean_logprob", "answer_logprob", "logprob", "mean_answer_logprob",
             "answer_mean_logprob")

# Per-problem JSON shards: pathway11_h100/data/k1_greedy/problem_NNN.json
K1_SHARD_DIR = ROOT / "pathway11_h100/data/k1_greedy"


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def _extract_from_records(records: list, keys) -> np.ndarray | None:
    if not isinstance(records, list) or len(records) != N:
        return None
    for key in keys:
        if all(isinstance(r, dict) and key in r and r[key] is not None for r in records):
            try:
                return np.asarray([float(r[key]) for r in records], dtype=np.float64)
            except (TypeError, ValueError):
                continue
    return None


def load_token_logprob() -> np.ndarray | None:
    """Return a (500,) per-problem mean answer-span log-prob, or None if unavailable."""
    # 1) NPZ candidates.
    for path, keys in LOGPROB_NPZ_CANDIDATES:
        if not path.exists():
            continue
        try:
            blob = np.load(path, allow_pickle=False)
        except Exception:
            continue
        for key in keys:
            if key in blob.files:
                arr = np.asarray(blob[key], dtype=np.float64).ravel()
                if arr.shape == (N,):
                    return arr

    # 2) JSON candidates: list-of-records or dict-with-list.
    for path in LOGPROB_JSON_CANDIDATES:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, list):
            arr = _extract_from_records(data, JSON_KEYS)
            if arr is not None:
                return arr
        elif isinstance(data, dict):
            for key in JSON_KEYS:
                v = data.get(key)
                if isinstance(v, list) and len(v) == N:
                    try:
                        return np.asarray([float(x) for x in v], dtype=np.float64)
                    except (TypeError, ValueError):
                        pass
            for container_key in ("problems", "results", "records", "per_problem"):
                recs = data.get(container_key)
                if isinstance(recs, list):
                    arr = _extract_from_records(recs, JSON_KEYS)
                    if arr is not None:
                        return arr

    # 3) Per-problem shards problem_000.json … problem_499.json.
    if K1_SHARD_DIR.is_dir():
        vals = np.empty(N, dtype=np.float64)
        ok = True
        for i in range(N):
            shard = K1_SHARD_DIR / f"problem_{i:03d}.json"
            if not shard.exists():
                ok = False
                break
            try:
                rec = json.loads(shard.read_text())
            except Exception:
                ok = False
                break
            found = False
            for key in JSON_KEYS:
                if isinstance(rec, dict) and rec.get(key) is not None:
                    try:
                        vals[i] = float(rec[key])
                        found = True
                        break
                    except (TypeError, ValueError):
                        pass
            if not found:
                ok = False
                break
        if ok:
            return vals

    return None


def oof_dom_auroc(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    """OOF difference-of-means projection scores on raw features X."""
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, ytr = X[train_mask], y[train_mask]
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        scores[test_idx] = X[test_idx] @ d_vec
    return scores


def oof_residual_dom_auroc(X: np.ndarray, y: np.ndarray, z: np.ndarray,
                           folds: list[np.ndarray]) -> np.ndarray:
    """OOF DoM after partialling token-prob z out of every prefill column.

    Per fold: fit per-column linear regression of X on [z, 1] on the train rows,
    residualize both train and test columns with the train coefficients, refit the
    DoM direction on residualized train, project residualized test.
    """
    n, d = X.shape
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        ztr = z[train_mask]
        ytr = y[train_mask]
        A = np.column_stack([ztr, np.ones(len(ztr))])          # (n_tr, 2)
        coeffs, _, _, _ = np.linalg.lstsq(A, X[train_mask], rcond=None)  # (2, d)
        slope, intercept = coeffs[0], coeffs[1]                 # each (d,)
        res_tr = X[train_mask] - (np.outer(ztr, slope) + intercept[None, :])
        res_te = X[test_idx] - (np.outer(z[test_idx], slope) + intercept[None, :])
        d_vec = res_tr[ytr].mean(axis=0) - res_tr[~ytr].mean(axis=0)
        scores[test_idx] = res_te @ d_vec
    return scores


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float64)
    y = cache["correct"].astype(bool)
    assert X.shape == (N, 1536) and y.shape == (N,), (X.shape, y.shape)

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert dom_score.shape == (N,)

    z = load_token_logprob()
    if z is None:
        print("MISSING_REGEN_INPUT token-prob logprob cache "
              "(searched npz/json/shard candidates)", file=sys.stderr)
        return 2
    if not np.all(np.isfinite(z)):
        print("MISSING_REGEN_INPUT token-prob contains non-finite values",
              file=sys.stderr)
        return 2

    folds = stratified_kfold(y, N_FOLDS, SEED)

    # 1) Token-prob alone. Single monotone feature → logistic AUROC == raw AUROC.
    auroc_token_prob = auroc(z, y)

    # 2) Prefill DoM — precomputed cached score and a fresh OOF DoM for parity.
    auroc_dom_cached = auroc(dom_score, y)
    dom_oof = oof_dom_auroc(X, y, folds)
    auroc_dom_oof = auroc(dom_oof, y)

    # 3) Residual prefill-DoM after partialling out token-prob.
    resid_oof = oof_residual_dom_auroc(X, y, z, folds)
    auroc_resid_dom_oof = auroc(resid_oof, y)

    # Diagnostics: how correlated are token-prob and the cached DoM score?
    corr_tokenprob_dom = float(np.corrcoef(z, dom_score)[0, 1])

    delta_vs_token = auroc_resid_dom_oof - auroc_token_prob
    drop_from_residualizing = auroc_dom_oof - auroc_resid_dom_oof

    out = {
        "experiment": "P11-FE355",
        "n": int(N),
        "n_correct": int(y.sum()),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "auroc_token_prob_alone": auroc_token_prob,
        "auroc_prefill_dom_cached": auroc_dom_cached,
        "auroc_prefill_dom_oof": auroc_dom_oof,
        "auroc_residual_dom_after_tokenprob_oof": auroc_resid_dom_oof,
        "f2_reference_auroc": 0.7731,
        "corr_tokenprob_vs_dom": corr_tokenprob_dom,
        "delta_residual_dom_minus_tokenprob": delta_vs_token,
        "drop_from_residualizing_tokenprob": drop_from_residualizing,
        "residual_survives_0p7": bool(auroc_residual_dom_oof_gt(auroc_resid_dom_oof)),
        "interpretation": (
            "residual prefill-DoM AUROC > 0.7 ⇒ prefill DoM carries information not "
            "explained by token-level probability (Pawitan & Holmes framing holds for "
            "our internal signal); collapse toward 0.5 ⇒ logprob re-skin"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in (
        "auroc_token_prob_alone",
        "auroc_prefill_dom_oof",
        "auroc_residual_dom_after_tokenprob_oof",
    )}, indent=2))
    return 0


def auroc_residual_dom_oof_gt(value: float, thresh: float = 0.70) -> bool:
    return bool(np.isfinite(value) and value > thresh)


if __name__ == "__main__":
    raise SystemExit(main())