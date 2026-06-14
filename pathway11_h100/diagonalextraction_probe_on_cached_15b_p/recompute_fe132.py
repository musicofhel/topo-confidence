"""P11-FE132 — Deng et al. (2311.01460) diagonal-extraction probe vs F-2.

Replicates the "diagonal indexing" feature-extraction scheme from Deng et al.
(Implicit CoT via knowledge distillation, arXiv 2311.01460) on cached
Qwen-2.5-1.5B prefill activations. For a problem with prefill length T over
L layers, the diagonal picks one token position per layer:

    t_l = floor(1 + Delta * (l - 1)),  Delta = (T - 1) / (L - 1)

The per-layer diagonal hidden states are concatenated into a single
(L * 1536) feature, standardized to zero-mean / unit-std (the paper's
normalization), and fed to an OOF 5-fold logistic correctness probe.
We compare the resulting AUROC against F-2's L19-only DoM ceiling (0.7731).

Decision rule (from the FE brief): the diagonal trajectory carries
non-redundant cross-layer information — and F-9's redundancy claim is
*refuted* — if the diagonal AUROC beats the L19-only baseline by at least
+0.02.

This requires a multi-layer, per-token prefill cache (all 28 layers, every
prefill token). The single-layer L19 cache (m15b_prefill.npz) is used only
for labels and seq_len. If the multi-layer cache is absent the script reports
MISSING_REGEN_INPUT and exits 2 (it cannot be reconstructed from L19 alone).
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
# Multi-layer, per-token prefill cache: hidden (500, n_layers, max_seq, 1536),
# seq_len (500,). Not part of the standard manifest — must be regenerated.
ALL_LAYERS = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_alllayers.npz"
OUT_JSON = ROOT / "pathway11_h100/diagonal_extraction/results.json"

SEED = 9999
N_FOLDS = 5
HIDDEN_DIM = 1536
N_PROBLEMS = 500

# F-2 reference: L19-only prefill DoM AUROC (OOF 5-fold).
F2_L19_AUROC = 0.7731
# Refute F-9's redundancy claim if the diagonal lifts AUROC by at least this.
REFUTE_THRESHOLD = 0.02


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


def diagonal_indices(seq_len: int, n_layers: int) -> np.ndarray:
    """Deng et al. diagonal token positions (0-based), one per layer."""
    T = int(max(seq_len, 1))
    L = int(n_layers)
    if L <= 1:
        return np.zeros(L, dtype=np.int64)
    delta = (T - 1) / (L - 1)
    layers = np.arange(L, dtype=np.float64)  # l - 1 for l = 1..L
    t_l = np.floor(1.0 + delta * layers).astype(np.int64)  # 1-based positions
    t_l = np.clip(t_l, 1, T)
    return t_l - 1  # 0-based


def build_diagonal_features(hidden: np.ndarray, seq_len: np.ndarray) -> np.ndarray:
    """hidden: (N, L, max_seq, H) -> diagonal features (N, L*H)."""
    n, n_layers, _, h = hidden.shape
    feats = np.zeros((n, n_layers * h), dtype=np.float64)
    for i in range(n):
        idx = diagonal_indices(int(seq_len[i]), n_layers)
        # hidden[i, l, idx[l], :] for each layer
        rows = hidden[i, np.arange(n_layers), idx, :].astype(np.float64)  # (L, H)
        feats[i] = rows.reshape(-1)
    return feats


def fit_logistic(X: np.ndarray, y: np.ndarray, C: float = 1e-2,
                 max_iter: int = 2000) -> "object":
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    clf.fit(X, y.astype(int))
    return clf


def oof_probe(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OOF logistic probe with per-fold zero-mean unit-std normalization."""
    folds = stratified_kfold(y, N_FOLDS, SEED)
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd < 1e-8] = 1.0
        Xtr_n = (Xtr - mu) / sd
        Xte_n = (Xte - mu) / sd
        clf = fit_logistic(Xtr_n, ytr)
        scores[test_idx] = clf.decision_function(Xte_n)
    return scores


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not ALL_LAYERS.exists():
        print("MISSING_REGEN_INPUT", ALL_LAYERS, file=sys.stderr); return 2

    base = np.load(CACHE)
    y = base["correct"].astype(bool)
    assert y.shape == (N_PROBLEMS,), f"unexpected label shape {y.shape}"

    blob = np.load(ALL_LAYERS)
    if "hidden" not in blob:
        print("MISSING_REGEN_INPUT", "hidden key", ALL_LAYERS, file=sys.stderr)
        return 2
    hidden = blob["hidden"]
    if "seq_len" in blob:
        seq_len = blob["seq_len"].astype(np.int64)
    else:
        seq_len = base["seq_len"].astype(np.int64)

    if hidden.ndim != 4 or hidden.shape[0] != N_PROBLEMS or hidden.shape[3] != HIDDEN_DIM:
        print("MISSING_REGEN_INPUT", "bad hidden shape", hidden.shape, file=sys.stderr)
        return 2
    n_layers = int(hidden.shape[1])

    # Diagonal feature (all layers).
    X_diag = build_diagonal_features(hidden, seq_len)
    diag_scores = oof_probe(X_diag, y)
    auroc_diag = auroc(diag_scores, y)

    # L19-only single-layer probe on the same diagonal cache (control), using
    # the standard L19 prefill vector for a like-for-like baseline.
    X_l19 = base["prefill"].astype(np.float64)
    l19_scores = oof_probe(X_l19, y)
    auroc_l19 = auroc(l19_scores, y)

    delta_vs_f2 = float(auroc_diag - F2_L19_AUROC)
    delta_vs_l19_probe = float(auroc_diag - auroc_l19)
    refutes_f9 = bool(delta_vs_f2 >= REFUTE_THRESHOLD)

    out = {
        "experiment": "P11-FE132",
        "method": "Deng 2311.01460 diagonal extraction across layers",
        "n_layers": n_layers,
        "n_problems": int(N_PROBLEMS),
        "feature_dim": int(X_diag.shape[1]),
        "auroc_diagonal_oof": float(auroc_diag),
        "auroc_l19_probe_oof": float(auroc_l19),
        "f2_l19_reference": F2_L19_AUROC,
        "delta_auroc_vs_f2": delta_vs_f2,
        "delta_auroc_vs_l19_probe": delta_vs_l19_probe,
        "refute_threshold": REFUTE_THRESHOLD,
        "refutes_f9_redundancy": refutes_f9,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())