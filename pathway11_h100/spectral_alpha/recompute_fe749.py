"""FE749 — Spectral α (power-law SVD exponent) head-to-head with DoM.

For each (problem, layer), compute the top-K=50 singular values of the
(T_i, hidden_dim) hidden-state matrix and fit a power-law exponent
    log(σ_k) = log(c) − α · log(k)
via OLS. The exponent α is a single scalar per (problem, layer).

Per-layer 5-fold stratified OOF logistic regression with α as the only
feature → α-AUROC. Compare best-layer α-AUROC and α at L19 to F-2's
0.7731 (1.5B). Bonus: joint [α_L*, prefill_DoM_proj_L19] AUROC and a 7B run.

References: Martin & Mahoney HT-SR theory (2002.03175 et seq.).
The FE749 spec calls this the "spectral alpha single-scalar AUROC test."
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/musicofhel/topo-confidence")
NPZ_15B = ROOT / "pathway8_layerwise/data/math500"
NPZ_7B = ROOT / "pathway11_h100/data/math500_7b"
OUT_JSON = ROOT / "pathway11_h100/spectral_alpha/results.json"

K_TOP = 50
N_LAYERS = 29
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


def oof_dom_auroc_1d(x: np.ndarray, y: np.ndarray, k: int, seed: int) -> float:
    """1D DoM-via-mean-difference probe — fold-safe single-feature classifier."""
    folds = stratified_kfold(y, k, seed)
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train = np.ones(n, dtype=bool); train[test_idx] = False
        mu_pos = x[train & y].mean()
        mu_neg = x[train & ~y].mean()
        # Use sign of (mu_pos - mu_neg) to orient: if > 0, higher x ⇒ positive class.
        # Score is x − (mu_pos+mu_neg)/2, signed by the orientation.
        s = (mu_pos - mu_neg)
        scores[test_idx] = (x[test_idx] - 0.5 * (mu_pos + mu_neg)) * np.sign(s + 1e-12)
    return auroc(scores, y)


def oof_2feat_auroc(X: np.ndarray, y: np.ndarray, k: int, seed: int) -> float:
    """OOF DoM-style 2-feature probe via Σ⁻¹ (μ_pos − μ_neg) on 2 features."""
    folds = stratified_kfold(y, k, seed)
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train = np.ones(n, dtype=bool); train[test_idx] = False
        Xtr = X[train]
        d = X[train & y].mean(0) - X[train & ~y].mean(0)
        Xc = Xtr - Xtr.mean(0)
        Sigma = Xc.T @ Xc / max(len(Xtr) - 1, 1)
        Sigma += 1e-3 * np.trace(Sigma) / 2 * np.eye(2)
        w = np.linalg.solve(Sigma, d)
        scores[test_idx] = X[test_idx] @ w
    return auroc(scores, y)


def spectral_alpha(X: np.ndarray, k_top: int = K_TOP) -> float:
    """Top-k singular values, OLS slope of log σ vs log rank → −α."""
    # X is (T, H). Use full_matrices=False to get min(T,H) singular values,
    # then take top-k_top.
    sigma = np.linalg.svd(X, compute_uv=False)
    k = min(k_top, len(sigma))
    if k < 5:
        return float("nan")
    sig = sigma[:k]
    if (sig <= 0).any():
        return float("nan")
    log_k = np.log(np.arange(1, k + 1, dtype=np.float64))
    log_s = np.log(sig.astype(np.float64))
    # OLS slope.
    log_k_c = log_k - log_k.mean()
    log_s_c = log_s - log_s.mean()
    slope = (log_k_c * log_s_c).sum() / (log_k_c * log_k_c).sum()
    return float(-slope)  # α = −slope (since σ_k ∝ k^{-α})


def compute_alpha_matrix(npz_dir: Path, model_label: str) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(npz_dir.glob("problem_*.npz"))
    n = len(files)
    print(f"[{model_label}] n_files={n}")
    alphas = np.full((n, N_LAYERS), np.nan, dtype=np.float64)
    correct = np.zeros(n, dtype=bool)
    t0 = time.time()
    for i, f in enumerate(files):
        d = np.load(f)
        states = d["states"]  # (29, T_i, hidden) fp16
        correct[i] = bool(d["correct"])
        for layer in range(N_LAYERS):
            X = states[layer].astype(np.float64)  # (T_i, hidden)
            alphas[i, layer] = spectral_alpha(X, K_TOP)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"[{model_label}] {i+1}/{n}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
    return alphas, correct


def per_layer_auroc(alphas: np.ndarray, correct: np.ndarray) -> tuple[list[float], int, float]:
    """Returns (auroc_per_layer, best_layer, best_auroc)."""
    auroc_per_layer = []
    for layer in range(N_LAYERS):
        x = alphas[:, layer]
        if not np.isfinite(x).all():
            auroc_per_layer.append(float("nan"))
            continue
        a = oof_dom_auroc_1d(x, correct, N_FOLDS, SEED)
        auroc_per_layer.append(a)
    valid = [(i, a) for i, a in enumerate(auroc_per_layer) if not np.isnan(a)]
    if not valid:
        return auroc_per_layer, -1, float("nan")
    # "Best" is max distance from 0.5 (since negative slope can flip orientation).
    best_layer, best_a = max(valid, key=lambda ia: abs(ia[1] - 0.5))
    return auroc_per_layer, best_layer, best_a


def main() -> int:
    if not NPZ_15B.exists():
        print(f"MISSING_REGEN_INPUT {NPZ_15B}", file=sys.stderr); return 2

    # ---- 1.5B ----
    a_15b, y_15b = compute_alpha_matrix(NPZ_15B, "1.5B")
    auroc_15b_per_layer, best_15b, best_a_15b = per_layer_auroc(a_15b, y_15b)
    auroc_15b_l19 = auroc_15b_per_layer[19]

    # joint [α at best layer, DoM-proj on prefill at L19]: load m15b_prefill cache
    prefill_path = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
    if prefill_path.exists():
        pf = np.load(prefill_path)
        Xp = pf["prefill"].astype(np.float64)
        yp = pf["correct"].astype(bool)
        # Use the full-data DoM direction (in-sample, for projection only — used
        # alongside α as a 2-feature probe with proper OOF on the joint score).
        d_full = Xp[yp].mean(0) - Xp[~yp].mean(0)
        dom_proj = Xp @ d_full
        # Align: m15b_prefill has 500 rows in the same order as pathway8 problem_*.npz.
        if len(dom_proj) == len(y_15b) and (yp == y_15b).all():
            X_joint = np.column_stack([a_15b[:, best_15b], dom_proj])
            joint_auroc_15b = oof_2feat_auroc(X_joint, y_15b, N_FOLDS, SEED)
        else:
            joint_auroc_15b = float("nan")
            print("[1.5B] alignment mismatch between m15b_prefill and per-problem cache; "
                  "skipping joint")
    else:
        joint_auroc_15b = float("nan")

    # ---- 7B ----
    if NPZ_7B.exists() and len(list(NPZ_7B.glob("problem_*.npz"))) >= 100:
        a_7b, y_7b = compute_alpha_matrix(NPZ_7B, "7B")
        auroc_7b_per_layer, best_7b, best_a_7b = per_layer_auroc(a_7b, y_7b)
        auroc_7b_l19 = auroc_7b_per_layer[19]
    else:
        a_7b = None; y_7b = None
        auroc_7b_per_layer = [float("nan")] * N_LAYERS
        best_7b = -1; best_a_7b = float("nan"); auroc_7b_l19 = float("nan")

    out = {
        "n_15b": int(len(y_15b)),
        "n_7b": int(len(y_7b)) if y_7b is not None else 0,
        "k_top": K_TOP,
        "n_layers": N_LAYERS,
        "n_folds": N_FOLDS,
        "seed": SEED,
        # 1.5B
        "alpha_auroc_15b_per_layer": auroc_15b_per_layer,
        "alpha_best_layer_15b": int(best_15b),
        "alpha_best_auroc_15b": float(best_a_15b),
        "alpha_l19_auroc_15b": float(auroc_15b_l19),
        "joint_alpha_dom_auroc_15b": float(joint_auroc_15b),
        # 7B
        "alpha_auroc_7b_per_layer": auroc_7b_per_layer,
        "alpha_best_layer_7b": int(best_7b),
        "alpha_best_auroc_7b": float(best_a_7b),
        "alpha_l19_auroc_7b": float(auroc_7b_l19),
        # Decision
        "f9_alpha_beats_dom_threshold": 0.7731,
        "f9_alpha_beats_dom_15b": bool(best_a_15b > 0.7731),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    for k in ("alpha_best_layer_15b", "alpha_best_auroc_15b", "alpha_l19_auroc_15b",
              "joint_alpha_dom_auroc_15b", "alpha_best_layer_7b",
              "alpha_best_auroc_7b", "alpha_l19_auroc_7b"):
        v = out[k]
        if isinstance(v, float):
            print(f"{k}={v:.10f}")
        else:
            print(f"{k}={v}")
    print(f"f9_alpha_beats_dom_15b={int(out['f9_alpha_beats_dom_15b'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
