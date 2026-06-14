"""P11-FE1298 — DySIB residual delta-predictor flow test (F-3 recast).

F-3 measures cos(prefill_DoM, final_DoM) ~= 0.046 (0.0008 residualized): two
STATIC directions. DySIB (Eq.9-10) instead asks whether the L19 prefill cloud
and the final-token cloud are two states of one smooth latent flow. We fit
    z_final ~= z_prefill + mu_delta(z_prefill)
on a shared PCA-reduced latent space (OOF 5-fold), where mu_delta is a ridge
delta-predictor, then report:
  * InfoNCE predictive MI lower bound (nats) of the predicted vs true z_final,
    against identity (no-delta) and mean-delta baselines;
  * the R^2 of the delta-predictor (smoothness / predictability proxy);
  * the ||mu_delta|| distribution.
High MI + small, smooth (high-R^2) delta => prefill/final are one dynamical
manifold, recasting F-3's near-orthogonality as a flow rather than independence.
This is a pure structural measurement of the prefill->final transformation,
NOT a correctness-beats-baseline claim.
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
from scipy.special import logsumexp

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/dysib_flow/results.json"

N_FOLDS = 5
SEED = 9999
PCA_DIM = 64
RIDGE_REL = 1e-2          # ridge penalty relative to feature trace
TAU_GRID = [0.05, 0.1, 0.2, 0.5, 1.0]


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


def load_final() -> np.ndarray | None:
    """Locate the (500,1536) final-token L19 cloud across likely cache layouts."""
    candidates = [
        (CACHE, ["final", "final_token", "final_hidden", "z_final", "final_states"]),
        (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz",
         ["final", "final_token", "final_hidden", "hidden", "prefill", "z_final"]),
        (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_finaltoken.npz",
         ["final", "final_token", "hidden", "prefill"]),
        (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final_token.npz",
         ["final", "final_token", "hidden", "prefill"]),
    ]
    for path, keys in candidates:
        if not path.exists():
            continue
        blob = np.load(path)
        for k in keys:
            if k in blob.files and tuple(blob[k].shape) == (500, 1536):
                return blob[k].astype(np.float64)
    return None


def fit_pca(X: np.ndarray, dim: int):
    mu = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:dim]


def fit_ridge(X: np.ndarray, Y: np.ndarray, rel: float):
    """Closed-form ridge with bias; penalty scaled to feature trace."""
    Xa = np.column_stack([X, np.ones(len(X))])
    d = Xa.shape[1]
    G = Xa.T @ Xa
    lam = rel * float(np.trace(G[:-1, :-1])) / max(d - 1, 1)
    reg = lam * np.eye(d)
    reg[-1, -1] = 0.0  # do not penalize bias
    W = np.linalg.solve(G + reg, Xa.T @ Y)
    return W


def predict_ridge(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.column_stack([X, np.ones(len(X))]) @ W


def infonce_mi(z_hat: np.ndarray, z_tgt: np.ndarray, tau: float) -> float:
    """InfoNCE lower bound on MI (nats): log(N) - L_NCE, cosine critic / tau."""
    a = z_hat / (np.linalg.norm(z_hat, axis=1, keepdims=True) + 1e-12)
    b = z_tgt / (np.linalg.norm(z_tgt, axis=1, keepdims=True) + 1e-12)
    n = len(a)
    S = (a @ b.T) / tau                       # (n, n)
    log_p = np.diag(S) - logsumexp(S, axis=1)
    return float(np.log(n) - (-log_p.mean()))


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    blob = np.load(CACHE)
    prefill = blob["prefill"].astype(np.float64)
    correct = blob["correct"].astype(bool)
    assert prefill.shape == (500, 1536) and correct.shape == (500,)

    final = load_final()
    if final is None:
        print("MISSING_REGEN_INPUT", "final-token L19 cloud (m15b_final.npz)",
              file=sys.stderr)
        return 2

    folds = stratified_kfold(correct, N_FOLDS, SEED)
    n = len(correct)

    # Per-fold containers (PCA bases differ per fold -> InfoNCE is computed
    # within each fold's own basis, then averaged).
    fold_pairs = []           # list of (z_hat_delta, z_hat_id, z_hat_mean, z_final)
    delta_r2_folds = []
    cos_pf_test = []          # cos(z_prefill, z_final) in PCA space, test points
    mu_delta_norms = []       # ||predicted delta|| over test points

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        tr, te = train_mask, test_idx

        combined = np.vstack([prefill[tr], final[tr]])
        mu, comps = fit_pca(combined, PCA_DIM)

        zp_tr = (prefill[tr] - mu) @ comps.T
        zf_tr = (final[tr] - mu) @ comps.T
        zp_te = (prefill[te] - mu) @ comps.T
        zf_te = (final[te] - mu) @ comps.T

        delta_tr = zf_tr - zp_tr
        mean_delta = delta_tr.mean(axis=0)

        W = fit_ridge(zp_tr, delta_tr, RIDGE_REL)
        delta_pred_te = predict_ridge(zp_te, W)

        # Predicted final-state under each model.
        zhat_delta = zp_te + delta_pred_te
        zhat_id = zp_te.copy()                      # identity / no-delta flow
        zhat_mean = zp_te + mean_delta              # constant mean-delta flow
        fold_pairs.append((zhat_delta, zhat_id, zhat_mean, zf_te))

        # Smoothness proxy: out-of-fold R^2 of the delta-predictor.
        delta_te = zf_te - zp_te
        ss_res = float(((delta_te - delta_pred_te) ** 2).sum())
        ss_tot = float(((delta_te - delta_te.mean(axis=0)) ** 2).sum())
        delta_r2_folds.append(1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))

        # Static near-orthogonality sanity (F-3 recast) in PCA space.
        num = (zp_te * zf_te).sum(axis=1)
        den = (np.linalg.norm(zp_te, axis=1) * np.linalg.norm(zf_te, axis=1)) + 1e-12
        cos_pf_test.extend((num / den).tolist())

        mu_delta_norms.extend(np.linalg.norm(delta_pred_te, axis=1).tolist())

    # InfoNCE MI per tau, averaged over folds; pick the tightest (max) bound.
    def mi_for(model_idx: int) -> tuple[float, float]:
        best_mi, best_tau = -np.inf, None
        for tau in TAU_GRID:
            mis = [infonce_mi(p[model_idx], p[3], tau) for p in fold_pairs]
            m = float(np.mean(mis))
            if m > best_mi:
                best_mi, best_tau = m, tau
        return best_mi, best_tau

    mi_delta, tau_delta = mi_for(0)
    mi_id, tau_id = mi_for(1)
    mi_mean, tau_mean = mi_for(2)

    mu_norms = np.asarray(mu_delta_norms)
    cos_arr = np.asarray(cos_pf_test)

    out = {
        "experiment": "P11-FE1298",
        "n": int(n),
        "pca_dim": PCA_DIM,
        "ridge_rel": RIDGE_REL,
        "tau_grid": TAU_GRID,
        "mi_infonce_delta_nats_oof": mi_delta,
        "mi_infonce_delta_tau": tau_delta,
        "mi_infonce_identity_nats_oof": mi_id,
        "mi_infonce_identity_tau": tau_id,
        "mi_infonce_meandelta_nats_oof": mi_mean,
        "mi_infonce_meandelta_tau": tau_mean,
        "mi_delta_gain_over_identity": mi_delta - mi_id,
        "mi_delta_gain_over_meandelta": mi_delta - mi_mean,
        "mi_max_possible_nats": float(np.log(n // N_FOLDS)),
        "delta_predictor_r2_oof_mean": float(np.nanmean(delta_r2_folds)),
        "delta_predictor_r2_folds": [float(x) for x in delta_r2_folds],
        "mu_delta_norm_mean": float(mu_norms.mean()),
        "mu_delta_norm_std": float(mu_norms.std()),
        "mu_delta_norm_p10": float(np.percentile(mu_norms, 10)),
        "mu_delta_norm_p50": float(np.percentile(mu_norms, 50)),
        "mu_delta_norm_p90": float(np.percentile(mu_norms, 90)),
        "cos_prefill_final_pca_mean": float(cos_arr.mean()),
        "cos_prefill_final_pca_std": float(cos_arr.std()),
        "dom_sanity_auroc": float(
            auroc(np.load(CACHE)["prefill"].astype(np.float64) @
                  (prefill[correct].mean(0) - prefill[~correct].mean(0)), correct)
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())