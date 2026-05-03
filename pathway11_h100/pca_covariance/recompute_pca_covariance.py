"""PCA-on-prefill-covariance ablation.

Question: is the L19 prefill DoM (AUROC 0.7731 OOF) just the dominant
covariance principal component, or does it recover supervised
correctness-predictive structure that is *not* the dominant variance
direction?

Method: on cached 500x1536 fp16 prefill at L19 + correct labels:
  1. center, eigendecompose covariance, take top-k=10 PCs
  2. for each PC, 5-fold StratifiedKFold OOF logistic AUROC of correct on
     projection (single-feature probe)
  3. recompute DoM = mu_correct - mu_incorrect under the same 5-fold
     protocol (train DoM on train fold, project test fold, fit 1-feature
     logistic, OOF AUROC)
  4. decompose DoM in PC basis: c_i = <DoM_full, v_i>; report cumulative
     DoM-energy / cumulative variance share

Emits results.json + Tier-1 regen-readback key=value prints to stdout.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).parent
CACHE = HERE.parent / "prefill_inversion" / "cache" / "m15b_prefill.npz"
OUT = HERE / "results.json"

K_PC = 10
N_FOLDS = 5
SEED = 0


def oof_single_feature_auroc(x, y, seed=SEED):
    """5-fold OOF AUROC for a single feature x against binary y."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    proba = np.zeros_like(y, dtype=float)
    for tr, te in skf.split(x.reshape(-1, 1), y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x[tr].reshape(-1, 1), y[tr])
        proba[te] = clf.predict_proba(x[te].reshape(-1, 1))[:, 1]
    return float(roc_auc_score(y, proba))


def main():
    d = np.load(CACHE)
    X = d["prefill"].astype(np.float64)  # 500 x 1536
    y = d["correct"].astype(int)
    n, p = X.shape
    print(f"loaded prefill X={X.shape} dtype=fp64 (cast from fp16) y_correct_rate={y.mean():.4f}", file=sys.stderr)

    mu = X.mean(axis=0)
    Xc = X - mu

    # Eigendecompose centered covariance via SVD on Xc (numerically stable).
    # Xc = U S Vt, cov = Vt^T diag(S^2/(n-1)) Vt; columns of V are eigenvectors.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals_full = (S ** 2) / (n - 1)
    V = Vt.T  # p x rank, eigenvectors as columns
    rank = V.shape[1]

    total_var = float(eigvals_full.sum())
    pc_var_share = (eigvals_full / total_var).tolist()

    # PC projection AUROCs (single-feature probe, 5-fold OOF).
    pc_aurocs = []
    for k in range(K_PC):
        proj = Xc @ V[:, k]
        au = oof_single_feature_auroc(proj, y)
        pc_aurocs.append(au)
        print(f"pc_{k+1}_auroc={au:.10f}", file=sys.stderr)

    # OOF DoM AUROC under matched 5-fold protocol.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    dom_proba = np.zeros(n, dtype=float)
    for tr, te in skf.split(X, y):
        mu_c = X[tr][y[tr] == 1].mean(axis=0)
        mu_i = X[tr][y[tr] == 0].mean(axis=0)
        dom = mu_c - mu_i
        dom = dom / (np.linalg.norm(dom) + 1e-12)
        proj = X[te] @ dom
        clf = LogisticRegression(max_iter=1000)
        # train logistic on train-fold projection
        train_proj = X[tr] @ dom
        clf.fit(train_proj.reshape(-1, 1), y[tr])
        dom_proba[te] = clf.predict_proba(proj.reshape(-1, 1))[:, 1]
    dom_oof_auroc = float(roc_auc_score(y, dom_proba))

    # Full-data DoM (no folding) for the basis-decomposition diagnostic.
    mu_c_full = X[y == 1].mean(axis=0)
    mu_i_full = X[y == 0].mean(axis=0)
    dom_full = mu_c_full - mu_i_full
    dom_full_unit = dom_full / (np.linalg.norm(dom_full) + 1e-12)

    # CAST PCA-PC1 protocol (Lee et al. 2024, ICLR'25): class-mean-of-means
    # centering instead of global-mean. Stack correct then incorrect rows,
    # subtract μ_class = (μ_+ + μ_-)/2 from every row, take PC1.
    mu_class = 0.5 * (mu_c_full + mu_i_full)
    Xcast = X - mu_class
    Ucast, Scast, Vcastt = np.linalg.svd(Xcast, full_matrices=False)
    Vcast = Vcastt.T
    cast_pc1 = Vcast[:, 0]
    # Sign-fix so projection has positive correlation with correct.
    if (Xcast @ cast_pc1)[y == 1].mean() < (Xcast @ cast_pc1)[y == 0].mean():
        cast_pc1 = -cast_pc1
    cast_proj = Xcast @ cast_pc1
    cast_pc1_auroc = oof_single_feature_auroc(cast_proj, y)
    cast_pc1_dom_cosine = float(cast_pc1 @ dom_full_unit)
    cast_pc1_global_pc1_cosine = float(cast_pc1 @ V[:, 0])
    print(f"cast_pc1_auroc={cast_pc1_auroc:.10f}", file=sys.stderr)
    print(f"cast_pc1_dom_cosine={cast_pc1_dom_cosine:.10f}", file=sys.stderr)

    # Two-feature [PC1, PC9] OOF logistic — does adding the secondary
    # correctness PC close the gap to supervised DoM 0.7679? If yes,
    # DoM ≈ PC1 + c9·PC9 plus noise and the basis-decomposition story
    # closes cleanly.
    proj_pc1 = Xc @ V[:, 0]
    proj_pc9 = Xc @ V[:, 8]
    two_feat = np.column_stack([proj_pc1, proj_pc9])
    skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    proba2 = np.zeros(n, dtype=float)
    for tr, te in skf2.split(two_feat, y):
        clf2 = LogisticRegression(max_iter=1000)
        clf2.fit(two_feat[tr], y[tr])
        proba2[te] = clf2.predict_proba(two_feat[te])[:, 1]
    pc1_pc9_oof_auroc = float(roc_auc_score(y, proba2))
    print(f"pc1_pc9_oof_auroc={pc1_pc9_oof_auroc:.10f}", file=sys.stderr)

    # Decompose DoM in PC basis: c_i = <dom_full_unit, v_i>.
    coeffs = (V.T @ dom_full_unit)  # shape (rank,)
    energy = coeffs ** 2  # sums to 1 since V is orthonormal full basis
    cumul_energy_top = []
    for k in (1, 5, 10, 25, 50, 100, 250, 500, rank):
        if k <= rank:
            cumul_energy_top.append({"k": int(k), "share": float(energy[:k].sum())})

    # Which PC carries the largest DoM coefficient?
    best_pc_for_dom = int(np.argmax(np.abs(coeffs)))
    best_pc_coeff = float(coeffs[best_pc_for_dom])
    best_pc_var_rank = best_pc_for_dom + 1  # 1-indexed PC rank

    # Best single-PC AUROC (could be lower-ranked PC).
    best_pc_auroc = max(pc_aurocs)
    best_pc_auroc_idx = int(np.argmax(pc_aurocs))

    # Cosine sim between DoM and top PC.
    cos_dom_pc1 = float(V[:, 0] @ dom_full_unit)

    out = {
        "schema_version": 1,
        "n_problems": int(n),
        "p_features": int(p),
        "k_pc_evaluated": K_PC,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "correct_rate": float(y.mean()),
        "total_variance": total_var,
        "pc_aurocs_top10": pc_aurocs,
        "pc_var_share_top10": pc_var_share[:K_PC],
        "best_pc_auroc": best_pc_auroc,
        "best_pc_auroc_index": best_pc_auroc_idx + 1,  # 1-indexed
        "dom_oof_auroc": dom_oof_auroc,
        "dom_pc1_cosine": cos_dom_pc1,
        "dom_basis_coeffs_top10": coeffs[:K_PC].tolist(),
        "dom_basis_energy_top10": energy[:K_PC].tolist(),
        "dom_cumulative_energy": cumul_energy_top,
        "dom_largest_coeff_pc_rank": best_pc_var_rank,
        "dom_largest_coeff_value": best_pc_coeff,
        "dom_energy_top10_share": float(energy[:10].sum()),
        "cast_pc1_auroc": cast_pc1_auroc,
        "cast_pc1_dom_cosine": cast_pc1_dom_cosine,
        "cast_pc1_global_pc1_cosine": cast_pc1_global_pc1_cosine,
        "pc1_pc9_oof_auroc": pc1_pc9_oof_auroc,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}", file=sys.stderr)

    # Tier-1 regen-readback (key=value lines on stdout).
    print(f"pca.dom_oof_auroc={dom_oof_auroc:.10f}")
    print(f"pca.best_pc_auroc={best_pc_auroc:.10f}")
    print(f"pca.best_pc_auroc_index={float(best_pc_auroc_idx + 1):.10f}")
    print(f"pca.pc1_auroc={pc_aurocs[0]:.10f}")
    print(f"pca.pc1_var_share={pc_var_share[0]:.10f}")
    print(f"pca.dom_pc1_cosine={cos_dom_pc1:.10f}")
    print(f"pca.dom_largest_coeff_pc_rank={float(best_pc_var_rank):.10f}")
    print(f"pca.dom_energy_top10_share={float(energy[:10].sum()):.10f}")
    print(f"pca.cast_pc1_auroc={cast_pc1_auroc:.10f}")
    print(f"pca.cast_pc1_dom_cosine={cast_pc1_dom_cosine:.10f}")
    print(f"pca.pc1_pc9_oof_auroc={pc1_pc9_oof_auroc:.10f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
