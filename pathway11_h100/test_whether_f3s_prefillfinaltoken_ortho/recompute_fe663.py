"""P11-FE663 — Does F-3's prefill/final orthogonality survive the unified preference-utility reframing?

F-3 reports cos(prefill_DoM, final_DoM) = 0.046: the L19 direction that predicts
correctness from the *prefill* token is geometrically unrelated to the one that
predicts it from the *final* token. The "unified preference-utility" framework
(paper Eq. 15) instead posits a single shared preference direction omega_p whose
projection drives a log-odds-of-correctness link at *every* sequence position;
under that model the same omega_p should explain correctness well at both
positions (their prediction: per-position pseudo-R^2 > 0.95 at both).

This script fits one shared omega_p across the pooled prefill + final-token L19
activations (Qwen-2.5-1.5B, MATH-500) via a logistic link, projects each
position's held-out activations onto it (5-fold OOF to avoid the 1536-d fit
overfitting), and reports per-position McFadden / Tjur / linear R^2. It also
reproduces the bare cos(prefill_DoM, final_DoM) for context.

Decision: if both positions land R^2 > 0.95, the shared-direction framework
absorbs F-3 (the 0.046 cosine was a per-position-centering artifact of how DoM
was built). If one position lands R^2 < 0.7, F-3's orthogonality is a genuine
geometric claim that the shared-direction model cannot fit.
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
# Final-token L19 activations live in a parallel cache; try the known-likely
# names/keys and degrade gracefully if none is present on this machine.
FINAL_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final_token.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_finaltoken.npz",
]
FINAL_KEYS = ["final", "final_token", "finaltoken", "final_hidden", "hidden"]
OUT_JSON = ROOT / "pathway11_h100/preference_utility/results.json"

N_FOLDS = 5
SEED = 9999
LOGIT_C = 1.0  # mild regularization for the shared-direction logistic fit


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


def load_final_activations() -> tuple[np.ndarray | None, str | None]:
    """Return (X_final, source_str) or (None, None) if no cache is found."""
    for path in FINAL_CANDIDATES:
        if not path.exists():
            continue
        blob = np.load(path)
        for key in FINAL_KEYS:
            if key in blob.files:
                arr = blob[key]
                if arr.ndim == 2 and arr.shape == (500, 1536):
                    return arr.astype(np.float64), f"{path.name}:{key}"
        # fall back to any (500, 1536) array in the file
        for key in blob.files:
            arr = blob[key]
            if getattr(arr, "ndim", 0) == 2 and arr.shape == (500, 1536):
                return arr.astype(np.float64), f"{path.name}:{key}"
    return None, None


def fit_shared_direction(X_stack: np.ndarray, y_stack: np.ndarray) -> np.ndarray:
    """One shared preference direction omega_p via a logistic link (Eq. 15)."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        C=LOGIT_C, max_iter=5000, solver="lbfgs", fit_intercept=True
    )
    clf.fit(X_stack, y_stack)
    w = clf.coef_.ravel().astype(np.float64)
    nrm = np.linalg.norm(w)
    return w / nrm if nrm > 0 else w


def mcfadden_r2(s: np.ndarray, y: np.ndarray) -> float:
    """McFadden pseudo-R^2 of a 1-D logistic fit of y on shared-direction score s."""
    from sklearn.linear_model import LogisticRegression

    if s.std() < 1e-12 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    clf = LogisticRegression(C=1e6, max_iter=5000, solver="lbfgs")
    clf.fit(s.reshape(-1, 1), y)
    p = np.clip(clf.predict_proba(s.reshape(-1, 1))[:, 1], 1e-12, 1 - 1e-12)
    ll_model = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    p0 = float(y.mean())
    ll_null = len(y) * (p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))
    if ll_null == 0:
        return float("nan")
    return float(1.0 - ll_model / ll_null)


def tjur_r2(s: np.ndarray, y: np.ndarray) -> float:
    """Tjur coefficient of discrimination: mean(p|y=1) - mean(p|y=0)."""
    from sklearn.linear_model import LogisticRegression

    if s.std() < 1e-12 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    clf = LogisticRegression(C=1e6, max_iter=5000, solver="lbfgs")
    clf.fit(s.reshape(-1, 1), y)
    p = clf.predict_proba(s.reshape(-1, 1))[:, 1]
    return float(p[y.astype(bool)].mean() - p[~y.astype(bool)].mean())


def linear_r2(s: np.ndarray, y: np.ndarray) -> float:
    """Plain OLS R^2 of binary y on 1-D score s (= squared Pearson correlation)."""
    if s.std() < 1e-12:
        return float("nan")
    r = np.corrcoef(s, y.astype(np.float64))[0, 1]
    return float(r * r)


def oof_shared_scores(
    X_pre: np.ndarray, X_fin: np.ndarray, y: np.ndarray, folds: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """5-fold OOF projections of each position onto the shared omega_p.

    For each fold the shared direction is fit on the pooled (prefill + final)
    *training* rows, then applied to the held-out rows of both positions.
    Centering uses training-fold means per position to avoid leakage.
    """
    n = len(y)
    s_pre = np.zeros(n, dtype=np.float64)
    s_fin = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        mu_pre = X_pre[train_mask].mean(axis=0)
        mu_fin = X_fin[train_mask].mean(axis=0)
        X_stack = np.vstack([X_pre[train_mask] - mu_pre, X_fin[train_mask] - mu_fin])
        y_stack = np.concatenate([y[train_mask], y[train_mask]])
        omega = fit_shared_direction(X_stack, y_stack)
        s_pre[test_idx] = (X_pre[test_idx] - mu_pre) @ omega
        s_fin[test_idx] = (X_fin[test_idx] - mu_fin) @ omega
    return s_pre, s_fin


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X_pre = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X_pre.shape == (500, 1536) and y.shape == (500,)

    X_fin, final_src = load_final_activations()
    if X_fin is None:
        print(
            "MISSING_REGEN_INPUT final-token L19 activations "
            f"(looked in {[str(p) for p in FINAL_CANDIDATES]})",
            file=sys.stderr,
        )
        return 2

    folds = stratified_kfold(y, N_FOLDS, SEED)

    # Per-position DoM directions + bare orthogonality (reproduces F-3's 0.046).
    dom_pre = X_pre[y].mean(0) - X_pre[~y].mean(0)
    dom_fin = X_fin[y].mean(0) - X_fin[~y].mean(0)
    cos_dom = float(
        dom_pre @ dom_fin / (np.linalg.norm(dom_pre) * np.linalg.norm(dom_fin))
    )

    # Shared preference direction, OOF per-position projections.
    s_pre, s_fin = oof_shared_scores(X_pre, X_fin, y, folds)

    yf = y.astype(np.float64)
    result = {
        "experiment": "P11-FE663",
        "final_activation_source": final_src,
        "cos_prefill_dom_final_dom": cos_dom,
        "shared_direction": {
            "prefill": {
                "mcfadden_r2_oof": mcfadden_r2(s_pre, yf),
                "tjur_r2_oof": tjur_r2(s_pre, yf),
                "linear_r2_oof": linear_r2(s_pre, yf),
                "auroc_oof": auroc(s_pre, y),
            },
            "final": {
                "mcfadden_r2_oof": mcfadden_r2(s_fin, yf),
                "tjur_r2_oof": tjur_r2(s_fin, yf),
                "linear_r2_oof": linear_r2(s_fin, yf),
                "auroc_oof": auroc(s_fin, y),
            },
        },
    }

    # Verdict: framework predicts >0.95 at both; F-3 predicts <0.7 at one.
    r2_pre = result["shared_direction"]["prefill"]["mcfadden_r2_oof"]
    r2_fin = result["shared_direction"]["final"]["mcfadden_r2_oof"]
    both_high = (r2_pre > 0.95) and (r2_fin > 0.95)
    one_low = (r2_pre < 0.70) or (r2_fin < 0.70)
    if both_high:
        verdict = "FRAMEWORK_ABSORBS_F3"
    elif one_low:
        verdict = "F3_SURVIVES"
    else:
        verdict = "INCONCLUSIVE"
    result["verdict"] = verdict

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"verdict={verdict} r2_prefill={r2_pre:.4f} r2_final={r2_fin:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())