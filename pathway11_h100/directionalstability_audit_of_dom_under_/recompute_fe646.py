"""P11-FE646 — Directional-stability audit of the DoM direction.

Measures the noise floor of the supervised DoM direction (mean(correct) -
mean(incorrect)) under a K=20 x 20%-drop ablation protocol, on cached L19
prefill and (if present) final-token activations for Qwen2.5-1.5B. For each of
K=20 resamples we drop 20% of the 500 problems (label-stratified), refit the
DoM direction on the retained 80%, and record it. Stability = mean absolute
pairwise cosine over the 20 directions. The same protocol is applied to a
RAPTOR-style L2 logistic-regression direction on the same retained subsets, so
DoM's noise floor can be read against RAPTOR's Table-2 benchmark (0.87-0.98).

Decision rule: if DoM mean |cosine| on prefill < 0.5 (well below RAPTOR), then
F-3's cos(prefill_DoM, final_DoM)=0.046 orthogonality claim is dominated by
DoM's own sampling noise and must be re-stated.
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
# Final-token activations live under a sibling cache; key/name vary by run, so
# we probe a small set of candidates and accept the first (500, 1536) array.
FINAL_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_finaltoken.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final_token.npz",
    ROOT / "pathway11_h100/prefill_gated_compute/phase2_final_token.npz",
]
OUT_JSON = ROOT / "pathway11_h100/dom_stability_audit/results.json"

K_RESAMPLES = 20
DROP_FRAC = 0.20
RAPTOR_C = 1.0
SEED = 9999
DOM_COSINE_THRESHOLD = 0.5
RAPTOR_BENCHMARK = [0.87, 0.98]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def extract_acts(path: Path) -> np.ndarray | None:
    """Return the first (500, 1536) array in an npz, or None."""
    if not path.exists():
        return None
    blob = np.load(path)
    for key in blob.files:
        arr = blob[key]
        if arr.ndim == 2 and arr.shape == (500, 1536):
            return arr.astype(np.float64)
    return None


def stratified_keep(y: np.ndarray, drop_frac: float, rng) -> np.ndarray:
    """Indices retained after dropping drop_frac of each class."""
    keep = []
    for cls in (False, True):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_keep = max(1, int(round(len(idx) * (1.0 - drop_frac))))
        keep.append(idx[:n_keep])
    return np.concatenate(keep)


def dom_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y].mean(axis=0) - X[~y].mean(axis=0)


def raptor_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """L2 logistic-regression weight vector on standardized features."""
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return None
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd
    clf = LogisticRegression(penalty="l2", C=RAPTOR_C, max_iter=2000,
                             solver="lbfgs")
    clf.fit(Xs, y.astype(int))
    return clf.coef_.ravel().astype(np.float64)


def mean_abs_pairwise_cosine(vecs: list[np.ndarray]) -> dict:
    M = np.array([v / (np.linalg.norm(v) + 1e-12) for v in vecs])
    G = M @ M.T
    iu, ju = np.triu_indices(len(M), k=1)
    pair = np.abs(G[iu, ju])
    return {
        "n_directions": int(len(M)),
        "n_pairs": int(len(pair)),
        "mean_abs_cosine": float(pair.mean()),
        "min_abs_cosine": float(pair.min()),
        "max_abs_cosine": float(pair.max()),
        "std_abs_cosine": float(pair.std()),
    }


def audit_representation(X: np.ndarray, y: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    dom_vecs, raptor_vecs = [], []
    raptor_ok = True
    for _ in range(K_RESAMPLES):
        keep = stratified_keep(y, DROP_FRAC, rng)
        Xk, yk = X[keep], y[keep]
        dom_vecs.append(dom_direction(Xk, yk))
        if raptor_ok:
            rv = raptor_direction(Xk, yk)
            if rv is None:
                raptor_ok = False
            else:
                raptor_vecs.append(rv)
    out = {
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "k_resamples": K_RESAMPLES,
        "drop_frac": DROP_FRAC,
        "auroc_full_dom": auroc(X @ dom_direction(X, y), y),
        "dom": mean_abs_pairwise_cosine(dom_vecs),
    }
    out["raptor"] = mean_abs_pairwise_cosine(raptor_vecs) if raptor_ok else None
    return out


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    blob = np.load(CACHE)
    Xp = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert Xp.shape == (500, 1536) and y.shape == (500,)

    out = {
        "experiment": "P11-FE646",
        "protocol": f"K={K_RESAMPLES} x {int(DROP_FRAC * 100)}%-drop, stratified",
        "raptor_benchmark": RAPTOR_BENCHMARK,
        "dom_cosine_threshold": DOM_COSINE_THRESHOLD,
        "prefill": audit_representation(Xp, y),
    }

    Xf = None
    final_path_used = None
    for cand in FINAL_CANDIDATES:
        Xf = extract_acts(cand)
        if Xf is not None:
            final_path_used = str(cand.relative_to(ROOT))
            break
    if Xf is not None:
        out["final_token"] = audit_representation(Xf, y)
        out["final_token_cache"] = final_path_used
    else:
        out["final_token"] = None
        out["final_token_cache"] = None
        out["final_token_note"] = "no (500,1536) final-token cache found among candidates"

    dom_pref = out["prefill"]["dom"]["mean_abs_cosine"]
    out["orthogonality_claim_unsafe"] = bool(dom_pref < DOM_COSINE_THRESHOLD)
    out["verdict"] = (
        "DoM prefill noise floor below threshold; F-3 cos=0.046 orthogonality "
        "claim is dominated by DoM sampling noise and must be re-stated"
        if dom_pref < DOM_COSINE_THRESHOLD
        else "DoM prefill direction stable above threshold; F-3 orthogonality "
        "claim survives its own noise floor"
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())