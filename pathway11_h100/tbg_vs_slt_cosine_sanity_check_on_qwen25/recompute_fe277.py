"""P11-FE277 — TBG vs SLT cosine sanity check on Qwen-2.5-1.5B.

Kossen et al. report similar AUROC at TBG (token-before-generation = our last
input/prefill token) and SLT (second-to-last / last generated answer token =
our final-token position) for semantic-entropy prediction, which is suggestive
of overlapping read directions. Our F-3 reports cos(prefill_DoM, final_DoM)
≈ 0.046 for *correctness* — near-orthogonal. This script trains two
accuracy-supervised readouts (DoM mean-difference + L2 logistic-regression
probe) at the TBG position and the SLT position and measures the cosine between
the resulting weight vectors.

Falsifiable seam: if cos(w_TBG, w_SLT) > 0.5, F-3 orthogonality would be a
property of residual-stream geometry generally rather than correctness-specific,
and the divergence from Kossen would collapse. If cos stays near 0, F-3 holds
and the SE/correctness directions genuinely differ.

Inputs:
  - prefill (TBG) activations: prefill_inversion/cache/m15b_prefill.npz["prefill"]
  - final-token (SLT) activations: a sibling final-token cache. The schema does
    not guarantee its presence ("need SLT activations extracted; have prefill"),
    so we probe a small set of candidate paths/keys and exit MISSING_REGEN_INPUT
    if none resolve.
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
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"

# The SLT (final-token) activation cache is not part of the guaranteed schema.
# Probe a few plausible sibling locations / array keys.
FINAL_CANDIDATES = [
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz",
     ("final", "final_token", "slt", "hidden", "prefill")),
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final_token.npz",
     ("final", "final_token", "slt", "hidden")),
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_slt.npz",
     ("slt", "final", "final_token", "hidden")),
    (ROOT / "pathway11_h100/data/m15b_final.npz",
     ("final", "final_token", "slt", "hidden")),
]

OUT_JSON = ROOT / "pathway11_h100/tbg_vs_slt_cosine_sanity_check_on_qwen25/results.json"

N_FOLDS = 5
SEED = 9999
LR_C = 0.001  # matches the L2 ceiling sweep elsewhere in the project


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def load_final() -> tuple[np.ndarray, str, str] | None:
    """Return (X_final (500,1536), path, key) for the first resolvable cache."""
    for path, keys in FINAL_CANDIDATES:
        if not path.exists():
            continue
        blob = np.load(path)
        for key in keys:
            if key in blob.files:
                arr = blob[key]
                if arr.ndim == 2 and arr.shape[1] == 1536 and arr.shape[0] == 500:
                    return arr.astype(np.float64), str(path), key
    return None


def dom_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y].mean(axis=0) - X[~y].mean(axis=0)


def lr_probe(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Full-fit L2 logistic probe on standardized features.

    Returns (w_std, w_orig, oof_auroc):
      w_std  — weight vector in standardized feature space
      w_orig — weight vector mapped back to raw activation space (w_std / std)
      oof_auroc — 5-fold out-of-fold AUROC of the probe (sanity check)
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd

    clf = LogisticRegression(C=LR_C, max_iter=5000, solver="lbfgs")
    clf.fit(Xs, y)
    w_std = clf.coef_.ravel().astype(np.float64)
    w_orig = w_std / sd  # decision score X @ w_orig + const is monotone in Xs @ w_std

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, te in skf.split(Xs, y):
        m = X[tr].mean(axis=0); s = X[tr].std(axis=0); s[s < 1e-12] = 1.0
        cfit = LogisticRegression(C=LR_C, max_iter=5000, solver="lbfgs")
        cfit.fit((X[tr] - m) / s, y[tr])
        oof[te] = cfit.decision_function((X[te] - m) / s)
    return w_std, w_orig, float(auroc(oof, y))


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    X_tbg = cache["prefill"].astype(np.float64)
    y = cache["correct"].astype(bool)
    assert X_tbg.shape == (500, 1536) and y.shape == (500,)

    final = load_final()
    if final is None:
        print("MISSING_REGEN_INPUT",
              "no SLT/final-token activation cache found among",
              [str(p) for p, _ in FINAL_CANDIDATES],
              file=sys.stderr)
        return 2
    X_slt, final_path, final_key = final

    # --- DoM (mean-difference) directions, raw activation space (F-3 comparable) ---
    d_tbg = dom_direction(X_tbg, y)
    d_slt = dom_direction(X_slt, y)
    cos_dom = cosine(d_tbg, d_slt)

    # --- L2 logistic-regression probes ---
    w_tbg_std, w_tbg_orig, auroc_tbg = lr_probe(X_tbg, y, SEED)
    w_slt_std, w_slt_orig, auroc_slt = lr_probe(X_slt, y, SEED)
    cos_lr_std = cosine(w_tbg_std, w_slt_std)
    cos_lr_orig = cosine(w_tbg_orig, w_slt_orig)

    # Headline cosine: the LR probe in standardized space is the closest analogue
    # to Kossen's supervised SE readouts; DoM-raw is the direct F-3 comparison.
    cos_headline = cos_lr_std
    threshold = 0.5
    f3_holds = abs(cos_headline) <= threshold and abs(cos_dom) <= threshold
    verdict = (
        "F3_HOLDS_correctness_specific" if f3_holds
        else "F3_REFUTED_overlapping_directions"
    )

    out = {
        "experiment": "P11-FE277",
        "description": "TBG vs SLT cosine sanity check (correctness probes)",
        "n": int(len(y)),
        "n_correct": int(y.sum()),
        "final_cache_path": final_path,
        "final_cache_key": final_key,
        "threshold": threshold,
        "cos_dom_raw_tbg_slt": cos_dom,
        "cos_lr_standardized_tbg_slt": cos_lr_std,
        "cos_lr_origspace_tbg_slt": cos_lr_orig,
        "cos_headline": cos_headline,
        "auroc_oof_tbg_dom": float(auroc(X_tbg @ d_tbg, y)),
        "auroc_oof_slt_dom": float(auroc(X_slt @ d_slt, y)),
        "auroc_oof_tbg_lr": auroc_tbg,
        "auroc_oof_slt_lr": auroc_slt,
        "f3_reference_cos_prefill_final_dom": 0.046,
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())