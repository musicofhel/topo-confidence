"""P11-FE246 — Neural-collapse battery on cached prefill / final-token activations.

Pre-registered measurement of all four Neural Collapse (NC) properties on the
pathway11_h100 caches, re-implementing the canonical metrics from Wu & Papyan's
`neural-collapse` package (footnote 3, github.com/rhubarbwu/neural-collapse) in
pure numpy (the package itself is a torch dependency and is banned in this
harness; the definitions below are faithful to its measure/kernels modules):

  - CDNV                  (within-class variability / class-distance norm. var.)
  - Equinormness          (NC2: coefficient of variation of class-mean norms)
  - Equiangularity        (NC2: off-diagonal cosines vs the -1/(C-1) ETF target,
                           reported as mean/std + CoV-interference)
  - Hyperspherical unif.  (logarithmic inverse-distance kernel over unit means)
  - Self-duality          (NC3: normalized Frobenius distance between the learned
                           linear classifier W and the centered class means M)
  - Uniform duality       (Frobenius distance of the W-gram from the ideal
                           simplex-ETF gram (I - J/C)/sqrt(C-1))
  - NCC agreement         (NC4: fraction of points where argmax(Wx+b) matches the
                           nearest-class-center assignment)
  - Simplex-ETF error     (same target applied to the M-gram, uniform-duality of
                           the means themselves)

Two partitionings of the same activations:
  - per-correctness-class : 2 classes {correct, incorrect}  (F-2 / F-4 setup)
  - per-bucket A/B/C/D     : 4 DoM-score quartile buckets    (F-7 setup)

One run yields anchors for FE7767/FE7768/FE7769 and a column comparable to
Wu & Papyan's TinyStories numbers. Results are written as a single table.
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
FINAL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/neural_collapse_battery/results.json"

EPS = 1e-12


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def fit_linear(X: np.ndarray, labels: np.ndarray, classes: np.ndarray):
    """Least-squares one-vs-rest linear classifier -> (W: C x d, b: C)."""
    Y = np.zeros((len(labels), len(classes)), dtype=np.float64)
    for i, c in enumerate(classes):
        Y[labels == c, i] = 1.0
    Xb = np.column_stack([X, np.ones(len(X), dtype=np.float64)])
    Wb, *_ = np.linalg.lstsq(Xb, Y, rcond=None)  # (d+1) x C
    return Wb[:-1].T, Wb[-1]  # W (C x d), b (C,)


def nc_battery(X: np.ndarray, labels: np.ndarray) -> dict:
    """Full NC battery for one partitioning of X by integer-coded `labels`."""
    classes = np.unique(labels)
    C = int(len(classes))
    n, d = X.shape

    mu_g = X.mean(axis=0)
    means = np.stack([X[labels == c].mean(axis=0) for c in classes])  # C x d
    counts = np.array([int((labels == c).sum()) for c in classes])

    # within-class variability: mean ||x - mu_c||^2 per class
    var_norms = np.array(
        [float(np.mean(np.sum((X[labels == c] - means[i]) ** 2, axis=1)))
         for i, c in enumerate(classes)]
    )

    # CDNV averaged over unordered class pairs
    cdnvs = []
    for a in range(C):
        for b in range(a + 1, C):
            den = 2.0 * float(np.sum((means[a] - means[b]) ** 2))
            cdnvs.append((var_norms[a] + var_norms[b]) / den if den > EPS else np.nan)
    cdnv = float(np.nanmean(cdnvs)) if cdnvs else float("nan")

    # centered class means
    M = means - mu_g  # C x d
    mnorms = np.linalg.norm(M, axis=1)

    # NC2 equinormness: coefficient of variation of mean norms
    equinorm_cov = float(np.std(mnorms) / (np.mean(mnorms) + EPS))

    # NC2 equiangularity (off-diagonal cosines vs ETF target -1/(C-1))
    if C > 1:
        Mn = M / (mnorms[:, None] + EPS)
        G = Mn @ Mn.T
        offdiag = G[~np.eye(C, dtype=bool)]
        target = -1.0 / (C - 1)
        cos_mean = float(np.mean(offdiag))
        cos_std = float(np.std(offdiag))
        equiangular_dev = float(np.mean(np.abs(offdiag - target)))
        cov_interference = float(cos_std / (abs(cos_mean) + EPS))

        # hyperspherical uniformity: log inverse-distance kernel over unit means
        uni = [-np.log(np.linalg.norm(Mn[i] - Mn[j]) + EPS)
               for i in range(C) for j in range(C) if i != j]
        hyper_uniformity = float(np.mean(uni))

        # simplex-ETF error of the M-gram (uniform duality of the means)
        Mf = M / (np.linalg.norm(M) + EPS)
        ideal = (np.eye(C) - np.ones((C, C)) / C) / np.sqrt(C - 1)
        etf_error = float(np.linalg.norm(Mf @ Mf.T - ideal))
    else:
        target = float("nan")
        cos_mean = cos_std = equiangular_dev = cov_interference = float("nan")
        hyper_uniformity = etf_error = float("nan")

    # NC3 self-duality + uniform duality from the learned linear classifier
    W, b = fit_linear(X, labels, classes)
    Wf = W / (np.linalg.norm(W) + EPS)
    Mf = M / (np.linalg.norm(M) + EPS)
    self_duality_err = float(np.linalg.norm(Wf - Mf))
    if C > 1:
        ideal = (np.eye(C) - np.ones((C, C)) / C) / np.sqrt(C - 1)
        uniform_duality_err = float(np.linalg.norm(Wf @ Wf.T - ideal))
    else:
        uniform_duality_err = float("nan")

    # NC4 nearest-class-center agreement with the linear classifier
    preds_clf = np.argmax(X @ W.T + b, axis=1)
    sq = np.stack([np.sum((X - means[i]) ** 2, axis=1) for i in range(C)], axis=1)
    preds_ncc = np.argmin(sq, axis=1)
    ncc_agreement = float(np.mean(preds_clf == preds_ncc))
    label_idx = np.searchsorted(classes, labels)
    ncc_accuracy = float(np.mean(preds_ncc == label_idx))

    return {
        "n_classes": C,
        "n_points": int(n),
        "class_counts": counts.tolist(),
        "cdnv": cdnv,
        "within_var_norms": var_norms.tolist(),
        "equinormness_cov": equinorm_cov,
        "mean_norms": mnorms.tolist(),
        "equiangular_target": float(target),
        "cos_offdiag_mean": cos_mean,
        "cos_offdiag_std": cos_std,
        "equiangular_abs_dev": equiangular_dev,
        "cov_interference": cov_interference,
        "hyperspherical_log_uniformity": hyper_uniformity,
        "simplex_etf_error": etf_error,
        "self_duality_error": self_duality_err,
        "uniform_duality_error": uniform_duality_err,
        "ncc_clf_agreement": ncc_agreement,
        "ncc_accuracy": ncc_accuracy,
    }


def bucket_labels(score: np.ndarray) -> np.ndarray:
    """Quartile buckets A/B/C/D (0..3) by ascending DoM score."""
    order = np.argsort(np.argsort(score, kind="stable"), kind="stable")
    q = (order * 4) // len(score)
    return np.clip(q, 0, 3).astype(int)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X_prefill = blob["prefill"].astype(np.float64)
    correct = blob["correct"].astype(bool)
    assert X_prefill.shape == (500, 1536) and correct.shape == (500,)

    # DoM scores for the A/B/C/D bucketing (F-7 setup); fall back to a
    # train-on-all DoM projection if the cached scores are absent.
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    else:
        d_vec = X_prefill[correct].mean(0) - X_prefill[~correct].mean(0)
        dom_score = X_prefill @ d_vec
    buckets = bucket_labels(dom_score)

    # correctness partition: 0 = incorrect, 1 = correct
    corr_labels = correct.astype(int)

    sources = {"prefill": X_prefill}
    if FINAL_CACHE.exists():
        fblob = np.load(FINAL_CACHE)
        fkey = next((k for k in ("final", "final_token", "hidden", "prefill")
                     if k in fblob.files), None)
        if fkey is not None:
            Xf = fblob[fkey].astype(np.float64)
            if Xf.shape == X_prefill.shape:
                sources["final"] = Xf

    table = {}
    for sname, Xs in sources.items():
        table[sname] = {
            "by_correctness": nc_battery(Xs, corr_labels),
            "by_bucket_ABCD": nc_battery(Xs, buckets),
        }

    out = {
        "experiment": "P11-FE246",
        "description": "neural-collapse battery (CDNV, equinormness, "
                       "equiangularity, hyperspherical uniformity, self/uniform "
                       "duality, NCC agreement) on prefill/final activations",
        "implementation_note": "pure-numpy reimplementation of Wu & Papyan "
                               "neural-collapse measure/kernels definitions",
        "sources": list(sources.keys()),
        "final_token_available": "final" in sources,
        "dom_score_source": "phase2_prefill_dom.npz" if DOM_NPZ.exists()
                            else "recomputed_train_on_all",
        "sanity_dom_auroc": float(auroc(dom_score, correct)),
        "table": table,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())