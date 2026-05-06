"""Shared utilities for the no-compute experiment plan."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import bootstrap, norm
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE_DIR = ROOT / "nocompute" / "cache"
RESULTS_DIR = ROOT / "nocompute" / "results"
FIGS_DIR = ROOT / "nocompute" / "figs"

DATA_1P5B = ROOT / "pathway8_layerwise" / "data" / "math500"
DATA_7B = ROOT / "pathway11_h100" / "data" / "math500_7b"
DATA_BBH = ROOT / "pathway8_layerwise" / "data" / "bbh"
BBH_SUBSETS = [
    "tracking_shuffled_objects_seven_objects",
    "logical_deduction_seven_objects",
    "web_of_lies",
]

N_LAYERS = 29
HIDDEN_1P5B = 1536
HIDDEN_7B = 3584
STEERING_LAYER = 19
SEED = 9999
N_FOLDS = 5


def ensure_dirs():
    for d in [CACHE_DIR, RESULTS_DIR, FIGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default))


def load_cache(name: str) -> dict:
    path = CACHE_DIR / f"{name}.npz"
    if not path.exists():
        sys.exit(f"Cache not found: {path} — run A1_cache.py first")
    return dict(np.load(path, allow_pickle=True))


def load_manifest(data_dir: Path) -> dict:
    return json.loads((data_dir / "manifest.json").read_text())


def parse_subject(unique_id: str) -> str:
    parts = unique_id.split("/")
    return parts[1] if len(parts) >= 2 else "unknown"


# ---------------------------------------------------------------------------
# DomProbe: difference-of-means direction with optional isotonic calibration
# ---------------------------------------------------------------------------

class DomProbe:
    def __init__(self):
        self.w_unit = None
        self.mu_center = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> DomProbe:
        mu_c = X[y].mean(axis=0)
        mu_i = X[~y].mean(axis=0)
        w = mu_c - mu_i
        self.w_unit = w / (np.linalg.norm(w) + 1e-12)
        self.mu_center = 0.5 * (mu_c + mu_i)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu_center) @ self.w_unit

    @property
    def direction(self) -> np.ndarray:
        return self.w_unit


def oof_dom_scores(X: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS,
                   seed: int = SEED) -> np.ndarray:
    scores = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        probe = DomProbe().fit(X[tr], y[tr])
        scores[te] = probe.score(X[te])
    return scores


def oof_dom_calibrated(X: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS,
                       seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Returns (calibrated_probs, raw_scores)."""
    n = len(y)
    probs = np.zeros(n, dtype=np.float64)
    raw = np.zeros(n, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        probe = DomProbe().fit(X[tr], y[tr])
        raw_tr = probe.score(X[tr])
        raw_te = probe.score(X[te])
        raw[te] = raw_te
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(raw_tr, y[tr])
        probs[te] = iso.predict(raw_te)
    return probs, raw


# ---------------------------------------------------------------------------
# Bootstrap and DeLong CIs
# ---------------------------------------------------------------------------

def bootstrap_auroc(y: np.ndarray, scores: np.ndarray,
                    n_resamples: int = 2000, seed: int = SEED) -> dict:
    point = float(roc_auc_score(y, scores))
    indices = (np.arange(len(y)),)

    def auroc_stat(idx):
        i = idx.astype(int)
        if len(np.unique(y[i])) < 2:
            return np.nan
        return roc_auc_score(y[i], scores[i])

    res_bca = bootstrap(indices, auroc_stat, n_resamples=n_resamples,
                        method="BCa", random_state=seed)
    res_pct = bootstrap(indices, auroc_stat, n_resamples=n_resamples,
                        method="percentile", random_state=seed)
    return {
        "point": point,
        "bca_lo": float(res_bca.confidence_interval.low),
        "bca_hi": float(res_bca.confidence_interval.high),
        "pct_lo": float(res_pct.confidence_interval.low),
        "pct_hi": float(res_pct.confidence_interval.high),
    }


def delong_ci(y_true: np.ndarray, y_score: np.ndarray,
              alpha: float = 0.05) -> dict:
    """DeLong CI for AUROC (Sun & Xu 2014)."""
    y = y_true.astype(bool)
    pos = y_score[y]
    neg = y_score[~y]
    m, n = len(pos), len(neg)

    V10 = np.array([
        ((neg < p).sum() + 0.5 * (neg == p).sum()) / n for p in pos
    ])
    V01 = np.array([
        ((pos > nv).sum() + 0.5 * (pos == nv).sum()) / m for nv in neg
    ])

    auc = V10.mean()
    s10 = np.var(V10, ddof=1)
    s01 = np.var(V01, ddof=1)
    var_auc = s10 / m + s01 / n
    se = np.sqrt(var_auc)
    z = norm.ppf(1 - alpha / 2)

    return {
        "point": float(auc),
        "delong_lo": float(auc - z * se),
        "delong_hi": float(auc + z * se),
        "se": float(se),
    }


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15,
              strategy: str = "uniform") -> float:
    if strategy == "uniform":
        bins = np.linspace(0, 1, n_bins + 1)
    else:
        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.unique(np.percentile(y_prob, quantiles))
        if len(bins) < 2:
            return 0.0
        bins[0] = 0.0
        bins[-1] = 1.0 + 1e-8

    ece = 0.0
    total = len(y_true)
    for i in range(len(bins) - 1):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / total * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Risk-coverage and AURC
# ---------------------------------------------------------------------------

def risk_coverage_curve(y: np.ndarray, scores: np.ndarray,
                        cov_grid: np.ndarray | None = None) -> list[dict]:
    if cov_grid is None:
        cov_grid = np.arange(0.05, 1.001, 0.05)
    order = np.argsort(-scores)
    y_sorted = y[order]
    rows = []
    for c in cov_grid:
        k = max(1, int(round(c * len(y))))
        acc = float(y_sorted[:k].mean())
        risk = 1.0 - acc
        rows.append({"coverage": float(c), "accuracy": acc, "risk": risk, "n_kept": k})
    return rows


def aurc(y: np.ndarray, scores: np.ndarray) -> float:
    """Area Under the Risk-Coverage curve (lower = better)."""
    order = np.argsort(-scores)
    y_sorted = y[order].astype(float)
    n = len(y)
    cum_correct = np.cumsum(y_sorted)
    cum_acc = cum_correct / np.arange(1, n + 1)
    cum_risk = 1.0 - cum_acc
    coverages = np.arange(1, n + 1) / n
    return float(np.trapezoid(cum_risk, coverages))


def oracle_aurc(y: np.ndarray) -> float:
    return aurc(y, y.astype(float))


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    return plt


def save_fig(fig, name: str):
    path = FIGS_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------

def timer(label: str):
    class _Timer:
        def __enter__(self):
            self.t0 = time.time()
            print(f"[{label}] starting...")
            return self
        def __exit__(self, *_):
            dt = time.time() - self.t0
            print(f"[{label}] done in {dt:.1f}s")
    return _Timer()
