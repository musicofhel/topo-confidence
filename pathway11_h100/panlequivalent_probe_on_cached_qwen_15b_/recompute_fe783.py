"""P11-FE783 — PANL-equivalent probe on cached Qwen 1.5B MATH-500 1024-tok residuals.

Tests Refutation #1 (paper reports PANL verification AUROC 0.986 on Gemma
TriviaQA and PANL A2-correctness 0.774 on Qwen 7B). Here we locate the
post-answer-newline (PANL) token in each K=1 MATH-500 generation, extract its
L19 residual (plus a layer sweep when multi-layer states are cached), train an
L2-regularized logistic probe (OOF 5-fold) against K=1 correctness, and compare
the resulting AUROC against the canonical prefill L19 DoM (0.7731) and
final-token L19 DoM (0.7186) baselines. If a same-architecture-family Qwen 1.5B
PANL probe beats prefill DoM, F-2's headline-position claim must be revised.

Inputs (cached P11 Stage-2 NPZs):
  - PANL_CACHE: post-answer-newline residuals per problem. Accepts either
      {'states': (500, L, 1536), 'layers': (L,)} or per-layer keys 'L<int>'.
  - main prefill cache: provides K=1 'correct' labels.
  - DoM npz: provides the prefill-DoM projection score for the baseline AUROC.
If PANL_CACHE is absent the probe cannot run -> MISSING_REGEN_INPUT, return 2.
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
PANL_CACHE = ROOT / "pathway11_h100/panl_probe/cache/m15b_panl.npz"
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/panl_probe/results.json"

SEED = 9999
N_FOLDS = 5
C_GRID = [0.001, 0.01, 0.1, 1.0]
TARGET_LAYER = 19
PREFILL_DOM_AUROC = 0.7731   # 1024tok canonical (FINDINGS F-2)
FINAL_DOM_AUROC = 0.7186     # 1024tok final-token L19 DoM


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def load_panl(cache) -> tuple[np.ndarray, np.ndarray]:
    """Return (states (N, L, H), layers (L,)) from a flexible PANL NPZ schema."""
    keys = list(cache.keys())
    if "states" in keys:
        states = np.asarray(cache["states"]).astype(np.float64)
        if states.ndim == 2:  # single layer stored as (N, H)
            states = states[:, None, :]
        layers = (
            np.asarray(cache["layers"]).astype(int)
            if "layers" in keys
            else np.arange(states.shape[1], dtype=int)
        )
        return states, layers
    layer_keys = sorted(
        (k for k in keys if k.startswith("L") and k[1:].isdigit()),
        key=lambda k: int(k[1:]),
    )
    if layer_keys:
        layers = np.array([int(k[1:]) for k in layer_keys], dtype=int)
        states = np.stack([np.asarray(cache[k]).astype(np.float64) for k in layer_keys], axis=1)
        return states, layers
    raise KeyError(f"PANL cache has no recognizable state keys: {keys}")


def oof_logistic_auroc(X: np.ndarray, y: np.ndarray, C: float) -> float:
    """Standardize per-fold, fit L2-logistic, accumulate OOF decision scores."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, te in skf.split(X, y):
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0)
        sd[sd < 1e-8] = 1.0
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd
        clf = LogisticRegression(
            penalty="l2", C=C, solver="lbfgs", max_iter=2000, random_state=SEED
        )
        clf.fit(Xtr, y[tr])
        oof[te] = clf.decision_function(Xte)
    return auroc(oof, y)


def main() -> int:
    if not PANL_CACHE.exists():
        print("MISSING_REGEN_INPUT", PANL_CACHE, file=sys.stderr)
        return 2
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    correct = np.load(CACHE)["correct"].astype(bool)
    assert correct.shape == (500,), correct.shape

    panl_blob = np.load(PANL_CACHE)
    states, layers = load_panl(panl_blob)
    if states.shape[0] != 500:
        print("MISSING_REGEN_INPUT", f"PANL N={states.shape[0]} != 500", file=sys.stderr)
        return 2

    # Per-layer probe: best-C OOF logistic AUROC across the cached layer sweep.
    layer_sweep: dict[str, dict] = {}
    for li, layer in enumerate(layers):
        X = states[:, li, :]
        per_C = {f"C={c}": oof_logistic_auroc(X, correct, c) for c in C_GRID}
        best_c = max(per_C, key=lambda k: (per_C[k] if not np.isnan(per_C[k]) else -1.0))
        layer_sweep[str(int(layer))] = {
            "auroc_by_C": per_C,
            "best_C": best_c,
            "best_auroc": per_C[best_c],
        }

    # Pull out L19 (the headline-position comparison) and the global best layer.
    l19_key = str(TARGET_LAYER)
    panl_l19_auroc = (
        layer_sweep[l19_key]["best_auroc"] if l19_key in layer_sweep else float("nan")
    )
    best_layer = max(
        layer_sweep,
        key=lambda k: (
            layer_sweep[k]["best_auroc"]
            if not np.isnan(layer_sweep[k]["best_auroc"])
            else -1.0
        ),
    )
    panl_best_auroc = layer_sweep[best_layer]["best_auroc"]

    # Recompute prefill-DoM AUROC from the cached scores for self-consistency.
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        dom_auroc_recomputed = auroc(dom_score, correct)
    else:
        dom_auroc_recomputed = float("nan")

    refutes_f2 = bool(
        not np.isnan(panl_l19_auroc) and panl_l19_auroc > PREFILL_DOM_AUROC
    )

    out = {
        "experiment": "P11-FE783",
        "n": int(correct.shape[0]),
        "k1_accuracy": float(correct.mean()),
        "layers_cached": [int(x) for x in layers],
        "panl_layer_sweep": layer_sweep,
        "panl_L19_auroc": float(panl_l19_auroc),
        "panl_best_layer": int(best_layer),
        "panl_best_auroc": float(panl_best_auroc),
        "baseline_prefill_dom_auroc": PREFILL_DOM_AUROC,
        "baseline_prefill_dom_auroc_recomputed": float(dom_auroc_recomputed),
        "baseline_final_token_dom_auroc": FINAL_DOM_AUROC,
        "panl_L19_minus_prefill_dom": float(panl_l19_auroc - PREFILL_DOM_AUROC),
        "panl_best_minus_prefill_dom": float(panl_best_auroc - PREFILL_DOM_AUROC),
        "refutes_f2_headline_position": refutes_f2,
        "verdict": (
            "PANL_BEATS_PREFILL_DOM" if refutes_f2 else "PREFILL_DOM_HOLDS"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())