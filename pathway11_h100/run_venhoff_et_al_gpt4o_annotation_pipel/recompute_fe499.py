"""P11-FE499 — Venhoff behavior-DoM decomposition test for F-2.

Refutation test for F-2's single-direction (prefill L19 DoM) framing. The
Venhoff et al. pipeline annotates each cached MATH-500 K=1 CoT (Qwen-2.5-1.5B)
with GPT-4o behavior labels (e.g. deduction, backtracking, verification,
restatement), then extracts a per-behavior Difference-of-Means vector at
L17/L18/L19. This recompute consumes the *cached* annotation product (the
GPT-4o pass and the L17/L18 activation re-extract are an offline, networked +
GPU step — not reproducible here) and:

  1. Computes cos(behavior_DoM, prefill_correctness_DoM) at each annotated layer.
  2. Projects the L19 prefill states onto each behavior DoM to form a
     4-behavior feature stack, scores 5-fold OOF logistic-regression AUROC.
  3. Compares the stacked AUROC to F-2's 0.7731 single-direction baseline.

If stacked AUROC > 0.79 the prefill DoM is an aggregated projection of behavior
axes and F-2 needs decomposition; the cosine numbers decide whether F-3's
prefill/final orthogonality is temporal or a behavior-decomposition artifact.

Required cached input (offline-produced; if absent the experiment cannot run on
local CPU and the script exits MISSING_REGEN_INPUT):
  pathway11_h100/venhoff_behaviors/cache/behavior_doms.npz
    behavior_dom_l19 : (n_behaviors, 1536) float — per-behavior DoM at L19
    behavior_dom_l18 : (n_behaviors, 1536) float — optional, per-behavior DoM L18
    behavior_dom_l17 : (n_behaviors, 1536) float — optional, per-behavior DoM L17
    behavior_names   : (n_behaviors,) str       — optional behavior labels
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
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
BEHAVIOR_NPZ = ROOT / "pathway11_h100/venhoff_behaviors/cache/behavior_doms.npz"
OUT_JSON = ROOT / "pathway11_h100/venhoff_behaviors/results.json"

F2_BASELINE = 0.7731
SEED = 9999
N_FOLDS = 5


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
    return float((a @ b) / (na * nb))


def main() -> int:
    for path in (CACHE, DOM_NPZ, BEHAVIOR_NPZ):
        if not path.exists():
            print("MISSING_REGEN_INPUT", path, file=sys.stderr)
            return 2

    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float64)
    y = cache["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)

    # Full-data prefill correctness DoM (reference axis for cosine).
    prefill_dom = X[y].mean(axis=0) - X[~y].mean(axis=0)

    blob = np.load(BEHAVIOR_NPZ, allow_pickle=True)
    if "behavior_dom_l19" not in blob:
        print("MISSING_REGEN_INPUT", BEHAVIOR_NPZ, "(no behavior_dom_l19)", file=sys.stderr)
        return 2
    behavior_dom_l19 = np.asarray(blob["behavior_dom_l19"], dtype=np.float64)
    if behavior_dom_l19.ndim != 2 or behavior_dom_l19.shape[1] != 1536:
        print("MISSING_REGEN_INPUT", BEHAVIOR_NPZ, "(bad behavior_dom_l19 shape)", file=sys.stderr)
        return 2
    n_behaviors = behavior_dom_l19.shape[0]

    if "behavior_names" in blob:
        names = [str(s) for s in np.asarray(blob["behavior_names"]).ravel().tolist()]
    else:
        names = [f"behavior_{i}" for i in range(n_behaviors)]

    # Per-layer cosines of each behavior DoM against the prefill correctness DoM.
    cosines_by_layer: dict[str, list[float]] = {}
    for layer_key in ("behavior_dom_l17", "behavior_dom_l18", "behavior_dom_l19"):
        if layer_key not in blob:
            continue
        mat = np.asarray(blob[layer_key], dtype=np.float64)
        if mat.shape != behavior_dom_l19.shape:
            continue
        cosines_by_layer[layer_key] = [cosine(mat[i], prefill_dom) for i in range(n_behaviors)]

    # 4-behavior feature stack: project L19 prefill states onto each behavior DoM.
    feats = X @ behavior_dom_l19.T  # (500, n_behaviors)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_stacked = np.zeros(len(y), dtype=np.float64)
    oof_single = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(feats, y):
        mu = feats[train_idx].mean(axis=0)
        sd = feats[train_idx].std(axis=0)
        sd[sd < 1e-12] = 1.0
        Ztr = (feats[train_idx] - mu) / sd
        Zte = (feats[test_idx] - mu) / sd
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(Ztr, y[train_idx])
        oof_stacked[test_idx] = clf.decision_function(Zte)

        # Single-behavior-axis comparison is the DoM score itself, refit per fold.
        d_single = X[train_idx][y[train_idx]].mean(0) - X[train_idx][~y[train_idx]].mean(0)
        oof_single[test_idx] = X[test_idx] @ d_single

    auroc_stacked = auroc(oof_stacked, y)
    auroc_single_refit = auroc(oof_single, y)
    auroc_dom_cached = auroc(dom_score, y)

    out = {
        "experiment": "P11-FE499",
        "n_behaviors": int(n_behaviors),
        "behavior_names": names,
        "f2_baseline": F2_BASELINE,
        "auroc_stacked_4behavior_oof": float(auroc_stacked),
        "auroc_single_dom_refit_oof": float(auroc_single_refit),
        "auroc_dom_cached": float(auroc_dom_cached),
        "delta_vs_f2_baseline": float(auroc_stacked - F2_BASELINE),
        "stacked_beats_0.79": bool(auroc_stacked > 0.79),
        "cosine_behavior_dom_vs_prefill_dom": {
            layer: {names[i]: float(vals[i]) for i in range(n_behaviors)}
            for layer, vals in cosines_by_layer.items()
        },
        "max_abs_cosine_l19": (
            float(np.nanmax(np.abs(cosines_by_layer["behavior_dom_l19"])))
            if "behavior_dom_l19" in cosines_by_layer else float("nan")
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())