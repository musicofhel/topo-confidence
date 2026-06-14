"""P11-FE728 — Isotonic recalibration of prefill L19 DoM scores vs Miao-Ungar ECE.

F-8 currently frames the prefill DoM probe as selective-prediction (71.6% acc at
coverage 0.5). Miao-Ungar (Table 4) use the same probe architecture as a
regression target with isotonic calibration to drive expected calibration error
(ECE) down 4-7x. This script fits out-of-fold isotonic regression mapping the
existing DoM scores onto K=8 empirical accuracy, then reports ECE / Brier / MAE
of the calibrated probabilities against that K=8 target (and against the binary
1024-tok correctness label). If the calibrated ECE < 10 (their unsteered-verbal
Mistral 35.1 / Llama 14.9 / Qwen 36.0), F-8 is under-framed and should be
re-headlined as a calibration result, not just selective-prediction.
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
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
OUT_JSON = ROOT / "pathway11_h100/isotonic_calibration/results.json"

SEED = 9999
N_FOLDS = 5
N_BINS = 10
N_PROBLEMS = 500

# Miao-Ungar Table 4 unsteered-verbal ECE (x100), for cross-reference only.
MIAO_UNGAR_TABLE4 = {"mistral": 35.1, "llama": 14.9, "qwen": 36.0}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def ece(probs: np.ndarray, target: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error (x100) of predicted probs vs a target accuracy.

    Equal-width bins on [0, 1]; per-bin |mean(prob) - mean(target)| weighted by
    bin mass. ``target`` may be binary {0,1} or a continuous empirical accuracy.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    n = len(probs)
    total = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        m = int(mask.sum())
        if m == 0:
            continue
        total += (m / n) * abs(probs[mask].mean() - target[mask].mean())
    return float(100.0 * total)


def main() -> int:
    for path in (CACHE, DOM_NPZ):
        if not path.exists():
            print("MISSING_REGEN_INPUT", path, file=sys.stderr)
            return 2
    if not K8_DIR.exists():
        print("MISSING_REGEN_INPUT", K8_DIR, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    correct = cache["correct"].astype(bool)
    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert correct.shape == (N_PROBLEMS,) and dom_score.shape == (N_PROBLEMS,)

    # K=8 empirical accuracy per problem as the isotonic regression target.
    emp_acc = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    missing = []
    for i in range(N_PROBLEMS):
        f = K8_DIR / f"problem_{i:03d}.npz"
        if not f.exists():
            missing.append(i)
            continue
        emp_acc[i] = np.load(f)["correct"].astype(np.float64).mean()
    if missing:
        print("MISSING_REGEN_INPUT", f"{len(missing)} k8 files (e.g. {missing[:3]})",
              file=sys.stderr)
        return 2

    # Out-of-fold isotonic calibration: dom_score -> K=8 empirical accuracy.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    p_iso = np.zeros(N_PROBLEMS, dtype=np.float64)
    for train_idx, test_idx in skf.split(dom_score, correct):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(dom_score[train_idx], emp_acc[train_idx])
        p_iso[test_idx] = iso.predict(dom_score[test_idx])

    # Raw min-max-mapped DoM probability (uncalibrated reference).
    lo, hi = dom_score.min(), dom_score.max()
    p_raw = (dom_score - lo) / (hi - lo) if hi > lo else np.full(N_PROBLEMS, 0.5)

    correct_f = correct.astype(np.float64)
    out = {
        "experiment": "P11-FE728",
        "n_problems": N_PROBLEMS,
        "auroc_dom_binary": auroc(dom_score, correct),
        "auroc_iso_binary": auroc(p_iso, correct),
        "k8_emp_acc_mean": float(emp_acc.mean()),
        # ECE (x100) of calibrated probs against the K=8 empirical-accuracy target.
        "ece_iso_vs_k8": ece(p_iso, emp_acc),
        "ece_raw_vs_k8": ece(p_raw, emp_acc),
        # ECE against the binary 1024-tok correctness label.
        "ece_iso_vs_binary": ece(p_iso, correct_f),
        "ece_raw_vs_binary": ece(p_raw, correct_f),
        # Brier / MAE of calibrated probs vs K=8 empirical accuracy.
        "brier_iso_vs_k8": float(np.mean((p_iso - emp_acc) ** 2)),
        "brier_raw_vs_k8": float(np.mean((p_raw - emp_acc) ** 2)),
        "mae_iso_vs_k8": float(np.mean(np.abs(p_iso - emp_acc))),
        "mae_raw_vs_k8": float(np.mean(np.abs(p_raw - emp_acc))),
        # Brier / MAE against binary label.
        "brier_iso_vs_binary": float(np.mean((p_iso - correct_f) ** 2)),
        "mae_iso_vs_binary": float(np.mean(np.abs(p_iso - correct_f))),
        "miao_ungar_table4_unsteered_verbal_ece": MIAO_UNGAR_TABLE4,
    }
    out["ece_reduction_factor_vs_k8"] = (
        float(out["ece_raw_vs_k8"] / out["ece_iso_vs_k8"])
        if out["ece_iso_vs_k8"] > 0 else float("inf")
    )
    out["iso_ece_under_10_vs_k8"] = bool(out["ece_iso_vs_k8"] < 10.0)
    out["iso_ece_under_10_vs_binary"] = bool(out["ece_iso_vs_binary"] < 10.0)
    # F-8 recalibration verdict: re-headline only if calibrated ECE beats 10.
    out["recommend_recalibration_reframe"] = bool(
        out["iso_ece_under_10_vs_k8"] or out["iso_ece_under_10_vs_binary"]
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())