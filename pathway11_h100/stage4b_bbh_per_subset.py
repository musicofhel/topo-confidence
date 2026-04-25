#!/usr/bin/env python3
"""Stage 4b: BBH per-subset diagnostics (CPU-only, ~10 min).

Depends on stage 4a (pathway8_layerwise/extract_bbh.py) having run.

For each of 3 BBH subsets (tracking_shuffled, logical_deduction, web_of_lies):
  1. Load per-problem all-layer states from pathway8_layerwise/data/bbh/{subset}/
  2. Build features:
     - CoE-60d (from pathway8_layerwise/coe_features.py:compute_coe_batch)
     - L19 last-token DoM score (single scalar via mean-diff direction)
  3. 5-fold CV AUROC (LogisticRegression on CoE; DoM is a single-feature score)
  4. Bootstrap 95% CI (n_boot=2000) on AUROC
  5. Feature sign-flip sanity: cos(DoM_real, DoM_label_shuffled) — ~0 if signal real
  6. Length correlation: Spearman(seq_len, DoM_score) per subset

Output: pathway11_h100/results/stage4b_bbh_per_subset.json with per-subset
× per-method table + bootstrap CIs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathway8_layerwise.coe_features import compute_coe_batch
from pathway11_h100.config import (
    BBH_DATA_DIR,
    BBH_SUBSETS,
    N_TOTAL_LAYERS,
    RESULTS_DIR,
    STEERING_LAYER,
    mark_done,
)

SEED = 9999
N_BOOT = 2000


def load_subset_states(subset: str) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Load per-problem states + correctness + seq_lens for a BBH subset.

    Returns (states_list, y, seq_lens).
    """
    subset_dir = BBH_DATA_DIR / subset
    files = sorted(subset_dir.glob("problem_*.npz"))
    states_list: list[np.ndarray] = []
    y = []
    seq_lens = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        s = data["states"]  # (29, n_tokens, 1536)
        states_list.append(s)
        y.append(bool(data["correct"]))
        seq_lens.append(int(s.shape[1]))
    return states_list, np.array(y, dtype=int), np.array(seq_lens)


def dom_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Difference-of-means direction, L2-normalized."""
    w = X[y == 1].mean(0) - X[y == 0].mean(0)
    return w / (np.linalg.norm(w) + 1e-12)


def cv_auroc_coe(X: np.ndarray, y: np.ndarray) -> float:
    """5-fold LR AUROC on CoE features."""
    if y.sum() < 5 or y.sum() > len(y) - 5:
        return float("nan")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    scores = cross_val_predict(lr, X, y, cv=skf, method="decision_function")
    return float(roc_auc_score(y, scores))


def cv_auroc_dom(X_L19: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """5-fold out-of-fold DoM score AUROC. Returns (auroc, oof_scores)."""
    if y.sum() < 5 or y.sum() > len(y) - 5:
        return float("nan"), np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.empty(len(y))
    for tr, te in skf.split(np.arange(len(y)), y):
        w = dom_direction(X_L19[tr], y[tr])
        oof[te] = X_L19[te] @ w
    return float(roc_auc_score(y, oof)), oof


def bootstrap_auroc(scores: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Bootstrap 95% CI on AUROC. Returns (lo, med, hi)."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        bi = rng.integers(0, n, size=n)
        yb = y[bi]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        vals.append(roc_auc_score(yb, scores[bi]))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    vals = np.array(vals)
    return (
        float(np.percentile(vals, 2.5)),
        float(np.median(vals)),
        float(np.percentile(vals, 97.5)),
    )


def sign_flip_check(X_L19: np.ndarray, y: np.ndarray, n_shuffles: int = 50) -> float:
    """Mean |cos(DoM_real, DoM_shuffled)| over n_shuffles. Near 0 if signal is real."""
    rng = np.random.default_rng(SEED)
    d_real = dom_direction(X_L19, y)
    cosines = []
    for _ in range(n_shuffles):
        y_shuf = y.copy()
        rng.shuffle(y_shuf)
        d_shuf = dom_direction(X_L19, y_shuf)
        cosines.append(abs(float(d_real @ d_shuf)))
    return float(np.mean(cosines))


def analyze_subset(subset: str) -> dict:
    """Run full diagnostic suite on one BBH subset."""
    print(f"\n--- Subset: {subset} ---")
    t0 = time.time()

    states_list, y, seq_lens = load_subset_states(subset)
    n = len(y)
    print(f"  n={n}, correct={int(y.sum())} ({y.mean():.1%})")

    # L19 last-token activations: (n, 1536)
    X_L19 = np.stack([
        s[STEERING_LAYER, -1, :].astype(np.float32) for s in states_list
    ])

    # CoE features: (n, 60)
    X_coe, _ = compute_coe_batch(states_list)

    result: dict = {"subset": subset, "n": int(n), "n_correct": int(y.sum()), "acc": float(y.mean())}

    # CoE
    coe_auroc = cv_auroc_coe(X_coe, y)
    # Bootstrap AUROC on CoE requires refitting — instead, use cross_val_predict scores
    if not np.isnan(coe_auroc):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        coe_scores = cross_val_predict(lr, X_coe, y, cv=skf, method="decision_function")
        lo, med, hi = bootstrap_auroc(coe_scores, y)
        result["coe"] = {"auroc": coe_auroc, "ci95": [lo, hi], "median_boot": med}
        print(f"  CoE:   AUROC={coe_auroc:.3f} 95%CI=[{lo:.3f}, {hi:.3f}]")
    else:
        result["coe"] = {"auroc": None, "reason": "class imbalance"}
        print(f"  CoE:   skipped (class imbalance)")

    # L19 DoM
    dom_auroc, dom_oof = cv_auroc_dom(X_L19, y)
    if not np.isnan(dom_auroc):
        lo, med, hi = bootstrap_auroc(dom_oof, y)
        result["l19_dom"] = {"auroc": dom_auroc, "ci95": [lo, hi], "median_boot": med}
        print(f"  L19 DoM: AUROC={dom_auroc:.3f} 95%CI=[{lo:.3f}, {hi:.3f}]")

        sf = sign_flip_check(X_L19, y)
        result["l19_dom"]["sign_flip_mean_abs_cos"] = sf
        print(f"  L19 DoM sign-flip sanity: mean|cos(real, shuffled)|={sf:.3f} (lower=better)")
    else:
        result["l19_dom"] = {"auroc": None, "reason": "class imbalance"}
        print(f"  L19 DoM: skipped (class imbalance)")

    # Length correlation with L19 DoM OOF score
    if not np.isnan(dom_auroc):
        rho, p = spearmanr(seq_lens, dom_oof)
        result["length_corr"] = {"rho_l19_dom_score_vs_seqlen": float(rho), "p": float(p)}
        print(f"  Length×L19_score: rho={rho:+.3f} (p={p:.2g})")

    # Length vs correctness (does length already predict correctness?)
    rho_len, p_len = spearmanr(seq_lens, y)
    result["length_vs_correct"] = {"rho": float(rho_len), "p": float(p_len)}
    print(f"  Length×correct:     rho={rho_len:+.3f} (p={p_len:.2g})")

    elapsed = time.time() - t0
    print(f"  [done in {elapsed:.1f}s]")
    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Stage 4b: BBH per-subset diagnostics")
    print("=" * 70)

    out = {
        "meta": {
            "n_boot": N_BOOT,
            "seed": SEED,
            "cv": "StratifiedKFold(5, shuffle=True, random_state=9999)",
            "steering_layer": STEERING_LAYER,
        },
        "subsets": {},
    }

    for subset in BBH_SUBSETS:
        subset_dir = BBH_DATA_DIR / subset
        if not subset_dir.exists() or not list(subset_dir.glob("problem_*.npz")):
            print(f"\n[SKIP] {subset}: no extracted data at {subset_dir}")
            continue
        res = analyze_subset(subset)
        out["subsets"][subset] = res

    # Summary table
    print("\n" + "=" * 70)
    print("Summary (AUROC [95% CI]):")
    print(f"  {'subset':>40} {'CoE':>14} {'L19_DoM':>14} {'sign_flip':>10}")
    for name, res in out["subsets"].items():
        coe = res.get("coe", {})
        dom = res.get("l19_dom", {})
        coe_str = (
            f"{coe['auroc']:.3f} [{coe['ci95'][0]:.2f}, {coe['ci95'][1]:.2f}]"
            if coe.get("auroc") is not None else "n/a"
        )
        dom_str = (
            f"{dom['auroc']:.3f} [{dom['ci95'][0]:.2f}, {dom['ci95'][1]:.2f}]"
            if dom.get("auroc") is not None else "n/a"
        )
        sf = dom.get("sign_flip_mean_abs_cos", float("nan"))
        print(f"  {name:>40} {coe_str:>14} {dom_str:>14} {sf:>10.3f}")

    # Save
    out_path = RESULTS_DIR / "stage4b_bbh_per_subset.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved: {out_path}")

    if out["subsets"]:
        mark_done("stage4b", RESULTS_DIR)
        print("[DONE] stage4b marker written")


if __name__ == "__main__":
    main()
