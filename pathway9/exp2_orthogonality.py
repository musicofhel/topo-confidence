"""Exp 2: PH vs CoE orthogonality.

Train LR on PH (168 layer-wise) and CoE (60) separately. Measure:
  - prediction correlation r(ph_probs, coe_probs) on holdout
  - error disagreement sets (high-confidence PH with low-confidence CoE, etc.)
  - simple 0.5+0.5 ensemble on holdout
  - STACKED ensemble using nested CV: out-of-fold base probs on train,
    meta-LR fit on those, then evaluate refitted base models on holdout.

Decision:
  - ensemble > max(PH, CoE) by >0.01 → complementary
  - prediction corr r > 0.8 → redundant (PH = noisy CoE)
  - prediction corr r < 0.5 → orthogonal

Expects RunPod volume has:
  pathway8_layerwise/results/exp1_features_all.npy    shape (500, 168)
  pathway8_layerwise/results/exp2_coe_features.npy    shape (500, 60)

Output: pathway9/results/exp2_orthogonality.json + exp2.done
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def load_data():
    new_labels = np.load(REPO / "pathway6_rebuild/phase0_relabel/baseline_correct_v2.npy")
    old_labels = np.load(REPO / "pathway2/track_a/phase0/baseline_correct.npy")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=9999)
    train_sss, hold_sss = next(sss.split(np.zeros(500), old_labels.astype(int)))
    train_idx = np.sort(train_sss)
    holdout_idx = np.sort(hold_sss)

    ph = np.load(REPO / "pathway8_layerwise/results/exp1_features_all.npy")
    coe = np.load(REPO / "pathway8_layerwise/results/exp2_coe_features.npy")
    assert ph.shape[0] == 500, f"PH wrong shape: {ph.shape}"
    assert coe.shape[0] == 500, f"CoE wrong shape: {coe.shape}"

    return dict(
        labels=new_labels,
        train_idx=train_idx, holdout_idx=holdout_idx,
        ph=ph, coe=coe,
    )


def make_pipe():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def oof_probs(X, y, cv=5):
    """Out-of-fold predicted probabilities via stratified k-fold."""
    probs = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y):
        pipe = make_pipe().fit(X[tr], y[tr])
        probs[te] = pipe.predict_proba(X[te])[:, 1]
    return probs


def main():
    print("=" * 60)
    print("EXP 2: PH vs CoE Orthogonality")
    print("=" * 60, flush=True)

    d = load_data()
    y_tr = d["labels"][d["train_idx"]]
    y_ho = d["labels"][d["holdout_idx"]]
    print(f"  PH shape: {d['ph'].shape}  CoE shape: {d['coe'].shape}")
    print(f"  Train pos: {y_tr.sum()}/{len(y_tr)}  Holdout pos: {y_ho.sum()}/{len(y_ho)}", flush=True)

    # Base models: holdout probabilities
    ph_tr, ph_ho = d["ph"][d["train_idx"]], d["ph"][d["holdout_idx"]]
    coe_tr, coe_ho = d["coe"][d["train_idx"]], d["coe"][d["holdout_idx"]]

    ph_pipe = make_pipe().fit(ph_tr, y_tr)
    coe_pipe = make_pipe().fit(coe_tr, y_tr)
    ph_probs_ho = ph_pipe.predict_proba(ph_ho)[:, 1]
    coe_probs_ho = coe_pipe.predict_proba(coe_ho)[:, 1]

    ph_auroc = float(roc_auc_score(y_ho, ph_probs_ho))
    coe_auroc = float(roc_auc_score(y_ho, coe_probs_ho))
    print(f"  PH holdout AUROC:  {ph_auroc:.3f}")
    print(f"  CoE holdout AUROC: {coe_auroc:.3f}", flush=True)

    # Prediction correlation
    r_probs = float(np.corrcoef(ph_probs_ho, coe_probs_ho)[0, 1])
    print(f"\n  Prediction correlation r(ph_probs, coe_probs) = {r_probs:+.3f}", flush=True)

    # Disagreement analysis
    ph_high = ph_probs_ho > 0.7
    ph_low = ph_probs_ho < 0.3
    coe_high = coe_probs_ho > 0.7
    coe_low = coe_probs_ho < 0.3

    disagreement_sets = {}
    for label, mask in [
        ("ph_high_coe_low", ph_high & coe_low),
        ("coe_high_ph_low", coe_high & ph_low),
        ("both_high", ph_high & coe_high),
        ("both_low", ph_low & coe_low),
    ]:
        n = int(mask.sum())
        frac_correct = float(y_ho[mask].mean()) if n > 0 else None
        disagreement_sets[label] = {"n": n, "frac_correct": frac_correct}
        print(f"  {label:20s} n={n}  frac_correct={frac_correct}")

    # Simple ensemble (avg of holdout probs)
    ens_probs = 0.5 * ph_probs_ho + 0.5 * coe_probs_ho
    ens_auroc = float(roc_auc_score(y_ho, ens_probs))
    print(f"\n  Simple average ensemble AUROC: {ens_auroc:.3f}", flush=True)

    # Stacked ensemble with nested CV
    ph_oof = oof_probs(ph_tr, y_tr, cv=5)
    coe_oof = oof_probs(coe_tr, y_tr, cv=5)
    stack_train = np.column_stack([ph_oof, coe_oof])
    stack_holdout = np.column_stack([ph_probs_ho, coe_probs_ho])
    stack_pipe = make_pipe().fit(stack_train, y_tr)
    stack_probs_ho = stack_pipe.predict_proba(stack_holdout)[:, 1]
    stack_auroc = float(roc_auc_score(y_ho, stack_probs_ho))
    stack_coefs = stack_pipe.named_steps["lr"].coef_[0].tolist()
    print(f"  Stacked ensemble AUROC: {stack_auroc:.3f}  (coefs: ph={stack_coefs[0]:+.3f}  coe={stack_coefs[1]:+.3f})", flush=True)

    # Decision
    best_single = max(ph_auroc, coe_auroc)
    best_ensemble = max(ens_auroc, stack_auroc)
    lift = best_ensemble - best_single

    if r_probs > 0.8:
        verdict = f"REDUNDANT — PH and CoE predictions correlate at r={r_probs:.2f}. PH is a noisy proxy for CoE."
    elif lift > 0.01:
        verdict = f"COMPLEMENTARY — ensemble lift {lift:+.3f} over best single (r={r_probs:.2f}). Worth keeping both."
    elif lift < -0.01:
        verdict = f"CONFLICTING — ensemble hurts by {lift:+.3f}. CoE alone is better."
    else:
        verdict = f"REDUNDANT (weak) — ensemble lift {lift:+.3f} within noise. CoE alone is sufficient."

    print(f"\n  VERDICT: {verdict}", flush=True)

    results = {
        "experiment": "exp2_orthogonality",
        "ph_auroc_holdout": ph_auroc,
        "coe_auroc_holdout": coe_auroc,
        "prediction_correlation": r_probs,
        "disagreement_sets": disagreement_sets,
        "simple_ensemble_auroc": ens_auroc,
        "stacked_ensemble_auroc": stack_auroc,
        "stacked_coefs": {"ph": stack_coefs[0], "coe": stack_coefs[1]},
        "best_single_auroc": float(best_single),
        "best_ensemble_auroc": float(best_ensemble),
        "ensemble_lift": float(lift),
        "verdict": verdict,
    }
    (RESULTS / "exp2_orthogonality.json").write_text(json.dumps(results, indent=2))
    (RESULTS / "exp2.done").touch()
    print("\n  Saved: pathway9/results/exp2_orthogonality.json")


if __name__ == "__main__":
    main()
