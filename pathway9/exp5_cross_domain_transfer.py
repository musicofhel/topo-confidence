"""Exp 5: Cross-domain transfer matrix (MATH-500 ↔ BBH pooled).

For each feature type (PH layer-wise, CoE, D2H-lite):
  - Train on MATH-500 (all 500), test on BBH (750 pooled)
  - Train on BBH (750 pooled), test on MATH-500 (all 500)
  - StandardScaler fit on TRAIN only.

Reuses pathway8_layerwise feature modules (must be importable).

Expects on RunPod:
  pathway8_layerwise/data/math500/problem_XXX.npz
  pathway8_layerwise/data/bbh/<subset>/problem_XXX.npz  (3 subsets × 250)

Output: pathway9/results/exp5_cross_domain.json + exp5.done
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pathway8_layerwise.config import (
    BBH_DATA_DIR, MATH500_DATA_DIR, LR_PARAMS,
    load_all_layer_states, load_manifest,
)
from pathway8_layerwise.coe_features import compute_coe_batch
from pathway8_layerwise.d2hscore_features import compute_d2h_batch
from pathway8_layerwise.layerwise_features import (
    compute_layerwise_ph_batch, fit_per_layer_pca,
)

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

BBH_SUBSETS = [
    "tracking_shuffled_objects_seven_objects",
    "logical_deduction_seven_objects",
    "web_of_lies",
]


def load_benchmark(name):
    """Returns (states_list, labels, manifest).

    MATH-500 has per-benchmark manifest.json co-located with problem files.
    BBH has ONE combined manifest at BBH_DATA_DIR/manifest.json covering all
    750 pooled problems; per-subset files are at BBH_DATA_DIR/<subset>/.
    """
    if name == "math500":
        data_dir = MATH500_DATA_DIR
        manifest = load_manifest(data_dir)
        n = manifest["n_problems"]
        states = load_all_layer_states(data_dir, n)
        labels = np.array([p.get("correct", False) for p in manifest["problems"]], dtype=bool)
        return states, labels, manifest

    # BBH subset: filter combined manifest, load states from subset dir.
    subset_dir = BBH_DATA_DIR / name
    combined = load_manifest(BBH_DATA_DIR)
    subset_problems = sorted(
        (p for p in combined["problems"] if p["subset"] == name),
        key=lambda p: p["subset_idx"],
    )
    n = len(subset_problems)
    states = load_all_layer_states(subset_dir, n)
    labels = np.array([p.get("correct", False) for p in subset_problems], dtype=bool)
    return states, labels, {"n_problems": n, "problems": subset_problems}


def compute_features(states, method, train_idx=None):
    """Compute features for a method. train_idx is only used by PH (per-layer PCA)."""
    if method == "layerwise_ph":
        if train_idx is None:
            train_idx = np.arange(len(states))
        transforms = fit_per_layer_pca(states, train_idx)
        return compute_layerwise_ph_batch(states, transforms, n_jobs=-1)
    elif method == "coe":
        feats, _ = compute_coe_batch(states)
        return feats
    elif method == "d2h_lite":
        feats, _ = compute_d2h_batch(states, full=False)
        return feats
    raise ValueError(method)


def transfer_test(X_train, y_train, X_test, y_test, description):
    """Train LR on source domain, test on target."""
    scaler = StandardScaler().fit(X_train)
    Xs_tr = scaler.transform(X_train)
    Xs_te = scaler.transform(X_test)
    lr = LogisticRegression(C=1.0, **LR_PARAMS).fit(Xs_tr, y_train)
    try:
        auroc = float(roc_auc_score(y_test, lr.predict_proba(Xs_te)[:, 1]))
    except ValueError:
        auroc = 0.5
    print(f"  {description}: AUROC={auroc:.3f}  (train n={len(y_train)}, test n={len(y_test)})", flush=True)
    return auroc


def main():
    print("=" * 60)
    print("EXP 5: Cross-Domain Transfer MATH-500 ↔ BBH pooled")
    print("=" * 60, flush=True)

    # Load MATH-500
    print("\n  Loading MATH-500 ...", flush=True)
    math_states, math_labels, _ = load_benchmark("math500")
    math_valid = [i for i, s in enumerate(math_states) if s is not None]
    math_states = [math_states[i] for i in math_valid]
    math_labels = math_labels[math_valid]
    print(f"  MATH-500: {len(math_states)} valid problems, {math_labels.sum()} correct", flush=True)

    # Load BBH pooled (3 subsets)
    print("\n  Loading BBH subsets ...", flush=True)
    bbh_states = []
    bbh_labels_list = []
    bbh_sizes = {}
    for subset in BBH_SUBSETS:
        s_states, s_labels, _ = load_benchmark(subset)
        valid = [i for i, s in enumerate(s_states) if s is not None]
        s_states = [s_states[i] for i in valid]
        s_labels = s_labels[valid]
        bbh_states.extend(s_states)
        bbh_labels_list.append(s_labels)
        bbh_sizes[subset] = len(s_states)
        print(f"    {subset}: {len(s_states)} valid, {s_labels.sum()} correct", flush=True)
    bbh_labels = np.concatenate(bbh_labels_list)
    print(f"  BBH pooled: {len(bbh_states)} problems, {bbh_labels.sum()} correct", flush=True)

    methods = ["layerwise_ph", "coe", "d2h_lite"]
    transfer_results = {}
    within_domain = {}

    # Within-domain 80/20 splits shared across methods.
    sss_m = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=9999)
    mtr, mho = next(sss_m.split(np.zeros(len(math_labels)), math_labels.astype(int)))
    sss_b = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=9999)
    btr, bho = next(sss_b.split(np.zeros(len(bbh_labels)), bbh_labels.astype(int)))

    for method in methods:
        print(f"\n  ---- Method: {method} ----", flush=True)
        t0 = time.time()

        if method == "layerwise_ph":
            # D11: transfer uses source-domain PCA applied to BOTH source and target.
            # MATH → BBH: fit PCA on MATH, apply to both.
            math_transforms = fit_per_layer_pca(math_states, np.arange(len(math_states)))
            mf_m = compute_layerwise_ph_batch(math_states, math_transforms, n_jobs=-1)
            bf_m = compute_layerwise_ph_batch(bbh_states, math_transforms, n_jobs=-1)
            print(f"    [MATH-PCA] MATH: {mf_m.shape}  BBH: {bf_m.shape}", flush=True)
            math2bbh = transfer_test(mf_m, math_labels, bf_m, bbh_labels, f"MATH→BBH ({method})")

            # BBH → MATH: fit PCA on BBH, apply to both.
            bbh_transforms = fit_per_layer_pca(bbh_states, np.arange(len(bbh_states)))
            mf_b = compute_layerwise_ph_batch(math_states, bbh_transforms, n_jobs=-1)
            bf_b = compute_layerwise_ph_batch(bbh_states, bbh_transforms, n_jobs=-1)
            print(f"    [BBH-PCA]  MATH: {mf_b.shape}  BBH: {bf_b.shape}", flush=True)
            bbh2math = transfer_test(bf_b, bbh_labels, mf_b, math_labels, f"BBH→MATH ({method})")

            transfer_results[method] = {
                "MATH_to_BBH_auroc": math2bbh,
                "BBH_to_MATH_auroc": bbh2math,
                "n_features": int(mf_m.shape[1]),
                "compute_time": round(time.time() - t0, 1),
            }

            # D12: within-domain baseline uses PCA fit on the 80% TRAIN split only.
            mtr_transforms = fit_per_layer_pca(math_states, mtr)
            mf_w = compute_layerwise_ph_batch(math_states, mtr_transforms, n_jobs=-1)
            within_math = transfer_test(mf_w[mtr], math_labels[mtr], mf_w[mho], math_labels[mho], f"MATH within ({method})")

            btr_transforms = fit_per_layer_pca(bbh_states, btr)
            bf_w = compute_layerwise_ph_batch(bbh_states, btr_transforms, n_jobs=-1)
            within_bbh = transfer_test(bf_w[btr], bbh_labels[btr], bf_w[bho], bbh_labels[bho], f"BBH within ({method})")

            within_domain[method] = {"MATH_within": within_math, "BBH_within": within_bbh}
        else:
            # CoE / D2H-lite: features don't depend on a domain-fit PCA, compute once.
            math_feats = compute_features(math_states, method)
            bbh_feats = compute_features(bbh_states, method)
            print(f"    MATH feats: {math_feats.shape}  BBH feats: {bbh_feats.shape}", flush=True)

            math2bbh = transfer_test(math_feats, math_labels, bbh_feats, bbh_labels, f"MATH→BBH ({method})")
            bbh2math = transfer_test(bbh_feats, bbh_labels, math_feats, math_labels, f"BBH→MATH ({method})")

            transfer_results[method] = {
                "MATH_to_BBH_auroc": math2bbh,
                "BBH_to_MATH_auroc": bbh2math,
                "n_features": int(math_feats.shape[1]),
                "compute_time": round(time.time() - t0, 1),
            }

            within_math = transfer_test(math_feats[mtr], math_labels[mtr], math_feats[mho], math_labels[mho], f"MATH within ({method})")
            within_bbh = transfer_test(bbh_feats[btr], bbh_labels[btr], bbh_feats[bho], bbh_labels[bho], f"BBH within ({method})")
            within_domain[method] = {"MATH_within": within_math, "BBH_within": within_bbh}

    # Summary
    print("\n\n  TRANSFER MATRIX (AUROC):")
    print(f"  {'method':15s}  {'MATH→BBH':>10s}  {'BBH→MATH':>10s}  {'MATH-within':>12s}  {'BBH-within':>11s}")
    for method in methods:
        r = transfer_results.get(method, {})
        w = within_domain.get(method, {})
        if "error" in r:
            print(f"  {method:15s}  [{r['error']}]")
            continue
        m2b = r.get("MATH_to_BBH_auroc", float('nan'))
        b2m = r.get("BBH_to_MATH_auroc", float('nan'))
        mw = w.get("MATH_within", float('nan'))
        bw = w.get("BBH_within", float('nan'))
        print(f"  {method:15s}  {m2b:>10.3f}  {b2m:>10.3f}  {mw:>12.3f}  {bw:>11.3f}")

    # Best transfer direction
    best_method = None
    best_transfer = 0.5
    for method, r in transfer_results.items():
        if "error" in r:
            continue
        avg = 0.5 * (r["MATH_to_BBH_auroc"] + r["BBH_to_MATH_auroc"])
        if avg > best_transfer:
            best_transfer = avg
            best_method = method

    if best_method and best_transfer > 0.65:
        verdict = f"Signal transfers: best method {best_method} avg AUROC={best_transfer:.3f} across directions."
    elif best_method:
        verdict = f"Weak transfer: best method {best_method} avg AUROC={best_transfer:.3f} (<0.65 threshold)."
    else:
        verdict = "No transfer: all methods near chance across domains."

    print(f"\n  VERDICT: {verdict}", flush=True)

    results = {
        "experiment": "exp5_cross_domain_transfer",
        "math_n": len(math_labels),
        "math_n_correct": int(math_labels.sum()),
        "bbh_n": len(bbh_labels),
        "bbh_n_correct": int(bbh_labels.sum()),
        "bbh_sizes": bbh_sizes,
        "transfer_matrix": transfer_results,
        "within_domain": within_domain,
        "best_method": best_method,
        "best_avg_transfer": best_transfer,
        "verdict": verdict,
    }
    (RESULTS / "exp5_cross_domain.json").write_text(json.dumps(results, indent=2))
    (RESULTS / "exp5.done").touch()
    print("\n  Saved: pathway9/results/exp5_cross_domain.json")


if __name__ == "__main__":
    main()
