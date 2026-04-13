#!/usr/bin/env python3
"""Phase 0.5: Probe Sweep & Layer Selection.

Train linear probes at candidate layers to find where correctness
is most linearly separable. Select the intervention layer.

Input: phase0/activations/layer_*.npy, phase0/baseline_correct.npy
Output: phase0/probe_aurocs.json, phase0/selected_layer.txt

Runtime: ~5 minutes (no GPU needed, just sklearn).
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common import (
    CANDIDATE_LAYERS,
    CV_PARAMS_10,
    LR_PARAMS,
    PHASE0_DIR,
    get_train_holdout_indices,
    logger,
)


def main():
    print("=" * 60)
    print("Phase 0.5: Probe Sweep & Layer Selection")
    print("=" * 60)

    # Load data
    train_idx, holdout_idx = get_train_holdout_indices()
    correct = np.load(PHASE0_DIR / "baseline_correct.npy")
    y_train = correct[train_idx]

    print(f"\nTrain-400: {len(train_idx)} problems, {y_train.sum()} correct")
    print(f"Candidate layers: {CANDIDATE_LAYERS}")
    print(f"Model depth: 28 layers (0-27)\n")

    # Probe each candidate layer
    probe_aurocs = {}

    for layer in CANDIDATE_LAYERS:
        path = PHASE0_DIR / "activations" / f"layer_{layer}.npy"
        activations = np.load(path)  # (500, 1536)
        X_train = activations[train_idx]  # (400, 1536)

        # 10-fold stratified CV
        cv = StratifiedKFold(**CV_PARAMS_10)
        clf = LogisticRegression(**LR_PARAMS)
        probs = cross_val_predict(
            clf, X_train, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        auroc = roc_auc_score(y_train, probs)

        depth_pct = layer / 27 * 100
        probe_aurocs[str(layer)] = round(float(auroc), 6)
        print(f"  Layer {layer:2d} ({depth_pct:4.1f}% depth): AUROC = {auroc:.4f}")

    # Select best layer
    best_layer = max(probe_aurocs, key=probe_aurocs.get)
    best_auroc = probe_aurocs[best_layer]

    # Tiebreaker: within 0.01, prefer 50-71% depth (layers 14-20)
    preferred_range = range(14, 21)  # 14-20 inclusive
    for layer_str, auroc in sorted(probe_aurocs.items(), key=lambda x: -x[1]):
        layer_int = int(layer_str)
        if auroc >= best_auroc - 0.01 and layer_int in preferred_range:
            if layer_str != best_layer:
                print(f"\n  Tiebreaker: preferring layer {layer_str} ({auroc:.4f}) "
                      f"over layer {best_layer} ({best_auroc:.4f}) "
                      f"— within 0.01 and in preferred depth range")
                best_layer = layer_str
                best_auroc = auroc
            break

    # HALT condition
    if best_auroc < 0.55:
        print(f"\n  *** HALT: Best probe AUROC {best_auroc:.4f} < 0.55 ***")
        print("  NO-GO: Correctness is not linearly separable at any candidate layer.")
        print("  Skip to NO-GO outcome.")

        # Still save results for record
        with open(PHASE0_DIR / "probe_aurocs.json", "w") as f:
            json.dump(probe_aurocs, f, indent=2)
        with open(PHASE0_DIR / "selected_layer.txt", "w") as f:
            f.write(f"HALT: best AUROC {best_auroc:.4f} < 0.55\n")
        return

    print(f"\n  Selected layer: {best_layer} (AUROC = {best_auroc:.4f})")

    # Save artifacts
    with open(PHASE0_DIR / "probe_aurocs.json", "w") as f:
        json.dump(probe_aurocs, f, indent=2)

    with open(PHASE0_DIR / "selected_layer.txt", "w") as f:
        f.write(f"{best_layer}\n")

    print(f"\n  Saved probe_aurocs.json and selected_layer.txt to {PHASE0_DIR}/")
    print(f"\n  Next: Run phase1_steering_vector.py")


if __name__ == "__main__":
    main()
