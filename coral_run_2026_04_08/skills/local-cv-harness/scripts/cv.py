"""Local 50-fold CV that matches the grader's StandardScaler+LR pipeline.

Usage: python cv.py  (runs from agent's features.py)
"""
import json
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

LABELS_PATH = "/home/musicofhel/coral-tasks/topo-auroc/eval/data/labels.json"


def run_cv(X, y, seed=42, n_splits=50):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(
                class_weight="balanced", max_iter=1000
            )),
        ])
        pipe.fit(X[tr], y[tr])
        preds[te] = pipe.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, preds)


def main():
    with open(LABELS_PATH) as f:
        y = np.array(json.load(f)["correct"], dtype=int)

    traj_data = np.load(
        os.path.expanduser(
            "~/topo-confidence/data/experiment1_v2/trajectories.npz"
        ),
        allow_pickle=True,
    )
    trajectories = [traj_data[f"traj_{i}"] for i in range(500)]
    layer_states = np.load(
        os.path.expanduser(
            "~/att-docs/data/transformer/math500_hidden_states_aligned.npz"
        )
    )["layer_hidden_states"]

    # Import from current working directory
    sys.path.insert(0, os.getcwd())
    from features import extract_features

    X, names = extract_features(trajectories, layer_states)
    print(f"Features: {X.shape} ({len(names)} columns)")

    # Single seed first
    auroc = run_cv(X, y, seed=42)
    print(f"Local CV (seed=42): {auroc:.6f}")

    # Multi-seed summary
    scores = [run_cv(X, y, seed=s) for s in range(10)]
    print(
        f"10-seed mean: {np.mean(scores):.6f} "
        f"(se={np.std(scores)/np.sqrt(10):.6f})"
    )


if __name__ == "__main__":
    main()
