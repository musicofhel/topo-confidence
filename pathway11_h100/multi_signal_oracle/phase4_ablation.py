#!/usr/bin/env python3
"""Exp 2b / Phase 4: feature ablation.

Start from the 3-feature classifier {prefill_dom, prefill_lpr, seq_len}.
Drop each feature one at a time, re-train under 5-fold CV, compare:
  - macro-F1
  - policy overall_accuracy (with K=1 fallback, so coverage=1)
  - D-bucket recall (routed to K=1)

Also try:
  - Single-feature baselines (just prefill_dom; just seq_len; just prefill_lpr)
  - Extended 5-feature {+ finaltok_dom, mean_logprob}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
OUT_DIR = ROOT / "pathway11_h100/multi_signal_oracle"

SEED = 9999
N_SPLITS = 5


def cv_predict(X, y, clf_kind="logreg"):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    y_pred = np.full_like(y, -1)
    fold_f1s = []
    for tr, te in skf.split(X, y):
        if clf_kind == "logreg":
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(
                solver="lbfgs", class_weight="balanced",
                max_iter=2000, random_state=SEED,
            ).fit(sc.transform(X[tr]), y[tr])
            y_pred[te] = clf.predict(sc.transform(X[te]))
        else:
            clf = RandomForestClassifier(
                n_estimators=400, class_weight="balanced",
                random_state=SEED, n_jobs=-1,
            ).fit(X[tr], y[tr])
            y_pred[te] = clf.predict(X[te])
        fold_f1s.append(f1_score(y[te], y_pred[te], average="macro"))
    return y_pred, np.mean(fold_f1s)


def simulate_k1fb(y_pred, k1, k8):
    """Coverage-1 sim: refuse falls back to K=1."""
    compute = np.where(y_pred == 0, 1, np.where(y_pred == 1, 8, 1))
    correct = np.where(
        y_pred == 0, k1.astype(int),
        np.where(y_pred == 1, k8.astype(int), k1.astype(int)),
    )
    return dict(
        compute_per_problem=float(compute.mean()),
        overall_accuracy=float(correct.mean()),
    )


def evaluate(feat_dict, feat_names, y, bkt, k1c, k8c, clf_kind="logreg"):
    X = np.stack([feat_dict[n] for n in feat_names], axis=1).astype(np.float64)
    y_pred, macro_f1 = cv_predict(X, y, clf_kind)
    sim = simulate_k1fb(y_pred, k1c, k8c)
    d_recall = float((y_pred[bkt == "D"] == 0).mean())
    c_refuse = float((y_pred[bkt == "C"] == 2).mean())
    b_recall_K8 = float((y_pred[bkt == "B"] == 1).mean())
    return dict(
        features=list(feat_names),
        macro_f1=macro_f1,
        compute_per_problem=sim["compute_per_problem"],
        overall_accuracy=sim["overall_accuracy"],
        D_recall_to_K1=d_recall,
        C_recall_to_refuse=c_refuse,
        B_recall_to_K8=b_recall_K8,
    )


def main():
    print("=" * 70)
    print("Exp 2b / Phase 4: feature ablation")
    print("=" * 70)

    feats = np.load(OUT_DIR / "features.npz", allow_pickle=True)
    feat_dict = {k: feats[k] for k in feats.files
                 if k not in ("y3", "bucket", "k1_correct", "k8_correct")}
    y = feats["y3"].astype(int)
    bkt = feats["bucket"]
    k1c = feats["k1_correct"].astype(bool)
    k8c = feats["k8_correct"].astype(bool)

    runs = []
    configs = [
        # Full set
        (["prefill_dom", "prefill_lpr", "seq_len"], "full (prefill_dom + prefill_lpr + seq_len)"),
        # Drop-one-out from full
        (["prefill_lpr", "seq_len"], "drop prefill_dom"),
        (["prefill_dom", "seq_len"], "drop prefill_lpr"),
        (["prefill_dom", "prefill_lpr"], "drop seq_len"),
        # Single-feature baselines
        (["prefill_dom"], "only prefill_dom"),
        (["prefill_lpr"], "only prefill_lpr"),
        (["seq_len"], "only seq_len"),
        # Extended
        (["prefill_dom", "prefill_lpr", "seq_len",
          "finaltok_dom", "finaltok_lpr", "mean_logprob"],
         "extended (+ finaltok_dom + finaltok_lpr + mean_logprob)"),
        # 2-feature subsets worth checking
        (["prefill_dom", "seq_len"], "prefill_dom + seq_len (no PR)"),
        (["finaltok_dom", "seq_len"], "finaltok_dom + seq_len"),
    ]

    for clf_kind in ["logreg", "rf"]:
        print(f"\n=== {clf_kind} ===")
        print(f"  {'features':<50} {'F1':>6} {'K':>6} {'over':>6} {'D→K1':>6} {'C→ref':>6} {'B→K8':>6}")
        for feat_names, label in configs:
            r = evaluate(feat_dict, feat_names, y, bkt, k1c, k8c, clf_kind)
            r["label"] = label
            r["clf"] = clf_kind
            runs.append(r)
            print(f"  {label:<50} {r['macro_f1']:>6.3f} "
                  f"{r['compute_per_problem']:>6.2f} "
                  f"{r['overall_accuracy']:>6.3f} "
                  f"{r['D_recall_to_K1']:>6.3f} "
                  f"{r['C_recall_to_refuse']:>6.3f} "
                  f"{r['B_recall_to_K8']:>6.3f}")

    with open(OUT_DIR / "phase4_ablation.json", "w") as f:
        json.dump({"runs": runs}, f, indent=2)
    print(f"\nSaved: {OUT_DIR/'phase4_ablation.json'}")

    # Summary table: how much does dropping each feature hurt?
    print("\n--- drop-one-out impact (vs full, logreg) ---")
    full_logreg = next(r for r in runs if r["label"].startswith("full") and r["clf"] == "logreg")
    for r in runs:
        if r["clf"] != "logreg" or not r["label"].startswith("drop"):
            continue
        d_over = r["overall_accuracy"] - full_logreg["overall_accuracy"]
        d_f1 = r["macro_f1"] - full_logreg["macro_f1"]
        d_drecall = r["D_recall_to_K1"] - full_logreg["D_recall_to_K1"]
        print(f"  {r['label']:<30} Δoverall={d_over:+.3f} ΔmacroF1={d_f1:+.3f} ΔD-recall={d_drecall:+.3f}")


if __name__ == "__main__":
    main()
