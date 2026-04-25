#!/usr/bin/env python3
"""Exp 2b / Phase 2: 5-fold CV classifier — {prefill DoM, prefill local PR, seq_len} → 3-class.

3 classes:
  0 = use K=1   (A + D)   — 243 problems, 48.6%
  1 = use K=8   (B)       —  68 problems, 13.6%
  2 = refuse    (C)       — 189 problems, 37.8%

Trains:
  - Logistic regression (balanced class weights, StandardScaler features)
  - Random forest (class_weight balanced, 400 trees)

Reports:
  - Per-class precision/recall/F1, confusion matrix, macro-F1
  - Out-of-fold predictions (used by Phase 3 for policy simulation)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
OUT_DIR = ROOT / "pathway11_h100/multi_signal_oracle"

SEED = 9999
N_SPLITS = 5
CLASS_NAMES = ["K1_AD", "K8_B", "refuse_C"]
FEATURE_SET = ["prefill_dom", "prefill_lpr", "seq_len"]  # the 3 the user specified


def build_X(feat: dict, feature_names):
    return np.stack([feat[n] for n in feature_names], axis=1).astype(np.float64)


def cv_predict(X: np.ndarray, y: np.ndarray, clf_kind: str, seed: int = SEED):
    """Return per-sample OOF predictions + fold-mean macro-F1."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    y_pred = np.full_like(y, -1)
    y_proba = np.zeros((len(y), 3))
    fold_f1s = []
    for fold_i, (tr, te) in enumerate(skf.split(X, y)):
        if clf_kind == "logreg":
            scaler = StandardScaler().fit(X[tr])
            Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
            clf = LogisticRegression(
                solver="lbfgs", class_weight="balanced",
                max_iter=2000, random_state=seed,
            ).fit(Xtr, y[tr])
            y_pred[te] = clf.predict(Xte)
            y_proba[te] = clf.predict_proba(Xte)
        elif clf_kind == "rf":
            clf = RandomForestClassifier(
                n_estimators=400, class_weight="balanced",
                random_state=seed, n_jobs=-1,
            ).fit(X[tr], y[tr])
            y_pred[te] = clf.predict(X[te])
            y_proba[te] = clf.predict_proba(X[te])
        else:
            raise ValueError(clf_kind)
        fold_f1s.append(f1_score(y[te], y_pred[te], average="macro"))
    return y_pred, y_proba, fold_f1s


def per_class_report(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    out = {}
    for i, name in enumerate(CLASS_NAMES):
        out[name] = dict(
            precision=float(p[i]), recall=float(r[i]),
            f1=float(f[i]), support=int(s[i]),
        )
    out["macro_f1"] = float(np.mean(f))
    out["accuracy"] = float((y_true == y_pred).mean())
    out["confusion_matrix"] = cm.astype(int).tolist()
    out["confusion_matrix_labels"] = CLASS_NAMES
    return out


def print_confusion(cm, labels):
    print(f"  conf matrix (rows=true, cols=pred):")
    print(f"    {'':>10}", " ".join(f"{l:>9}" for l in labels))
    for i, lab in enumerate(labels):
        print(f"    {lab:>10}", " ".join(f"{int(v):>9}" for v in cm[i]))


def main():
    print("=" * 70)
    print("Exp 2b / Phase 2: 5-fold CV classifier for 3-class oracle target")
    print("=" * 70)

    feats = np.load(OUT_DIR / "features.npz", allow_pickle=True)
    y = feats["y3"].astype(int)
    bkt = feats["bucket"]
    feat_dict = {k: feats[k] for k in feats.files}

    X = build_X(feat_dict, FEATURE_SET)
    print(f"X={X.shape}, features={FEATURE_SET}")
    print(f"y classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    results = {"features": FEATURE_SET, "n_samples": int(len(y)), "classes": CLASS_NAMES}

    for clf_kind in ["logreg", "rf"]:
        print(f"\n--- {clf_kind} ---")
        y_pred, y_proba, fold_f1s = cv_predict(X, y, clf_kind)
        print(f"fold macro-F1s: {[round(f, 4) for f in fold_f1s]} "
              f"mean={np.mean(fold_f1s):.4f} std={np.std(fold_f1s):.4f}")
        rep = per_class_report(y, y_pred)
        for cname in CLASS_NAMES:
            r = rep[cname]
            print(f"  {cname}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} (n={r['support']})")
        print(f"  overall acc: {rep['accuracy']:.3f}, macro-F1: {rep['macro_f1']:.3f}")
        print_confusion(rep["confusion_matrix"], CLASS_NAMES)

        # D-bucket recall: fraction of D correctly routed to K=1 (class 0)
        d_mask = bkt == "D"
        d_routed_K1 = (y_pred[d_mask] == 0).mean() if d_mask.any() else 0.0
        rep["D_bucket_recall_to_K1"] = float(d_routed_K1)
        print(f"  D-bucket → K=1 routing recall: {d_routed_K1:.3f} "
              f"({int((y_pred[d_mask] == 0).sum())}/{int(d_mask.sum())} D-problems routed correctly)")

        # Save OOF preds per classifier
        np.savez_compressed(
            OUT_DIR / f"oof_{clf_kind}.npz",
            y_pred=y_pred, y_proba=y_proba, y_true=y,
        )
        rep["fold_macro_f1_mean"] = float(np.mean(fold_f1s))
        rep["fold_macro_f1_std"] = float(np.std(fold_f1s))
        results[clf_kind] = rep

    # Baseline: majority class (K=1 for all)
    y_majority = np.zeros_like(y)
    rep_m = per_class_report(y, y_majority)
    rep_m["D_bucket_recall_to_K1"] = 1.0  # all routed to K=1
    results["baseline_majority"] = rep_m
    print(f"\n--- baseline (all K=1) ---")
    print(f"  acc={rep_m['accuracy']:.3f}, macro-F1={rep_m['macro_f1']:.3f}")
    print(f"  D-bucket recall (trivially 1.0 because everything goes to K=1)")

    with open(OUT_DIR / "phase2_classifier.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_DIR/'phase2_classifier.json'}")


if __name__ == "__main__":
    main()
