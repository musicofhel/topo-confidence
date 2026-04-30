#!/usr/bin/env python3
"""Tier-1 regen for §10 multi-signal oracle (Exp 2b) — logreg + RF macro-F1,
overall acc, D-recall.

Loads `oof_logreg.npz` and `oof_rf.npz` (gitignored 5-fold OOF predictions
{y_pred, y_proba, y_true}). Recomputes:
    {logreg,rf}_macro_f1, {logreg,rf}_overall_acc, {logreg,rf}_D_recall_to_K1

Stdout: key=value lines.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[2]
LOGREG = ROOT / "pathway11_h100/multi_signal_oracle/oof_logreg.npz"
RF = ROOT / "pathway11_h100/multi_signal_oracle/oof_rf.npz"
FEATURES = ROOT / "pathway11_h100/multi_signal_oracle/features.npz"


def emit(name: str, npz_path: Path) -> None:
    if not npz_path.exists():
        sys.exit(f"MISSING_REGEN_INPUT: {npz_path}")
    d = np.load(npz_path)
    yt = d["y_true"]
    yp = d["y_pred"]
    print(f"{name}_macro_f1={float(f1_score(yt, yp, average='macro')):.10f}")
    print(f"{name}_overall_acc={float(accuracy_score(yt, yp)):.10f}")

    # D-recall to K=1: what fraction of the D-bucket (pathological — K=1 right but K=8 wrong)
    # does the classifier route to "K=1" (class 0)? Per phase2_classifier.py, y3 is the
    # 3-way label {0:K=1-suffices (A∪D), 1:K=8-helps (B), 2:never-right (C)}, AND
    # `bucket` from features.npz tells us the original A/B/C/D split. D_recall_to_K1 =
    # (n D-bucket items predicted as class 0) / (n D-bucket items).
    if not FEATURES.exists():
        sys.exit(f"MISSING_REGEN_INPUT: {FEATURES}")
    feats = np.load(FEATURES, allow_pickle=False)
    bucket = feats["bucket"]
    is_D = bucket == "D"
    if is_D.sum() == 0:
        print(f"{name}_D_recall_to_K1=0.0")
    else:
        recall = float((yp[is_D] == 0).sum() / is_D.sum())
        print(f"{name}_D_recall_to_K1={recall:.10f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="all")
    ap.parse_args()
    emit("logreg", LOGREG)
    emit("rf", RF)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
