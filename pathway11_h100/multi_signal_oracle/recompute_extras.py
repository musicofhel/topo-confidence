#!/usr/bin/env python3
"""Tier-1 regen for §10 Exp 2b multi-vs-single comparison metrics.

Loads oof_logreg.npz + features.npz, simulates the K=1-fallback routing policy,
and compares the multi-signal classifier accuracy to the canonical best-single-
signal policy (neg_seq_len_threshold_25pct from phase3b_realistic).

Stdout:
    best_single_acc        — read from phase3b_realistic_policies.json (committed
                             but the file's content is itself a Tier-0 readback;
                             a deeper Tier-1 would re-run phase3b_realistic.py).
    multi_vs_single_pp     — multi-signal logreg K=1-fallback overall_acc minus
                             best_single, expressed in pp.
    threshold_2pp_met      — boolean (1 / 0).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OOF = ROOT / "pathway11_h100/multi_signal_oracle/oof_logreg.npz"
FEATURES = ROOT / "pathway11_h100/multi_signal_oracle/features.npz"
PHASE3B = ROOT / "pathway11_h100/prefill_gated_compute/phase3b_realistic_policies.json"
BEST_SINGLE_KEY = "threshold_neg_seq_len_25pct"   # closest committed match
FALLBACK_BEST = 0.516                              # canonical Exp 2 reference


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="all")
    ap.parse_args()
    if not OOF.exists() or not FEATURES.exists():
        sys.exit(f"MISSING_REGEN_INPUT: {OOF} or {FEATURES}")

    oof = np.load(OOF)
    feats = np.load(FEATURES)
    y_pred = oof["y_pred"]
    k1 = feats["k1_correct"].astype(bool)
    k8 = feats["k8_correct"].astype(bool)
    n = len(y_pred)

    # K=1-fallback policy: pred==0 -> K=1; pred==1 -> K=8; pred==2 -> K=1 fallback.
    correct = np.where(y_pred == 0, k1,
              np.where(y_pred == 1, k8, k1))
    overall_acc = float(correct.mean())

    # Best-single-signal at matched compute. Read from phase3b's
    # threshold-25-percentile neg_seq_len entry if available; else fall back
    # to the canonical 0.516 anchor.
    best_single = FALLBACK_BEST
    if PHASE3B.exists():
        d = json.loads(PHASE3B.read_text())
        sweep = d.get("threshold_neg_seq_len_sweep", [])
        for entry in sweep:
            if entry.get("tau_percentile") == 25:
                best_single = float(entry.get("overall_accuracy", FALLBACK_BEST))
                break

    gain_pp = round((overall_acc - best_single) * 100, 2)
    print(f"best_single_acc={best_single:.10f}")
    print(f"multi_vs_single_pp={gain_pp:.10f}")
    print(f"threshold_2pp_met={int(gain_pp >= 2.0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
