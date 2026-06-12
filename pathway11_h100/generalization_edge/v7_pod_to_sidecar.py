#!/usr/bin/env python3
"""v7_pod_to_sidecar.py — convert pod rescore JSON to the local sidecar npz.

The pod (v7_runpod_arms.py --task rescore) emits pod_results/
v7_rescore_<model>_<cell>.json; v7_phase1a.py expects results/
v7_rescore_<model>_<cell>.npz with the v7_rescore_local.py key set.
Same numbers, different container. Refuses alignment_corr < 0.90 (the
pod task prints but does not enforce the pre-registered gate).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def convert(name: str) -> None:
    src = HERE / "pod_results" / f"v7_rescore_{name}.json"
    d = json.loads(src.read_text())
    r = float(d["alignment_corr"])
    if r < 0.90:
        sys.exit(f"FAIL {name}: alignment corr {r:.4f} < 0.90 "
                 f"(prompt reconstruction drift)")
    out = HERE / "results" / f"v7_rescore_{name}.npz"
    np.savez(out,
             min_lp=np.array(d["min_lp"], dtype=np.float64),
             p10_lp=np.array(d["p10_lp"], dtype=np.float64),
             mean_entropy=np.array(d["mean_entropy"], dtype=np.float64),
             mean_top2_margin=np.array(d["mean_top2_margin"],
                                       dtype=np.float64),
             ans_span_lp=np.array(d["ans_span_lp"], dtype=np.float64),
             mean_lp_rescored=np.array(d["mean_lp_rescored"],
                                       dtype=np.float64),
             n_gen=np.array(d["n_gen"], dtype=np.int64),
             n_prompt=np.array(d["n_prompt"], dtype=np.int64),
             y=np.array(d["y"], dtype=bool),
             alignment_corr=r)
    print(f"{name}: n={len(d['y'])} corr={r:.4f} -> {out.name}")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["qwen1.5b_bbh", "qwen7b_math"]):
        convert(name)
