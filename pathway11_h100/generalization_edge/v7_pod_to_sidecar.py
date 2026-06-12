#!/usr/bin/env python3
"""v7_pod_to_sidecar.py — convert pod rescore JSON to the local sidecar npz.

The pod (v7_runpod_arms.py --task rescore) emits pod_results/
v7_rescore_<model>_<cell>.json; v7_phase1a.py expects results/
v7_rescore_<model>_<cell>.npz with the v7_rescore_local.py key set.
Same numbers, different container. Refuses alignment_corr < 0.90 (the
pod task prints but does not enforce the pre-registered gate).

Deviation D-4: families whose tokenizer does not round-trip decode->encode
(SmolLM2, OLMo-2) cannot be teacher-force rescored from cached texts; the
pod's `genscore` task regenerates greedily and captures features at
generation time. convert_genscore() validates faithfulness by text
equality instead: rows whose regenerated text differs from the pinned
cache are NaN'd (impute_median downstream; finite_fraction reported), and
the alignment gate becomes corr(gen-time mean logprob, cached
mean_logprob) on the matched subset.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FEATS = ("min_lp", "p10_lp", "mean_entropy", "mean_top2_margin",
         "ans_span_lp", "mean_lp_rescored")


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


def convert_genscore(name: str) -> None:
    src = HERE / "pod_results" / f"v7_genscore_{name}.json.gz"
    with gzip.open(src, "rt") as fh:
        d = json.load(fh)
    match = np.array(d["text_match"], dtype=bool)
    gen_lp = np.array(d["mean_lp_rescored"], dtype=np.float64)
    cached_lp = np.array(d["cached_mean_logprob"], dtype=np.float64)
    m = match & np.isfinite(gen_lp)
    if m.sum() < 30:
        sys.exit(f"FAIL {name}: only {int(m.sum())} text-matched rows — "
                 f"genscore regeneration drifted from the pinned cache")
    r = float(np.corrcoef(gen_lp[m], cached_lp[m])[0, 1])
    if r < 0.90:
        sys.exit(f"FAIL {name}: matched-subset alignment corr {r:.4f} "
                 f"< 0.90")
    arrs = {}
    for k in FEATS:
        x = np.array(d[k], dtype=np.float64)
        x[~match] = np.nan
        arrs[k] = x
    out = HERE / "results" / f"v7_genscore_{name}.npz"
    np.savez(out, **arrs,
             n_gen=np.array(d["n_gen"], dtype=np.int64),
             n_prompt=np.array(d["n_prompt"], dtype=np.int64),
             text_match=match,
             text_match_frac=float(d["text_match_frac"]),
             y=np.array(d["y"], dtype=bool),
             alignment_corr=r)
    print(f"{name}: n={len(d['y'])} match={match.mean():.3f} "
          f"corr(matched)={r:.4f} -> {out.name}")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["qwen1.5b_bbh", "qwen7b_math"]):
        if name.startswith("genscore:"):
            convert_genscore(name.split(":", 1)[1])
        else:
            convert(name)
