#!/usr/bin/env python3
"""v7_build_pod_payload.py — package cached texts for the SPEC v7 pod sessions.

The pod is stateless: everything it needs beyond HF downloads travels in
payload/. Per cell: JSONL.gz of {idx, problem(or question), text, y} where
`text` is the cached greedy generation (prompts are rebuilt pod-side from the
canonical templates so each model's tokenizer applies its own chat template).

Cells:
  qwen1.5b_math (500)  pathway8_layerwise/data/math500/
  qwen1.5b_bbh  (750)  pathway8_layerwise/data/bbh/<subset>/   (+ subset field)
  qwen7b_math   (500)  pathway11_h100/data/math500_7b/
  t5 smollm2/gemma/olmo2 (500 each)  generalization_edge/results/phase3_*.npz
plus cache/bbh_cot_prompts/*.txt (the exact 3-shot CoT prefixes).

Output: pathway11_h100/generalization_edge/pod_payload/
"""
from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
OUT = HERE / "pod_payload"

from pathway8_layerwise.extract_math500 import load_math500  # noqa: E402

BBH_SUBSETS = [
    "tracking_shuffled_objects_seven_objects",
    "logical_deduction_seven_objects",
    "web_of_lies",
]


def write_jsonl(path: Path, rows: list[dict]):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  {path.name}: {len(rows)} rows")


def main():
    OUT.mkdir(exist_ok=True)
    problems = load_math500()

    # --- MATH cells (1.5B + 7B share problem text) ---
    for cellname, src in [("qwen1.5b_math",
                           ROOT / "pathway8_layerwise/data/math500"),
                          ("qwen7b_math",
                           ROOT / "pathway11_h100/data/math500_7b")]:
        rows = []
        for i in range(500):
            d = np.load(src / f"problem_{i:03d}.npz", allow_pickle=True)
            rows.append({"idx": i, "problem": problems[i]["problem"],
                         "answer": problems[i]["answer"],
                         "text": str(d["text"]), "y": bool(d["correct"]),
                         "cached_mean_logprob": float(d["mean_logprob"])})
        write_jsonl(OUT / f"{cellname}.jsonl.gz", rows)

    # --- BBH 1.5B ---
    from datasets import load_dataset
    rows = []
    gi = 0
    for subset in BBH_SUBSETS:
        ds = load_dataset("lukaemon/bbh", subset, split="test")
        sub_dir = ROOT / "pathway8_layerwise/data/bbh" / subset
        n = len(list(sub_dir.glob("problem_*.npz")))
        for i in range(n):
            d = np.load(sub_dir / f"problem_{i:03d}.npz", allow_pickle=True)
            rows.append({"idx": gi, "subset": subset, "subset_idx": i,
                         "question": ds[i]["input"], "target": ds[i]["target"],
                         "text": str(d["text"]), "y": bool(d["correct"]),
                         "cached_mean_logprob": float(d["mean_logprob"])})
            gi += 1
    write_jsonl(OUT / "qwen1.5b_bbh.jsonl.gz", rows)

    # --- T5 held-out families (texts only; P1c) ---
    for key in ("smollm2", "gemma", "olmo2"):
        d = np.load(HERE / "results" / f"phase3_{key}.npz", allow_pickle=True)
        texts, y = d["texts"], d["y"].astype(bool)
        rows = [{"idx": i, "problem": problems[i]["problem"],
                 "answer": problems[i]["answer"],
                 "text": str(texts[i]), "y": bool(y[i]),
                 "cached_mean_logprob": float(d["mean_logprob"][i])}
                for i in range(len(y))]
        rows[0]["model"] = str(d["model"])
        write_jsonl(OUT / f"t5_{key}.jsonl.gz", rows)

    # --- BBH CoT prompts (exact generation-time prefixes) ---
    cot_src = HERE / "cache" / "bbh_cot_prompts"
    cot_dst = OUT / "bbh_cot_prompts"
    cot_dst.mkdir(exist_ok=True)
    for subset in BBH_SUBSETS:
        f = cot_src / f"{subset}.txt"
        if not f.exists():
            print(f"  WARNING: {f} missing — run v7_rescore_local.py --cell "
                  f"bbh first (it downloads/caches them)")
            continue
        shutil.copy(f, cot_dst / f.name)
        print(f"  cot prompt: {subset}")
    print(f"\npayload at {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
