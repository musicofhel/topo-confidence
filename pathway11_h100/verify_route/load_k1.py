#!/usr/bin/env python3
"""S1 — K=1 greedy GATE (CPU, no generation).

The K=1 greedy answers are ALREADY cached in pathway8_layerwise/data/math500/
problem_XXX.npz (fields: text, correct, states[29,n_gen,1536], mean_logprob).
Verified at impl: 500/500 non-empty text, mean(correct)=0.4860=243/500 == published 48.6%.

So this stage is a cheap cache-load that:
  - emits k1_greedy.json  ({idx, text, correct} per problem, in index order)
  - asserts acc in [47%, 50%]  (the reproduction gate; outside => halt, plumbing bug)
  - also re-emits the prefill L19 DoM OOF score (the C_infer baseline) for S5 convenience,
    pulled from prefill_gated_compute/phase2_prefill_dom.npz if present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
STAGE2_DIR = ROOT / "pathway8_layerwise/data/math500"
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "k1_greedy.json"
N = 500


def main() -> int:
    rows = []
    correct = np.zeros(N, dtype=bool)
    missing = []
    for i in range(N):
        fp = STAGE2_DIR / f"problem_{i:03d}.npz"
        if not fp.exists():
            missing.append(i)
            continue
        with np.load(fp, allow_pickle=True) as d:
            txt = str(d["text"])
            c = bool(d["correct"])
        correct[i] = c
        rows.append({"idx": i, "text": txt, "correct": c})

    if missing:
        print(f"  WARNING: {len(missing)} missing stage2 files: {missing[:10]}")

    n = len(rows)
    acc = correct[: ].mean() if n == N else np.mean([r["correct"] for r in rows])
    acc = float(np.mean([r["correct"] for r in rows]))
    print(f"  loaded {n}/{N} K=1 greedy answers | accuracy = {acc:.4f} ({sum(r['correct'] for r in rows)}/{n})")

    OUT.write_text(json.dumps({"n": n, "accuracy": acc, "rows": rows}, ensure_ascii=False))
    print(f"  wrote {OUT}")

    # GATE
    if not (0.47 <= acc <= 0.50):
        print(f"  GATE FAILED: K=1 greedy acc {acc:.4f} outside [0.47, 0.50] — plumbing bug, halting.")
        return 3
    print("  GATE PASSED (acc in [47%,50%], reproduces published 48.6%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
