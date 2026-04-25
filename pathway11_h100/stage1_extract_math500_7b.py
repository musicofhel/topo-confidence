#!/usr/bin/env python3
"""Stage 1: MATH-500 × Qwen2.5-7B-Instruct, all-layer per-token at 1024 tokens.

GPU REQUIRED. Run on RunPod H100.

Mirrors pathway8_layerwise/extract_math500.py exactly, with two differences:
  - MODEL_NAME -> Qwen/Qwen2.5-7B-Instruct (HIDDEN_DIM=3584)
  - Output dir -> pathway11_h100/data/math500_7b/

Output per problem: .npz with
  states: (29, n_gen, 3584) float16
  d2h_attn_entropy: (28,) float64
  text, correct (math_verify), mean_logprob

Expected accuracy: ~0.696 (matches phase6_5 FINAL_SUMMARY.md 348/500).
Estimated runtime: ~2-3 hr on H100.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathway8_layerwise.extract_math500 import (  # reuse scoring + prompt helpers
    check_correct,
    format_prompt,
    load_math500,
)
from pathway8_layerwise.extraction_utils import (
    extract_all_layers_and_attention,
    load_model,
    save_manifest,
    save_problem_checkpoint,
)
from pathway11_h100.config import (
    MATH500_7B_DATA_DIR,
    MAX_NEW_TOKENS,
    MODEL_NAME_7B,
    N_PROBLEMS,
    RESULTS_DIR,
    mark_done,
)


def main():
    MATH500_7B_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Stage 1: MATH-500 × Qwen2.5-7B-Instruct, 1024-token all-layer extraction")
    print("=" * 70)

    # Check for already-extracted problems
    done_set = set()
    for f in MATH500_7B_DATA_DIR.glob("problem_*.npz"):
        idx = int(f.stem.split("_")[1])
        done_set.add(idx)
    print(f"  {len(done_set)} problems already extracted, continuing from remainder")

    # Load 7B model
    model, tokenizer = load_model(MODEL_NAME_7B)

    # Load dataset
    problems = load_math500()
    assert len(problems) >= N_PROBLEMS, f"Expected {N_PROBLEMS} problems, got {len(problems)}"
    problems = problems[:N_PROBLEMS]

    # Extract
    manifest_entries = []
    t0 = time.time()
    n_correct = 0

    for i, prob in enumerate(problems):
        if i in done_set:
            npz_path = MATH500_7B_DATA_DIR / f"problem_{i:03d}.npz"
            data = np.load(npz_path, allow_pickle=True)
            correct = bool(data["correct"])
            n_correct += int(correct)
            manifest_entries.append({
                "idx": i,
                "unique_id": prob.get("unique_id", ""),
                "correct": correct,
                "mean_logprob": float(data["mean_logprob"]),
                "n_gen_tokens": int(data["states"].shape[1]),
            })
            continue

        prompt = format_prompt(prob["problem"], tokenizer)

        try:
            # capture_attention=False for 7B: avoids a second eager-attention
            # forward pass that isn't needed for pathway 10 v2 (DoM/steering
            # don't use D2H attn entropy). Reduces OOM risk on 7B GQA.
            result = extract_all_layers_and_attention(
                model, tokenizer, prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                capture_attention=False,
            )

            correct = check_correct(result["text"], prob["answer"])
            n_correct += int(correct)

            save_problem_checkpoint(
                output_dir=MATH500_7B_DATA_DIR,
                idx=i,
                states=result["states"],
                text=result["text"],
                correct=correct,
                mean_logprob=result["mean_logprob"],
                d2h_attn_entropy=result["d2h_attn_entropy"],
            )

            manifest_entries.append({
                "idx": i,
                "unique_id": prob.get("unique_id", ""),
                "correct": correct,
                "mean_logprob": result["mean_logprob"],
                "n_gen_tokens": result["n_gen_tokens"],
            })

        except Exception as e:
            print(f"  ERROR problem {i}: {e}")
            manifest_entries.append({
                "idx": i,
                "unique_id": prob.get("unique_id", ""),
                "correct": False,
                "error": str(e),
            })
            continue

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            done_new = sum(1 for j in range(i + 1) if j not in done_set)
            if done_new > 0:
                rate = done_new / elapsed
                remaining = sum(1 for j in range(i + 1, N_PROBLEMS) if j not in done_set)
                eta = remaining / rate if rate > 0 else 0
                acc = n_correct / (i + 1)
                print(
                    f"  {i+1}/{N_PROBLEMS}: acc={acc:.1%}, "
                    f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s"
                )

        if (i + 1) % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    total = time.time() - t0
    acc = n_correct / N_PROBLEMS
    print(f"\nDone: {total:.0f}s total, accuracy={acc:.1%} ({n_correct}/{N_PROBLEMS})")

    save_manifest(
        MATH500_7B_DATA_DIR, manifest_entries,
        model_name=MODEL_NAME_7B, benchmark="math500_7b",
    )
    print(f"Manifest saved to {MATH500_7B_DATA_DIR / 'manifest.json'}")

    n_files = len(list(MATH500_7B_DATA_DIR.glob("problem_*.npz")))
    print(f"Output: {n_files}/{N_PROBLEMS} npz files in {MATH500_7B_DATA_DIR}")

    if n_files == N_PROBLEMS:
        mark_done("stage1", RESULTS_DIR)
        print("[DONE] stage1 marker written")
    else:
        print(f"[WARN] only {n_files}/{N_PROBLEMS} complete — not marking stage1 done")


if __name__ == "__main__":
    main()
