#!/usr/bin/env python3
"""GPU: Extract per-token hidden states at ALL layers + D2H attention for MATH-500.

Saves per-problem npz files with:
  - states: (29, n_gen, 1536) float16
  - d2h_attn_entropy: (28,) float64
  - text, correct, mean_logprob

Uses math-verify for answer equivalence checking (not regex).
Checkpoints per problem; resume-safe.

Run on RunPod: python pathway8_layerwise/extract_math500.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathway8_layerwise.config import (
    MATH500_DATA_DIR,
    MODEL_NAME,
    N_PROBLEMS,
)
from pathway8_layerwise.extraction_utils import (
    load_model,
    extract_all_layers_and_attention,
    save_problem_checkpoint,
    save_manifest,
)

MAX_NEW_TOKENS = 1024  # Match Phase 6.5 deconfounded settings
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def load_math500() -> list[dict]:
    """Load MATH-500 from HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = []
    for item in ds:
        problems.append({
            "problem": item["problem"],
            "solution": item["solution"],
            "answer": item["answer"],
            "subject": item.get("subject", ""),
            "level": item.get("level", ""),
            "unique_id": item.get("unique_id", ""),
        })
    print(f"Loaded {len(problems)} MATH-500 problems")
    return problems


def check_correct(predicted_text: str, ground_truth: str) -> bool:
    """Check correctness using math-verify for LaTeX equivalence."""
    try:
        from math_verify import parse, verify

        pred_parsed = parse(f"${_extract_boxed(predicted_text)}$")
        gt_parsed = parse(f"${ground_truth}$")
        return bool(verify(gt_parsed, pred_parsed))
    except Exception:
        # Fallback: simple string comparison
        pred = _extract_boxed(predicted_text).strip()
        gt = ground_truth.strip()
        return pred == gt


def _extract_boxed(text: str) -> str:
    """Extract content from \\boxed{...} with balanced brace matching."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        # Try last number as fallback
        import re
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        return numbers[-1] if numbers else text.strip()

    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1]


def format_prompt(problem_text: str, tokenizer) -> str:
    """Format MATH-500 problem as chat prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main():
    MATH500_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Pathway 8: MATH-500 All-Layer Hidden State Extraction")
    print("=" * 70)

    # Check for already-extracted problems
    done_set = set()
    for f in MATH500_DATA_DIR.glob("problem_*.npz"):
        idx = int(f.stem.split("_")[1])
        done_set.add(idx)
    print(f"  {len(done_set)} problems already extracted, continuing from remainder")

    # Load model
    model, tokenizer = load_model(MODEL_NAME)

    # Load dataset
    problems = load_math500()
    assert len(problems) >= N_PROBLEMS, f"Expected {N_PROBLEMS} problems, got {len(problems)}"
    problems = problems[:N_PROBLEMS]

    # Extract
    manifest_entries = []
    t0 = time.time()
    n_correct = 0

    for i, prob in enumerate(problems):
        # Load existing result for manifest if already done
        if i in done_set:
            npz_path = MATH500_DATA_DIR / f"problem_{i:03d}.npz"
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
            result = extract_all_layers_and_attention(
                model, tokenizer, prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                capture_attention=True,
            )

            correct = check_correct(result["text"], prob["answer"])
            n_correct += int(correct)

            save_problem_checkpoint(
                output_dir=MATH500_DATA_DIR,
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

        # Periodic memory cleanup
        if (i + 1) % 50 == 0:
            import gc
            gc.collect()
            import torch
            torch.cuda.empty_cache()

    total = time.time() - t0
    acc = n_correct / N_PROBLEMS
    print(f"\nDone: {total:.0f}s total, accuracy={acc:.1%} ({n_correct}/{N_PROBLEMS})")

    # Save manifest
    save_manifest(
        MATH500_DATA_DIR, manifest_entries,
        model_name=MODEL_NAME, benchmark="math500",
    )
    print(f"Manifest saved to {MATH500_DATA_DIR / 'manifest.json'}")

    # Verify
    n_files = len(list(MATH500_DATA_DIR.glob("problem_*.npz")))
    print(f"Output: {n_files}/{N_PROBLEMS} npz files in {MATH500_DATA_DIR}")


if __name__ == "__main__":
    main()
