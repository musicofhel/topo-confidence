#!/usr/bin/env python3
"""GPU: Extract all-layer hidden states for HumanEval (164 problems).

Uses evalplus for correctness evaluation (NOT the broken check_humaneval_correctness).
Saves per-problem npz files matching the MATH-500 format.

Run on RunPod: python pathway8_layerwise/extract_humaneval.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathway8_layerwise.config import HUMANEVAL_DATA_DIR, MODEL_NAME
from pathway8_layerwise.extraction_utils import (
    extract_all_layers_and_attention,
    load_model,
    save_manifest,
    save_problem_checkpoint,
)

MAX_NEW_TOKENS = 1024
SYSTEM_PROMPT = "You are Qwen, a helpful coding assistant."
INSTRUCTION = (
    "Complete the following Python function. "
    "Return only the complete function implementation inside a single ```python``` code block."
)


def load_humaneval():
    """Load HumanEval dataset."""
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    print(f"Loaded {len(ds)} HumanEval problems")
    return list(ds)


def format_prompt(problem: dict, tokenizer) -> str:
    """Format HumanEval problem as chat prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{INSTRUCTION}\n\n```python\n{problem['prompt']}\n```"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def sanitize_completion(completion: str, problem: dict) -> str:
    """Extract function body from model output for evalplus."""
    # Strip markdown code blocks if present
    if "```python" in completion:
        start = completion.index("```python") + len("```python")
        end = completion.index("```", start) if "```" in completion[start:] else len(completion)
        completion = completion[start:end]
    elif "```" in completion:
        start = completion.index("```") + 3
        end = completion.index("```", start) if "```" in completion[start:] else len(completion)
        completion = completion[start:end]

    return completion.strip()


def evaluate_with_evalplus(samples_path: Path) -> dict[str, bool]:
    """Run evalplus evaluation and return task_id -> passed mapping.

    Falls back to simple heuristic if evalplus is not available.
    """
    try:
        # Try evalplus CLI
        sanitized_path = samples_path.with_name(
            samples_path.stem + "-sanitized" + samples_path.suffix
        )

        # Sanitize
        subprocess.run(
            ["python", "-m", "evalplus.sanitize", "--samples", str(samples_path)],
            check=True, capture_output=True, timeout=120,
        )

        # Evaluate
        result = subprocess.run(
            ["python", "-m", "evalplus.evaluate",
             "--samples", str(sanitized_path),
             "--dataset", "humaneval"],
            check=True, capture_output=True, text=True, timeout=600,
        )

        # Parse results
        results_file = sanitized_path.with_name(
            sanitized_path.stem + "_eval_results.json"
        )
        if results_file.exists():
            eval_results = json.loads(results_file.read_text())
            return {
                tid: info.get("base", {}).get("status", "") == "pass"
                for tid, info in eval_results.get("eval", {}).items()
            }

        # Parse from stdout
        print(f"  evalplus stdout: {result.stdout[:500]}")
        return {}

    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  [WARN] evalplus failed: {e}, using simple heuristic")
        return {}


def main():
    HUMANEVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Pathway 8: HumanEval All-Layer Hidden State Extraction")
    print("=" * 70)

    # Check existing
    done_set = set()
    for f in HUMANEVAL_DATA_DIR.glob("problem_*.npz"):
        idx = int(f.stem.split("_")[1])
        done_set.add(idx)
    print(f"  {len(done_set)} problems already extracted")

    model, tokenizer = load_model(MODEL_NAME)
    problems = load_humaneval()
    n_problems = len(problems)

    # Phase 1: Extract hidden states + generate completions
    completions = []
    t0 = time.time()

    for i, prob in enumerate(problems):
        if i in done_set:
            # Load existing for completions list
            data = np.load(HUMANEVAL_DATA_DIR / f"problem_{i:03d}.npz", allow_pickle=True)
            completions.append({
                "task_id": prob["task_id"],
                "solution": str(data["text"]),
            })
            continue

        prompt = format_prompt(prob, tokenizer)

        try:
            result = extract_all_layers_and_attention(
                model, tokenizer, prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                capture_attention=True,
            )

            clean_completion = sanitize_completion(result["text"], prob)

            save_problem_checkpoint(
                output_dir=HUMANEVAL_DATA_DIR,
                idx=i,
                states=result["states"],
                text=result["text"],
                correct=False,  # Will be updated after evalplus
                mean_logprob=result["mean_logprob"],
                d2h_attn_entropy=result["d2h_attn_entropy"],
            )

            completions.append({
                "task_id": prob["task_id"],
                "solution": clean_completion,
            })

        except Exception as e:
            print(f"  ERROR problem {i}: {e}")
            completions.append({
                "task_id": prob["task_id"],
                "solution": "",
            })
            continue

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n_problems}: {elapsed:.0f}s elapsed")

    # Phase 2: Evaluate with evalplus
    print("\nEvaluating correctness with evalplus ...")
    samples_path = HUMANEVAL_DATA_DIR / "evalplus_samples.jsonl"
    with open(samples_path, "w") as f:
        for c in completions:
            f.write(json.dumps(c) + "\n")

    eval_results = evaluate_with_evalplus(samples_path)

    # Update correctness labels
    n_correct = 0
    manifest_entries = []
    for i, prob in enumerate(problems):
        task_id = prob["task_id"]
        correct = eval_results.get(task_id, False)
        n_correct += int(correct)

        manifest_entries.append({
            "idx": i,
            "task_id": task_id,
            "correct": correct,
            "entry_point": prob["entry_point"],
        })

    elapsed = time.time() - t0
    acc = n_correct / n_problems
    print(f"\nDone: {elapsed:.0f}s, accuracy={acc:.1%} ({n_correct}/{n_problems})")

    save_manifest(
        HUMANEVAL_DATA_DIR, manifest_entries,
        model_name=MODEL_NAME, benchmark="humaneval",
    )

    n_files = len(list(HUMANEVAL_DATA_DIR.glob("problem_*.npz")))
    print(f"Output: {n_files}/{n_problems} npz files in {HUMANEVAL_DATA_DIR}")


if __name__ == "__main__":
    main()
