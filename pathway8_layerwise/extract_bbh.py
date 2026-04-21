#!/usr/bin/env python3
"""GPU: Extract all-layer hidden states for BBH (3 subsets × 250 problems).

Uses lukaemon/bbh dataset with 3-shot CoT prompts from Suzgun's GitHub.
Saves per-problem npz files matching the MATH-500 format.

Run on RunPod: python pathway8_layerwise/extract_bbh.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathway8_layerwise.config import BBH_DATA_DIR, MODEL_NAME
from pathway8_layerwise.extraction_utils import (
    extract_all_layers_and_attention,
    load_model,
    save_manifest,
    save_problem_checkpoint,
)

MAX_NEW_TOKENS = 512

BBH_SUBSETS = [
    "tracking_shuffled_objects_seven_objects",
    "logical_deduction_seven_objects",
    "web_of_lies",
]

COT_PROMPTS_DIR = Path("/opt/bbh-prompts")
COT_PROMPTS_URL = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/cot-prompts/{subset}.txt"

# Tiered answer extraction regex (tested)
_ANSWER_PATS = [
    re.compile(r"(?:the\s+answer\s+is)\s*\(([A-Z])\)", re.I),
    re.compile(r"answer\s*(?:is|:)\s*\(([A-Z])\)", re.I),
    re.compile(r"(?:the\s+answer\s+is)\s*([A-Z])\b", re.I),
    re.compile(r"answer\s*(?:is|:)\s*([A-Z])\b", re.I),
    re.compile(r"\\boxed\{\(?([A-Z])\)?\}", re.I),
    re.compile(r"\(([A-Z])\)\s*\.?\s*$", re.M),
    # Yes/No for web_of_lies
    re.compile(r"(?:the\s+answer\s+is)\s*(Yes|No)\b", re.I),
    re.compile(r"answer\s*(?:is|:)\s*(Yes|No)\b", re.I),
]


def load_cot_prompt(subset: str) -> str:
    """Load 3-shot CoT prompt for a BBH subset."""
    local_path = COT_PROMPTS_DIR / f"{subset}.txt"
    if local_path.exists():
        return local_path.read_text()

    # Fallback: download
    url = COT_PROMPTS_URL.format(subset=subset)
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.read().decode()
    except Exception as e:
        print(f"  [WARN] Could not load CoT prompt for {subset}: {e}")
        return ""


def extract_answer(response: str) -> str:
    """Extract answer from BBH CoT response using tiered regex."""
    for pat in _ANSWER_PATS:
        matches = pat.findall(response)
        if matches:
            ans = matches[-1].strip()
            # Normalize: letter answers get parens, Yes/No stay as-is
            if len(ans) == 1 and ans.isalpha() and ans.upper() != "N":
                return f"({ans.upper()})"
            return ans.capitalize() if ans.lower() in ("yes", "no") else ans
    return ""


def format_prompt(question: str, cot_prompt: str, tokenizer) -> str:
    """Format BBH question with 3-shot CoT as chat prompt."""
    user_content = f"{cot_prompt}\n\nQ: {question}\nA: Let's think step by step."
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main():
    BBH_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Pathway 8: BBH All-Layer Hidden State Extraction")
    print("=" * 70)

    model, tokenizer = load_model(MODEL_NAME)

    all_manifest_entries = []
    global_idx = 0
    t_total = time.time()

    for subset in BBH_SUBSETS:
        subset_dir = BBH_DATA_DIR / subset
        subset_dir.mkdir(parents=True, exist_ok=True)

        # Check existing
        done_set = set()
        for f in subset_dir.glob("problem_*.npz"):
            idx = int(f.stem.split("_")[1])
            done_set.add(idx)

        # Load dataset
        from datasets import load_dataset
        ds = load_dataset("lukaemon/bbh", subset, split="test")
        print(f"\n  {subset}: {len(ds)} problems ({len(done_set)} already done)")

        # Load CoT prompt
        cot_prompt = load_cot_prompt(subset)
        if not cot_prompt:
            print(f"  [WARN] No CoT prompt for {subset}, skipping")
            continue

        n_correct = 0
        t0 = time.time()

        for i, item in enumerate(ds):
            if i in done_set:
                data = np.load(subset_dir / f"problem_{i:03d}.npz", allow_pickle=True)
                correct = bool(data["correct"])
                n_correct += int(correct)
                all_manifest_entries.append({
                    "idx": global_idx,
                    "subset": subset,
                    "subset_idx": i,
                    "correct": correct,
                })
                global_idx += 1
                continue

            prompt = format_prompt(item["input"], cot_prompt, tokenizer)

            try:
                result = extract_all_layers_and_attention(
                    model, tokenizer, prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    capture_attention=True,
                )

                pred = extract_answer(result["text"])
                target = item["target"].strip()
                # Normalize target format
                if len(target) == 1 and target.isalpha():
                    target = f"({target.upper()})"

                correct = pred == target
                n_correct += int(correct)

                save_problem_checkpoint(
                    output_dir=subset_dir,
                    idx=i,
                    states=result["states"],
                    text=result["text"],
                    correct=correct,
                    mean_logprob=result["mean_logprob"],
                    d2h_attn_entropy=result["d2h_attn_entropy"],
                )

                all_manifest_entries.append({
                    "idx": global_idx,
                    "subset": subset,
                    "subset_idx": i,
                    "correct": correct,
                    "pred": pred,
                    "target": target,
                })

            except Exception as e:
                print(f"    ERROR {subset}[{i}]: {e}")
                all_manifest_entries.append({
                    "idx": global_idx,
                    "subset": subset,
                    "subset_idx": i,
                    "correct": False,
                    "error": str(e),
                })

            global_idx += 1

            if (i + 1) % 50 == 0:
                acc = n_correct / (i + 1)
                print(f"    {i+1}/{len(ds)}: acc={acc:.1%}, {time.time()-t0:.0f}s")

        acc = n_correct / len(ds)
        print(f"    Done: {acc:.1%} ({n_correct}/{len(ds)}), {time.time()-t0:.0f}s")

    # Save combined manifest
    save_manifest(
        BBH_DATA_DIR, all_manifest_entries,
        model_name=MODEL_NAME, benchmark="bbh",
    )

    total = time.time() - t_total
    print(f"\nAll BBH done: {total:.0f}s total")
    print(f"Manifest saved to {BBH_DATA_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
