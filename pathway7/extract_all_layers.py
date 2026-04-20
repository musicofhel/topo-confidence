#!/usr/bin/env python3
"""GPU: Extract per-token hidden states at ALL layers for zigzag persistence.

Saves (n_layers, n_tokens_sub, hidden_dim) per problem after subsampling
to 100 tokens. Total output: ~8.5 GB for 500 problems.

Run on RunPod: python pathway7/extract_all_layers.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Match winning_features.py constants
SUBSAMPLE = 100
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = Path(__file__).resolve().parent / "data_all_layers"
CHECKPOINT_INTERVAL = 50


def subsample_idx(n_tokens: int, n_sub: int = SUBSAMPLE) -> np.ndarray:
    if n_tokens <= n_sub:
        return np.arange(n_tokens)
    return np.linspace(0, n_tokens - 1, n_sub, dtype=int)


def load_math500_prompts() -> list[str]:
    """Load MATH-500 prompts."""
    from datasets import load_dataset
    ds = load_dataset("hendrycks/math", split="test")
    # Use first 500 problems
    prompts = [item["problem"] for item in ds][:500]
    print(f"Loaded {len(prompts)} MATH-500 prompts")
    return prompts


def extract_all_layers_single(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
) -> np.ndarray | None:
    """Extract per-token hidden states at all layers for a single problem.

    Returns (n_layers, n_tokens_sub, hidden_dim) float16 array,
    or None on failure.
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # out.hidden_states[t][layer]: (batch, seq_len_t, hidden_dim)
    # t=0: prompt, seq_len_0 = prompt_len
    # t>0: each new token, seq_len_t = 1
    n_layers = len(out.hidden_states[0])  # 29 for 1.5B (embedding + 28 layers)
    hidden_dim = out.hidden_states[0][0].shape[-1]

    # Gather per-generated-token hidden states at each layer
    # Shape: (n_gen_tokens, n_layers, hidden_dim)
    gen_tokens = len(out.hidden_states)
    all_layer_states = np.zeros((n_layers, gen_tokens, hidden_dim), dtype=np.float16)

    for t in range(gen_tokens):
        for layer in range(n_layers):
            vec = out.hidden_states[t][layer][0, -1, :].float().cpu().numpy()
            all_layer_states[layer, t, :] = vec.astype(np.float16)

    # Subsample tokens
    sub_idx = subsample_idx(gen_tokens, SUBSAMPLE)
    all_layer_states = all_layer_states[:, sub_idx, :]  # (n_layers, n_sub, hidden_dim)

    return all_layer_states


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Extract All-Layer Hidden States for Zigzag Persistence")
    print("=" * 70)

    # Load model
    print(f"\nLoading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded on {model.device}")

    # Load prompts
    prompts = load_math500_prompts()

    # Check for existing checkpoint
    done = set()
    for f in OUTPUT_DIR.glob("all_layers_*.npz"):
        # Parse problem index from filename
        stem = f.stem  # e.g., "all_layers_000"
        idx = int(stem.split("_")[-1])
        done.add(idx)
    print(f"  {len(done)} problems already extracted, continuing from remainder")

    t0 = time.time()
    for i, prompt in enumerate(prompts):
        if i in done:
            continue

        try:
            states = extract_all_layers_single(model, tokenizer, prompt)
            if states is not None:
                out_path = OUTPUT_DIR / f"all_layers_{i:03d}.npz"
                np.savez_compressed(out_path, states=states)
        except Exception as e:
            print(f"  ERROR problem {i}: {e}")
            continue

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            done_count = i + 1 - len(done)
            if done_count > 0:
                rate = done_count / elapsed
                remaining = sum(1 for j in range(i + 1, 500) if j not in done)
                eta = remaining / rate if rate > 0 else 0
                print(f"  {i+1}/500 ({elapsed:.0f}s, ETA {eta:.0f}s)")

    total = time.time() - t0
    print(f"\nDone: {total:.0f}s total")

    # Verify
    n_files = len(list(OUTPUT_DIR.glob("all_layers_*.npz")))
    print(f"Output: {n_files}/500 files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
