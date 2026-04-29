#!/usr/bin/env python3
"""Extract output logits for MATH-500 problems.

Standalone script that only does the generation pass (no hidden state extraction)
to collect output logits for entropy/probability baselines.

Output:
  - output_logits.pkl: list of (gen_len, vocab_size) numpy arrays (gitignored)
  - output_entropy_scores.json: per-problem entropy, max_prob, first_prob (committed)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem. Give your final answer after '####'.\n\n"
    "Problem: {problem}\n\nSolution:"
)

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_math500():
    """Load MATH-500 from HuggingFace datasets."""
    logger.info("Loading MATH-500 from HuggingFace datasets...")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return list(ds)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def score_logits_on_gpu(logits_tensor: torch.Tensor) -> dict:
    """Compute entropy/maxprob/firstprob from logits on GPU, return scalars.

    Args:
        logits_tensor: (gen_len, vocab_size) tensor on GPU.

    Returns:
        dict with entropy, max_token_prob, first_token_prob, gen_len.
    """
    if logits_tensor.numel() == 0:
        return {"entropy": 0.0, "max_token_prob": 1.0, "first_token_prob": 1.0, "gen_len": 0}

    # Compute softmax on GPU (much faster than numpy for 152K vocab)
    probs = torch.softmax(logits_tensor.float(), dim=-1)

    # Mean per-token entropy
    token_entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
    mean_entropy = float(token_entropy.mean().item())

    # Mean max token probability
    max_probs = probs.max(dim=-1).values
    mean_max_prob = float(max_probs.mean().item())

    # First token probability
    first_prob = float(probs[0].max().item())

    return {
        "entropy": mean_entropy,
        "max_token_prob": mean_max_prob,
        "first_token_prob": first_prob,
        "gen_len": int(logits_tensor.shape[0]),
    }


@torch.no_grad()
def extract_scores(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
    max_length: int = 512,
) -> list[dict]:
    """Generate text and compute entropy/prob scores on-the-fly.

    Processes one prompt at a time, computes scores on GPU, and discards
    the raw logits immediately. This avoids OOM for large vocab models
    (Qwen2.5 has 152K vocab → ~150MB per problem in float32).
    """
    all_scores = []

    for i, prompt in enumerate(prompts):
        if i % 25 == 0:
            logger.info("Generating %d/%d...", i, len(prompts))

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        if outputs.scores:
            # scores is a tuple of (gen_len,) tensors, each (1, vocab_size)
            logits = torch.stack(outputs.scores, dim=0).squeeze(1)  # (gen_len, vocab_size)
            score = score_logits_on_gpu(logits)
        else:
            score = {"entropy": 0.0, "max_token_prob": 1.0, "first_token_prob": 1.0, "gen_len": 0}

        all_scores.append(score)

        # Explicitly free GPU memory
        del outputs
        if i % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_scores


def main():
    parser = argparse.ArgumentParser(description="Extract output logits for MATH-500")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-dir", default="data/experiment1_v2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = output_dir / "output_entropy_scores.json"

    # Check if already done
    if scores_path.exists():
        logger.info("Scores already exist at %s. Delete to re-extract.", scores_path)
        with open(scores_path) as f:
            existing = json.load(f)
        logger.info("Existing scores: %d problems", len(existing["scores"]))
        return

    # Load problems
    problems = load_math500()
    prompts = []
    for p in problems:
        prompt_text = p.get("problem", p.get("question", ""))
        prompts.append(MATH_PROMPT_TEMPLATE.format(problem=prompt_text))
    logger.info("Loaded %d prompts", len(prompts))

    # Load model
    device = resolve_device(args.device)
    dtype = DTYPE_MAP[args.dtype]
    logger.info("Loading %s on %s (%s)...", args.model, device, args.dtype)
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    load_time = time.perf_counter() - t0
    logger.info("Model loaded in %.1fs", load_time)

    # Extract scores on-the-fly (no raw logits stored — avoids OOM)
    # Qwen2.5 vocab=152K → ~150MB/problem in float32. For 500 problems that's
    # ~75GB, so we compute entropy/maxprob/firstprob on GPU and discard immediately.
    logger.info("Extracting scores (max_new_tokens=%d)...", args.max_new_tokens)
    t0 = time.perf_counter()
    scores = extract_scores(
        model, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
    )
    extract_time = time.perf_counter() - t0
    logger.info("Extraction complete in %.1fs (%.2fs/problem)", extract_time, extract_time / len(prompts))

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scores_data = {
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "n_problems": len(scores),
        "extraction_time_s": round(extract_time, 1),
        "summary": {
            "mean_entropy": round(float(np.mean([s["entropy"] for s in scores])), 4),
            "std_entropy": round(float(np.std([s["entropy"] for s in scores])), 4),
            "mean_max_token_prob": round(float(np.mean([s["max_token_prob"] for s in scores])), 4),
            "mean_first_token_prob": round(float(np.mean([s["first_token_prob"] for s in scores])), 4),
            "mean_gen_len": round(float(np.mean([s["gen_len"] for s in scores])), 1),
        },
        "scores": scores,
    }

    with open(scores_path, "w") as f:
        json.dump(scores_data, f, indent=2)
    logger.info("Scores saved to %s", scores_path)

    # Summary
    print("\n" + "=" * 60)
    print("Output Logit Extraction Complete")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Problems: {len(scores)}")
    print(f"Extraction time: {extract_time:.1f}s ({extract_time/len(scores):.2f}s/problem)")
    print(f"Mean entropy: {scores_data['summary']['mean_entropy']:.4f}")
    print(f"Mean max token prob: {scores_data['summary']['mean_max_token_prob']:.4f}")
    print(f"Mean first token prob: {scores_data['summary']['mean_first_token_prob']:.4f}")
    print(f"Mean gen length: {scores_data['summary']['mean_gen_len']:.1f} tokens")
    print(f"Scores: {scores_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
