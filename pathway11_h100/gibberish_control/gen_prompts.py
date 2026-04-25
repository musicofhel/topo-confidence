#!/usr/bin/env python3
"""Generate the two prompt JSONs consumed by extract.py --prompts-file.

Produces:
  prompts_random.json  — 20 random-token sequences (decoded to strings)
  prompts_stream.json  — 20 stream-of-consciousness prompts (hand-written)

Runs on the pod so the tokenizer is already available locally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent

# Hand-written stream-of-consciousness prompts. Intentionally varied wording to
# avoid all 20 producing the exact same generation. Same chat template applied.
STREAM_PROMPTS = [
    "Write a stream of consciousness about nothing.",
    "Ramble for a while about whatever crosses your mind.",
    "Describe nothing, in detail.",
    "Free-associate without any particular goal.",
    "Let your thoughts drift and write them down as they come.",
    "Say whatever comes to mind, with no structure.",
    "Write an unedited inner monologue about the color blue.",
    "Produce a meandering reflection with no point.",
    "Pretend you are daydreaming and transcribe it.",
    "Describe the feeling of waiting for something with no subject.",
    "Write a passage with no beginning, middle, or end.",
    "Let words flow without censoring them. Start anywhere.",
    "Describe an empty room without using concrete details.",
    "Write whatever text feels right next, sentence by sentence.",
    "Free-write about the concept of free-writing.",
    "Produce a disorganized essay about a non-topic.",
    "Talk to yourself in prose.",
    "Generate filler text that has no purpose.",
    "Narrate the passage of a minute in which nothing happens.",
    "Write a loose, unfocused paragraph about whatever.",
]


def generate_random_token_prompts(
    tokenizer_name: str,
    n_prompts: int,
    tokens_per_prompt: int,
    seed: int,
) -> list[str]:
    """Sample random token IDs and decode them to strings.

    Excludes special tokens (by id) so we don't seed with <|im_end|>, <|endoftext|>, etc.
    The decoded string will be re-tokenized on the extraction side — not exactly
    identical to the sampled IDs, but close enough for a "meaningless token sequence"
    null condition.
    """
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    vocab_size = tok.vocab_size
    special_ids = set(tok.all_special_ids or [])
    # Qwen also has a block of added tokens outside vocab_size we just avoid.
    rng = np.random.default_rng(seed)

    prompts: list[str] = []
    # Sample with rejection of special tokens. vocab_size excludes added tokens
    # for most HF tokenizers.
    candidate_ids = np.arange(vocab_size, dtype=np.int64)
    keep = np.array([i not in special_ids for i in candidate_ids])
    allowed = candidate_ids[keep]

    for _ in range(n_prompts):
        idx = rng.integers(0, len(allowed), size=tokens_per_prompt)
        ids = allowed[idx].tolist()
        text = tok.decode(ids, skip_special_tokens=True)
        if not text.strip():
            text = "x"  # guard against an empty decode
        prompts.append(text)
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tokens-per-prompt", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(HERE))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random_prompts = generate_random_token_prompts(
        args.tokenizer, args.n, args.tokens_per_prompt, args.seed
    )
    (out_dir / "prompts_random.json").write_text(
        json.dumps(
            {
                "prompts": random_prompts,
                "meta": {
                    "tokenizer": args.tokenizer,
                    "n": args.n,
                    "tokens_per_prompt": args.tokens_per_prompt,
                    "seed": args.seed,
                    "mode": "random_tokens_decoded",
                },
            },
            indent=2,
        )
    )
    print(f"Wrote {out_dir/'prompts_random.json'} ({len(random_prompts)} prompts)")

    stream = STREAM_PROMPTS[: args.n]
    (out_dir / "prompts_stream.json").write_text(
        json.dumps({"prompts": stream, "meta": {"mode": "stream_of_consciousness"}}, indent=2)
    )
    print(f"Wrote {out_dir/'prompts_stream.json'} ({len(stream)} prompts)")


if __name__ == "__main__":
    main()
