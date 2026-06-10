#!/usr/bin/env python3
"""FE269 shared utilities — model loader, correctness checkers, prompt formatting,
batched generation. Logic copied verbatim from pathway8_layerwise (extraction_utils.py,
extract_math500.py) so the harness is self-contained on the pod (no config.py side
effects, no import-path fragility).

Key invariants (DO NOT change):
  - bf16, NEVER fp16 (Qwen2 MLP overflow at intermediate_size=8960)
  - sdpa attention (flash-attn2 broken with padding on Qwen2)
  - pad_token=<|endoftext|> (151643), padding_side="left"
"""
from __future__ import annotations

import re
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def load_model(model_name: str = MODEL_NAME):
    """Qwen2.5-1.5B-Instruct, batched-generation-ready (bf16, sdpa, left-pad)."""
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token = "<|endoftext|>"
    tok.pad_token_id = 151643
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    dev = next(model.parameters()).device
    print(f"  Model loaded: {model_name} (bf16, sdpa) on {dev}", flush=True)
    return model, tok


# ---- correctness (verbatim from pathway8_layerwise/extract_math500.py) ----

def _extract_boxed(text: str) -> str:
    """Extract content from \\boxed{...} with balanced brace matching."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
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


def check_correct(predicted_text: str, ground_truth: str) -> bool:
    """math-verify LaTeX equivalence; string-compare fallback."""
    try:
        from math_verify import parse, verify
        pred_parsed = parse(f"${_extract_boxed(predicted_text)}$")
        gt_parsed = parse(f"${ground_truth}$")
        return bool(verify(gt_parsed, pred_parsed))
    except Exception:
        return _extract_boxed(predicted_text).strip() == ground_truth.strip()


def _last_number(text: str) -> str | None:
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def check_correct_numeric(predicted_text: str, gold: str) -> bool:
    """GSM8K-style: compare boxed-or-last-number against a numeric gold."""
    boxed = _extract_boxed(predicted_text)
    pred = _last_number(boxed) or _last_number(predicted_text)
    if pred is None:
        return False
    try:
        gold_f = float(str(gold).replace(",", "").replace("$", "").strip())
        return abs(float(pred) - gold_f) < 1e-4
    except Exception:
        return str(pred).strip() == str(gold).strip()


# ---- prompt formatting ----

def format_chat(user_content: str, tokenizer, system: str = SYSTEM_PROMPT) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---- batched greedy generation ----

@torch.inference_mode()
def batched_generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 1024,
    batch_size: int = 32,
) -> list[str]:
    """Greedy, left-padded, batched. Returns decoded completions (prompt stripped).

    Correctness under interventions: the edit is per-position-independent and padded
    positions are attention-masked, so batching does not change any non-pad token's
    output relative to batch-size-1 (modulo bf16 reduction-order noise).
    """
    eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
    out_texts: list[str] = []
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s : s + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )
        # left-padded => prompt occupies the first enc.input_ids.shape[1] columns
        new_tokens = gen[:, enc.input_ids.shape[1] :]
        out_texts.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return out_texts
