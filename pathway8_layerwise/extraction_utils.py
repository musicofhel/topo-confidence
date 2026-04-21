"""Shared hidden-state extraction utilities for Pathway 8.

Provides a unified function to:
1. Generate a completion with all-layer hidden states captured
2. Run a second forward pass for attention-based D2HScore features
3. Save per-problem checkpoints with atomic writes
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Qwen2.5-1.5B-Instruct with correct settings.

    - bf16 (NEVER fp16 — MLP overflow with intermediate_size=8960)
    - sdpa attention (NOT flash-attn-2 — broken with right-padding on Qwen2)
    - pad_token = <|endoftext|> (id 151643), NOT <|im_end|> (id 151645)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.pad_token_id = 151643
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()

    print(f"  Model loaded: {model_name} (bf16, sdpa)")
    print(f"  Device: {next(model.parameters()).device}")
    return model, tokenizer


def extract_all_layers_and_attention(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    capture_attention: bool = True,
) -> dict:
    """Generate completion and capture per-layer hidden states + attention stats.

    Returns dict with:
      states: (29, n_gen, 1536) float16 — all-layer hidden states
      d2h_attn_entropy: (28,) float64 — per-layer mean attention entropy (if capture_attention)
      text: str — generated text
      n_gen_tokens: int — number of generated tokens
      mean_logprob: float — mean token log-probability
      prompt_len: int — length of tokenized prompt
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    # Step 1: Generate with hidden state capture
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
            ],
        )

    # Decode completion
    gen_ids = out.sequences[0][prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    n_gen = len(out.hidden_states)

    # Extract per-generated-token hidden states at ALL layers
    # out.hidden_states[t][layer]: (batch, seq_len_t, hidden_dim)
    # t=0: full prompt forward (seq_len_0 = prompt_len), t>0: (batch, 1, hidden_dim)
    n_layers = len(out.hidden_states[0])  # 29 for Qwen2.5-1.5B
    hidden_dim = out.hidden_states[0][0].shape[-1]

    states = np.zeros((n_layers, n_gen, hidden_dim), dtype=np.float16)
    for t in range(n_gen):
        for layer in range(n_layers):
            vec = out.hidden_states[t][layer][0, -1, :].float().cpu().numpy()
            states[layer, t, :] = vec.astype(np.float16)

    # Mean token log-probability
    mean_logprob = -100.0
    if len(out.scores) > 0 and len(gen_ids) > 0:
        logprobs = []
        for i, scores in enumerate(out.scores):
            if i >= len(gen_ids):
                break
            lp = torch.log_softmax(scores[0].float(), dim=-1)
            logprobs.append(lp[gen_ids[i]].item())
        if logprobs:
            mean_logprob = float(np.mean(logprobs))

    result = {
        "states": states,
        "text": text,
        "n_gen_tokens": n_gen,
        "mean_logprob": mean_logprob,
        "prompt_len": prompt_len,
    }

    # Step 2: Second forward pass for attention-based D2HScore features
    if capture_attention:
        result["d2h_attn_entropy"] = _compute_attention_entropy(
            model, out.sequences[0], prompt_len
        )
    else:
        result["d2h_attn_entropy"] = np.zeros(n_layers - 1, dtype=np.float64)

    # Free generate outputs to reclaim GPU memory
    del out
    torch.cuda.empty_cache()

    return result


def _compute_attention_entropy(
    model: AutoModelForCausalLM,
    full_sequence: torch.Tensor,
    prompt_len: int,
) -> np.ndarray:
    """Run a forward pass on the full sequence to get attention matrices.

    Computes per-layer mean attention entropy from the last generated token
    attending to all other tokens. Returns (28,) float64 array.

    Memory: at T=1024, attention tensors are ~1.4 GB — well within H100.

    Note: sdpa does not support output_attentions, so we temporarily switch
    each layer to eager attention for this pass.
    """
    # sdpa cannot output attentions — temporarily switch config to eager
    orig_attn = getattr(model.config, "_attn_implementation", "sdpa")
    model.config._attn_implementation = "eager"
    for layer in model.model.layers:
        layer.self_attn._attn_implementation = "eager"

    try:
        with torch.inference_mode():
            fwd_out = model(
                input_ids=full_sequence.unsqueeze(0),
                output_attentions=True,
                use_cache=False,
            )
    finally:
        # Restore sdpa for generation performance
        model.config._attn_implementation = orig_attn
        for layer in model.model.layers:
            layer.self_attn._attn_implementation = orig_attn

    # fwd_out.attentions: tuple of 28 tensors, each (1, n_heads, seq_len, seq_len)
    n_layers = len(fwd_out.attentions)
    attn_entropy = np.zeros(n_layers, dtype=np.float64)

    for layer_idx, attn in enumerate(fwd_out.attentions):
        # attn: (1, n_heads, seq_len, seq_len)
        # Get attention from last token to all tokens, averaged over heads
        last_tok_attn = attn[0, :, -1, :]  # (n_heads, seq_len)
        # Compute entropy per head, then average
        # Add eps to avoid log(0)
        eps = 1e-12
        entropy_per_head = -(last_tok_attn * torch.log(last_tok_attn + eps)).sum(dim=-1)
        attn_entropy[layer_idx] = entropy_per_head.float().mean().cpu().item()

    del fwd_out
    torch.cuda.empty_cache()

    return attn_entropy


def save_problem_checkpoint(
    output_dir: Path,
    idx: int,
    states: np.ndarray,
    text: str,
    correct: bool,
    mean_logprob: float,
    d2h_attn_entropy: np.ndarray,
    extra_metadata: dict | None = None,
) -> None:
    """Save per-problem checkpoint with atomic write (write to .tmp, then rename)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"problem_{idx:03d}.npz"

    # Atomic write: write to temp file, then rename
    fd, tmp_path = tempfile.mkstemp(
        suffix=".npz", dir=output_dir, prefix=f"problem_{idx:03d}_"
    )
    os.close(fd)

    try:
        np.savez_compressed(
            tmp_path,
            states=states,
            d2h_attn_entropy=d2h_attn_entropy,
            text=np.array(text),
            correct=np.array(correct),
            mean_logprob=np.array(mean_logprob),
        )
        os.rename(tmp_path, final_path)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def save_manifest(
    output_dir: Path,
    problems: list[dict],
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    benchmark: str = "math500",
) -> None:
    """Save manifest.json with metadata for all problems."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": model_name,
        "benchmark": benchmark,
        "n_problems": len(problems),
        "n_correct": sum(1 for p in problems if p.get("correct", False)),
        "accuracy": sum(1 for p in problems if p.get("correct", False)) / max(len(problems), 1),
        "problems": problems,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))


def subsample_tokens(
    points: np.ndarray,
    n: int = 100,
) -> np.ndarray:
    """Evenly-spaced subsample preserving temporal coverage."""
    if len(points) <= n:
        return points
    idx = np.linspace(0, len(points) - 1, n, dtype=int)
    return points[idx]
