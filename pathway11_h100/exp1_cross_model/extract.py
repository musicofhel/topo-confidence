#!/usr/bin/env python3
"""Exp 1: Cross-model temporal PR extraction — Phi-3-mini & Llama-3.2-1B.

Greedy @ T=0, max 1024 tokens. For each of 500 MATH-500 problems:
  - prefill_all_layers: (n_layers+1, hidden_dim) float16 — all layers at end-of-prefill
  - final_all_layers:   (n_layers+1, hidden_dim) float16 — all layers at final generated token
  - twothirds_positions:(n_pos, hidden_dim) float16 — 2/3-depth layer at positions {1,10,25,50,100,200,final}
  - positions_actual:   (n_pos,) int32 — the actual generation-token indices captured
  - twothirds_layer_idx, n_gen_tokens, text, correct, mean_logprob

Usage:
  python pathway11_h100/exp1_cross_model/extract.py --model phi3mini
  python pathway11_h100/exp1_cross_model/extract.py --model llama32-1b
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pathway8_layerwise.extract_math500 import (  # reuse MATH-500 loader + checker
    check_correct,
    load_math500,
)
from pathway11_h100.config import MAX_NEW_TOKENS, N_PROBLEMS, RESULTS_DIR, mark_done

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = {
    "phi3mini": {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "n_transformer_layers": 32,
        "subdir": "phi3mini",
        "attn_implementation": "eager",  # sdpa unsupported for Phi3 sliding-window
        "trust_remote_code": False,       # use built-in Phi3ForCausalLM class
    },
    "llama32-1b": {
        "name": "unsloth/Llama-3.2-1B-Instruct",
        "n_transformer_layers": 16,
        "subdir": "llama32_1b",
        "attn_implementation": "sdpa",
        "trust_remote_code": False,
    },
    "qwen25-1.5b": {
        # Qwen's eos is <|im_end|> (151645); padding must be <|endoftext|> (151643)
        # or right-padded batches silently corrupt. See pathway8_layerwise/extraction_utils.py.
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "n_transformer_layers": 28,
        "subdir": "qwen25_15b",
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
        "pad_token": "<|endoftext|>",
        "pad_token_id": 151643,
    },
}

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# Generation-step indices at which to snapshot the 2/3-depth layer.
# Index 1 = first generated token; "final" is added at runtime.
TARGET_POSITIONS = [1, 10, 25, 50, 100, 200]

EXP1_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP1_DIR / "data"


# ---------------------------------------------------------------------------
# Model loading (generic — works for Phi-3 and Llama-3.2)
# ---------------------------------------------------------------------------


def load_model(
    model_name: str,
    attn_implementation: str,
    trust_remote_code: bool,
    pad_token: str | None = None,
    pad_token_id: int | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    if pad_token is not None:
        tokenizer.pad_token = pad_token
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if pad_token_id is not None:
        tokenizer.pad_token_id = pad_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    print(f"  Model loaded: {model_name} (bf16, {attn_implementation})")
    print(f"  Device: {next(model.parameters()).device}")
    return model, tokenizer


def format_prompt(problem_text: str, tokenizer, system_prompt: str = SYSTEM_PROMPT) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def twothirds_layer_index(n_transformer_layers: int) -> int:
    """Index into hidden_states[t] for the 2/3-depth transformer layer output.

    hidden_states[t] is a tuple of length (n_transformer_layers + 1) where
    index 0 is the embedding output and index k is the output of layer k.
    Matches Qwen convention: 28 layers → index 19.
    """
    return int(round(n_transformer_layers * 2 / 3))


def extract_one(
    model, tokenizer, prompt: str, twothirds_idx: int, max_new_tokens: int
) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

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
        )

    gen_ids = out.sequences[0][prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # out.hidden_states is a tuple of length (num_forward_passes).
    # Index 0 = prefill forward (tensors shape (1, prompt_len, d)).
    # Index t>0 = one-token forward for generation step t (shape (1, 1, d)).
    # Each element is itself a tuple of (n_transformer_layers + 1) tensors.
    hs = out.hidden_states
    n_steps = len(hs)  # prefill + num_generated
    n_gen_tokens = n_steps - 1

    n_layers_total = len(hs[0])  # n_transformer_layers + 1
    hidden_dim = hs[0][0].shape[-1]

    # All-layer snapshot at end-of-prefill (last prompt position, all layers)
    prefill_all = np.zeros((n_layers_total, hidden_dim), dtype=np.float16)
    for layer in range(n_layers_total):
        prefill_all[layer] = (
            hs[0][layer][0, -1, :].float().cpu().numpy().astype(np.float16)
        )

    # All-layer snapshot at final generated token
    final_all = np.zeros((n_layers_total, hidden_dim), dtype=np.float16)
    for layer in range(n_layers_total):
        final_all[layer] = (
            hs[-1][layer][0, -1, :].float().cpu().numpy().astype(np.float16)
        )

    # 2/3-depth layer at specific generation positions.
    # Generation-step index p corresponds to hs[p] (p=1..n_gen_tokens).
    positions = [p for p in TARGET_POSITIONS if p <= n_gen_tokens]
    if n_gen_tokens >= 1 and (not positions or positions[-1] != n_gen_tokens):
        positions.append(n_gen_tokens)
    positions = np.array(positions, dtype=np.int32)

    twothirds_data = np.zeros((len(positions), hidden_dim), dtype=np.float16)
    for i, p in enumerate(positions):
        twothirds_data[i] = (
            hs[int(p)][twothirds_idx][0, -1, :]
            .float()
            .cpu()
            .numpy()
            .astype(np.float16)
        )

    # Mean token log-probability
    mean_logprob = -100.0
    if len(out.scores) > 0 and len(gen_ids) > 0:
        lps = []
        for i, scores in enumerate(out.scores):
            if i >= len(gen_ids):
                break
            lp = torch.log_softmax(scores[0].float(), dim=-1)
            lps.append(lp[gen_ids[i]].item())
        if lps:
            mean_logprob = float(np.mean(lps))

    del out
    torch.cuda.empty_cache()

    return {
        "prefill_all_layers": prefill_all,
        "final_all_layers": final_all,
        "twothirds_positions": twothirds_data,
        "positions_actual": positions,
        "twothirds_layer_idx": twothirds_idx,
        "n_gen_tokens": n_gen_tokens,
        "text": text,
        "mean_logprob": mean_logprob,
        "prompt_len": prompt_len,
    }


def save_checkpoint(out_dir: Path, idx: int, result: dict, correct: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"problem_{idx:03d}.npz"
    fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=out_dir, prefix=f"problem_{idx:03d}_")
    os.close(fd)
    try:
        np.savez_compressed(
            tmp_path,
            prefill_all_layers=result["prefill_all_layers"],
            final_all_layers=result["final_all_layers"],
            twothirds_positions=result["twothirds_positions"],
            positions_actual=result["positions_actual"],
            twothirds_layer_idx=np.array(result["twothirds_layer_idx"]),
            n_gen_tokens=np.array(result["n_gen_tokens"]),
            text=np.array(result["text"]),
            correct=np.array(bool(correct)),
            mean_logprob=np.array(result["mean_logprob"]),
        )
        os.rename(tmp_path, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True)
    parser.add_argument("--n-problems", type=int, default=N_PROBLEMS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help='JSON with {"prompts": [str, ...]}. Overrides MATH-500. No correctness check.',
    )
    parser.add_argument(
        "--skip-chat-template",
        action="store_true",
        help="Pass prompt strings straight to tokenizer (for random-token inputs).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Override SYSTEM_PROMPT (used only when chat template is applied).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output dir (default: data/<subdir>/).",
    )
    args = parser.parse_args()

    model_cfg = MODELS[args.model]
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = DATA_DIR / model_cfg["subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Exp 1 extraction: {model_cfg['name']}")
    print(f"  Output: {out_dir}")
    print(f"  n_problems={args.n_problems}, max_new_tokens={args.max_new_tokens}")
    print("=" * 70)

    twothirds_idx = twothirds_layer_index(model_cfg["n_transformer_layers"])
    print(
        f"  2/3-depth layer index (into hidden_states): {twothirds_idx} "
        f"of {model_cfg['n_transformer_layers']+1} total"
    )

    # Resume set
    done_set = set()
    for f in out_dir.glob("problem_*.npz"):
        try:
            done_set.add(int(f.stem.split("_")[1]))
        except ValueError:
            pass
    print(f"  {len(done_set)} problems already extracted, resuming")

    model, tokenizer = load_model(
        model_cfg["name"],
        attn_implementation=model_cfg["attn_implementation"],
        trust_remote_code=model_cfg["trust_remote_code"],
        pad_token=model_cfg.get("pad_token"),
        pad_token_id=model_cfg.get("pad_token_id"),
    )

    if args.prompts_file is not None:
        with open(args.prompts_file) as f:
            prompts_json = json.load(f)
        prompt_strs = prompts_json["prompts"]
        problems = [
            {"problem": s, "answer": "", "unique_id": f"custom_{k:03d}"}
            for k, s in enumerate(prompt_strs)
        ]
        print(
            f"  Loaded {len(problems)} prompts from {args.prompts_file} "
            f"(skip_chat_template={args.skip_chat_template})"
        )
        args.n_problems = min(args.n_problems, len(problems))
    else:
        problems = load_math500()
        assert len(problems) >= args.n_problems, (
            f"Expected >= {args.n_problems} problems, got {len(problems)}"
        )
    problems = problems[: args.n_problems]

    manifest_entries = []
    n_correct = 0
    t0 = time.time()

    for i, prob in enumerate(problems):
        if i in done_set:
            npz_path = out_dir / f"problem_{i:03d}.npz"
            try:
                data = np.load(npz_path, allow_pickle=True)
                correct = bool(data["correct"])
                n_correct += int(correct)
                manifest_entries.append({
                    "idx": i,
                    "unique_id": prob.get("unique_id", ""),
                    "correct": correct,
                    "mean_logprob": float(data["mean_logprob"]),
                    "n_gen_tokens": int(data["n_gen_tokens"]),
                })
                continue
            except Exception as e:
                print(f"  [WARN] problem {i} cached file unreadable ({e}); re-running")

        if args.skip_chat_template:
            prompt = prob["problem"]
        else:
            sys_prompt = args.system_prompt if args.system_prompt is not None else SYSTEM_PROMPT
            prompt = format_prompt(prob["problem"], tokenizer, sys_prompt)

        try:
            result = extract_one(
                model, tokenizer, prompt,
                twothirds_idx=twothirds_idx,
                max_new_tokens=args.max_new_tokens,
            )
            if args.prompts_file is not None:
                correct = False  # no ground truth for custom prompts
            else:
                correct = check_correct(result["text"], prob["answer"])
            n_correct += int(correct)
            save_checkpoint(out_dir, i, result, correct)
            manifest_entries.append({
                "idx": i,
                "unique_id": prob.get("unique_id", ""),
                "correct": bool(correct),
                "mean_logprob": float(result["mean_logprob"]),
                "n_gen_tokens": int(result["n_gen_tokens"]),
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
                remaining = sum(1 for j in range(i + 1, args.n_problems) if j not in done_set)
                eta = remaining / rate if rate > 0 else 0
                acc = n_correct / (i + 1)
                print(
                    f"  {i+1}/{args.n_problems}: acc={acc:.1%}, "
                    f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s"
                )

        if (i + 1) % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    total = time.time() - t0
    acc = n_correct / args.n_problems if args.n_problems else 0.0
    print(f"\nDone: {total:.0f}s total, accuracy={acc:.1%} ({n_correct}/{args.n_problems})")

    if args.prompts_file is not None:
        benchmark_tag = f"custom:{Path(args.prompts_file).name}"
        if args.skip_chat_template:
            benchmark_tag += "+no_chat_template"
    else:
        benchmark_tag = "math500_greedy_T0"

    manifest = {
        "model": model_cfg["name"],
        "n_transformer_layers": model_cfg["n_transformer_layers"],
        "twothirds_layer_idx": twothirds_idx,
        "benchmark": benchmark_tag,
        "max_new_tokens": args.max_new_tokens,
        "target_positions": TARGET_POSITIONS + ["final"],
        "n_problems": args.n_problems,
        "n_correct": n_correct,
        "accuracy": acc,
        "problems": manifest_entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    print(f"Manifest: {out_dir/'manifest.json'}")

    n_files = len(list(out_dir.glob("problem_*.npz")))
    print(f"Output: {n_files}/{args.n_problems} npz files")
    if n_files == args.n_problems and args.prompts_file is None:
        mark_done(f"exp1_{model_cfg['subdir']}", RESULTS_DIR)
        print(f"[DONE] exp1_{model_cfg['subdir']} marker written")


if __name__ == "__main__":
    main()
