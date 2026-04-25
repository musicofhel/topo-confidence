#!/usr/bin/env python3
"""Stage 3: K=8 self-consistency on MATH-500 × 1.5B, with per-sample L19.

GPU REQUIRED. Run on RunPod H100.

For each of 500 MATH-500 problems, sample K=8 completions at T=0.7
(max_new_tokens=1024, top_p=1.0, do_sample=True). Capture ONLY the L19
hidden state at each sample's final generated token (1536-d). Score
correctness via math_verify.

Output per problem: .npz with
  L19_samples: (8, 1536) float16 — final-token L19 per sample
  texts:       object array (8,) — decoded generations
  correct:     (8,) bool — math_verify correctness per sample

Checkpoint per problem (skip-if-exists). If a problem is mid-way, redo
the 8 samples (partial-sample resume not worth the complexity at 30s loss).

Estimated runtime: ~45 min on H100 (500 × 8 = 4000 generations).
"""
from __future__ import annotations

import gc
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathway8_layerwise.extract_math500 import (
    check_correct,
    format_prompt,
    load_math500,
)
from pathway8_layerwise.extraction_utils import load_model
from pathway11_h100.config import (
    K8_DATA_DIR,
    K_SAMPLES,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    N_PROBLEMS,
    RESULTS_DIR,
    SAMPLING_TEMPERATURE,
    STEERING_LAYER,
    mark_done,
)


def sample_with_L19(model, tokenizer, prompt: str) -> dict:
    """Single sample at T=0.7, returns text, correctness-unready, L19_final.

    Uses output_hidden_states=True during generation and keeps ONLY the
    L19 activation at the final generated token position. All other
    hidden-state tensors are discarded as soon as we slice out L19.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=SAMPLING_TEMPERATURE,
            top_p=1.0,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
            ],
        )

    # out.hidden_states is a tuple of (n_gen) per-step tuples.
    # Step t, layer L: shape (batch, seq_len_t, hidden). seq_len_0 = prompt_len;
    # seq_len_t>0 = 1. We want the LAST step's L19, last token.
    last_step_L19 = out.hidden_states[-1][STEERING_LAYER][0, -1, :]
    L19_final = last_step_L19.float().cpu().numpy().astype(np.float16)

    gen_ids = out.sequences[0][prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    del out
    torch.cuda.empty_cache()

    return {"text": text, "L19_final": L19_final}


def save_k8_checkpoint(
    output_dir: Path,
    idx: int,
    L19_samples: np.ndarray,
    texts: list[str],
    correct: np.ndarray,
) -> None:
    """Atomic save of K=8 samples for one problem."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"problem_{idx:03d}.npz"

    fd, tmp_path = tempfile.mkstemp(
        suffix=".npz", dir=output_dir, prefix=f"problem_{idx:03d}_"
    )
    os.close(fd)

    try:
        np.savez_compressed(
            tmp_path,
            L19_samples=L19_samples,
            texts=np.array(texts, dtype=object),
            correct=correct,
        )
        os.rename(tmp_path, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def main():
    K8_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Stage 3: K={K_SAMPLES} self-consistency × 1.5B, T={SAMPLING_TEMPERATURE}")
    print(f"         Saving L19 final-token activations per sample (1536-d float16)")
    print("=" * 70)

    # Optional smoke-test cap via env var
    n_problems_cap = int(os.environ.get("STAGE3_N_PROBLEMS", N_PROBLEMS))
    if n_problems_cap != N_PROBLEMS:
        print(f"  [SMOKE] STAGE3_N_PROBLEMS={n_problems_cap} (override)")

    done_set = set()
    for f in K8_DATA_DIR.glob("problem_*.npz"):
        idx = int(f.stem.split("_")[1])
        done_set.add(idx)
    print(f"  {len(done_set)} problems already sampled, continuing from remainder")

    model, tokenizer = load_model(MODEL_NAME)

    problems = load_math500()[:n_problems_cap]

    t0 = time.time()
    n_majority_correct = 0
    n_any_correct = 0
    n_all_correct = 0

    for i, prob in enumerate(problems):
        if i in done_set:
            data = np.load(K8_DATA_DIR / f"problem_{i:03d}.npz", allow_pickle=True)
            correct = data["correct"]
            n_majority_correct += int(correct.sum() >= (K_SAMPLES // 2 + 1))
            n_any_correct += int(correct.any())
            n_all_correct += int(correct.all())
            continue

        prompt = format_prompt(prob["problem"], tokenizer)

        L19_samples = np.zeros((K_SAMPLES, 1536), dtype=np.float16)
        texts: list[str] = []
        correct = np.zeros(K_SAMPLES, dtype=bool)

        try:
            for k in range(K_SAMPLES):
                res = sample_with_L19(model, tokenizer, prompt)
                L19_samples[k] = res["L19_final"]
                texts.append(res["text"])
                correct[k] = check_correct(res["text"], prob["answer"])

            save_k8_checkpoint(K8_DATA_DIR, i, L19_samples, texts, correct)

        except Exception as e:
            print(f"  ERROR problem {i}: {e}")
            continue

        n_majority_correct += int(correct.sum() >= (K_SAMPLES // 2 + 1))
        n_any_correct += int(correct.any())
        n_all_correct += int(correct.all())

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            done_new = (i + 1) - len([j for j in range(i + 1) if j in done_set])
            rate = done_new / elapsed if elapsed > 0 and done_new > 0 else 0
            remaining = n_problems_cap - (i + 1)
            eta = remaining / rate if rate > 0 else 0
            print(
                f"  {i+1}/{n_problems_cap}: "
                f"maj={n_majority_correct}/{i+1} "
                f"any={n_any_correct}/{i+1} "
                f"all={n_all_correct}/{i+1}, "
                f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s"
            )

        if (i + 1) % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    total = time.time() - t0
    n = max(n_problems_cap, 1)
    print(
        f"\nDone: {total:.0f}s total. "
        f"Majority-vote acc={n_majority_correct/n:.1%}, "
        f"any-correct rate={n_any_correct/n:.1%}, "
        f"all-correct rate={n_all_correct/n:.1%}"
    )

    n_files = len(list(K8_DATA_DIR.glob("problem_*.npz")))
    print(f"Output: {n_files}/{n_problems_cap} npz files in {K8_DATA_DIR}")

    if n_files == N_PROBLEMS and n_problems_cap == N_PROBLEMS:
        mark_done("stage3", RESULTS_DIR)
        print("[DONE] stage3 marker written")
    else:
        print("[WARN] stage3 not marked done (incomplete or smoke-test)")


if __name__ == "__main__":
    main()
