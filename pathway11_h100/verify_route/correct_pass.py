#!/usr/bin/env python3
"""S3 — Verify-then-correct (Arm C), GPU.

Kumaran 2604.22271's actual +3.7pp mechanism: tell the model its answer may be wrong
and have it self-correct. We run it on ALL 500 problems (full-coverage Δ vs K=1 48.6%);
the routing simulation (S5) then charges correction only on the routed subset.

Prompt = problem + the model's own K=1 boxed answer + "this may be wrong, reconsider".
Generate a fresh full solution (max_new_tokens=1024, greedy), grade vs gold answer.

Output: results/correct_pass.json  {idx, corrected_text, corrected_correct} + full-coverage acc.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "causal_dom"))
from common import load_model, format_chat, check_correct, _extract_boxed, SYSTEM_PROMPT  # noqa: E402

N = 500


def build_correct_prompt(problem: str, k1_answer_text: str, tokenizer) -> str:
    boxed = _extract_boxed(k1_answer_text)
    user = (
        f"{problem}\n\n"
        f"A first attempt gave the answer \\boxed{{{boxed}}}, but this may be incorrect. "
        f"Carefully reconsider the problem from scratch, reason step by step, and put your "
        f"final (corrected) answer within \\boxed{{}}."
    )
    return format_chat(user, tokenizer, system=SYSTEM_PROMPT)


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=1024)
    args = ap.parse_args()

    k1 = json.loads((HERE / "results" / "k1_greedy.json").read_text())["rows"]
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = [ds[i]["problem"] for i in range(N)]
    golds = [ds[i]["answer"] for i in range(N)]

    model, tok = load_model()
    eos_ids = [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]

    prompts = [build_correct_prompt(problems[r["idx"]], r["text"], tok) for r in k1]

    corrected: list[str] = []
    for s in range(0, len(prompts), args.batch):
        chunk = prompts[s : s + args.batch]
        enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(
            **enc, max_new_tokens=args.max_new, do_sample=False,
            pad_token_id=tok.pad_token_id, eos_token_id=eos_ids,
        )
        new = gen[:, enc.input_ids.shape[1] :]
        corrected.extend(tok.batch_decode(new, skip_special_tokens=True))
        print(f"  correct {s + len(chunk)}/{len(prompts)}", flush=True)

    rows = []
    for r, txt in zip(k1, corrected):
        cc = check_correct(txt, golds[r["idx"]])
        rows.append({"idx": r["idx"], "corrected_text": txt, "corrected_correct": bool(cc)})

    corrected_correct = np.array([r["corrected_correct"] for r in rows], dtype=bool)
    k1_correct = np.array([r["correct"] for r in k1], dtype=bool)
    full_cov_acc = float(corrected_correct.mean())

    (HERE / "results" / "correct_pass.json").write_text(json.dumps({
        "full_coverage_corrected_accuracy": full_cov_acc,
        "k1_accuracy": float(k1_correct.mean()),
        "delta_vs_k1": full_cov_acc - float(k1_correct.mean()),
        "n_flipped_right_to_wrong": int((k1_correct & ~corrected_correct).sum()),
        "n_flipped_wrong_to_right": int((~k1_correct & corrected_correct).sum()),
        "rows": rows,
    }, ensure_ascii=False))

    print(f"\n  full-coverage corrected accuracy: {full_cov_acc:.4f}  (K=1 was {k1_correct.mean():.4f}, "
          f"Δ={full_cov_acc - k1_correct.mean():+.4f})")
    print(f"  flips: right->wrong {(k1_correct & ~corrected_correct).sum()}, "
          f"wrong->right {(~k1_correct & corrected_correct).sum()}")
    print("  wrote results/correct_pass.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
