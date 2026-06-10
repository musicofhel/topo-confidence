#!/usr/bin/env python3
"""FE269 — MMLU capability check under a directional intervention (answer-letter logprob).

Second specificity benchmark. Single forward per question (no generation): read the
logits at the final position and compare the 4 answer-letter token logprobs; argmax =
prediction. The hook still fires on the prefill forward, so the intervention applies.
1000-Q fixed-seed subsample. We care about the DROP vs off-baseline, not absolute MMLU,
so identical prompting across modes makes the delta meaningful. No lm-eval-harness
(won't compose with forward hooks).

Usage:
  python eval_mmlu.py --mode off    --layers all --out results/mmlu_off.json
  python eval_mmlu.py --mode ablate --layers 18  --out results/mmlu_ablate_L19.json
  python eval_mmlu.py --mode ablate --layers all --out results/mmlu_ablate_all.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from common import load_model, format_chat, MODEL_NAME  # noqa: E402
from intervention import DirectionIntervention  # noqa: E402
from eval_math500 import parse_layers  # noqa: E402

N_SUB = 1000
SUB_SEED = 4321
LETTERS = ["A", "B", "C", "D"]
MMLU_SYSTEM = "Answer the multiple choice question. Respond with only the letter of the correct choice."


def load_mmlu(n: int = N_SUB):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = np.random.RandomState(SUB_SEED)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    idx.sort()
    items = []
    for i in idx:
        x = ds[int(i)]
        items.append({"q": x["question"], "choices": x["choices"], "answer": int(x["answer"])})
    return items


def build_prompt(item, tok) -> str:
    body = item["q"] + "\n" + "\n".join(
        f"{L}. {c}" for L, c in zip(LETTERS, item["choices"])
    )
    return format_chat(body, tok, system=MMLU_SYSTEM)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["off", "ablate", "add"], required=True)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--alpha-mult", type=float, default=0.0)
    ap.add_argument("--rhat", default=str(HERE / "r_hat.npy"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n", type=int, default=N_SUB)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r_hat = np.load(args.rhat)
    prov = json.loads((HERE / "rhat_provenance.json").read_text())
    r_norm = float(prov["r_norm"])
    alpha = args.alpha_mult * r_norm if args.mode == "add" else 0.0

    model, tok = load_model()
    layers = parse_layers(args.layers, len(model.model.layers))
    items = load_mmlu(args.n)
    prompts = [build_prompt(it, tok) for it in items]
    gold = np.array([it["answer"] for it in items])

    # answer-letter token ids (first token of " A".."D" as the model would emit)
    letter_ids = []
    for L in LETTERS:
        ids = tok.encode(L, add_special_tokens=False)
        letter_ids.append(ids[0])
    letter_ids_t = torch.tensor(letter_ids, device=model.device)

    print(f"[mmlu] mode={args.mode} layers={args.layers}({len(layers)}) n={len(items)} "
          f"letter_ids={letter_ids}", flush=True)

    preds = np.zeros(len(items), dtype=int)
    with DirectionIntervention(model, r_hat, mode=args.mode, alpha=alpha, layers=layers):
        for s in range(0, len(prompts), args.batch):
            chunk = prompts[s : s + args.batch]
            enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
            logits = model(**enc).logits[:, -1, :]          # (B, vocab) last position
            letter_logits = logits[:, letter_ids_t]          # (B, 4)
            preds[s : s + len(chunk)] = letter_logits.argmax(dim=-1).cpu().numpy()

    correct = (preds == gold)
    acc = float(correct.mean())
    out = {
        "benchmark": "mmlu", "model": MODEL_NAME, "mode": args.mode,
        "layers_arg": args.layers, "n_layers_hooked": len(layers),
        "alpha_mult": args.alpha_mult, "alpha": alpha,
        "subsample_seed": SUB_SEED, "n": len(items),
        "n_correct": int(correct.sum()), "accuracy": acc, "letter_ids": letter_ids,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"  acc = {acc:.4f} ({int(correct.sum())}/{len(items)})  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
