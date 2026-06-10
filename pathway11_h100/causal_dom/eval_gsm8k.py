#!/usr/bin/env python3
"""FE269 — GSM8K capability check under a directional intervention.

Capability/specificity arm: if ablating the MATH-correctness direction also tanks
GSM8K, the direction is not MATH-correctness-specific (GENERAL DAMAGE). We need the
DROP, not absolute accuracy, so a fixed 500-problem subsample (SE~2.2%) is ample for
the +-5 / +-15 pt thresholds and keeps the run inside the pod-hour budget.

Usage:
  python eval_gsm8k.py --mode off    --layers all --out results/gsm8k_off.json
  python eval_gsm8k.py --mode ablate --layers 18  --out results/gsm8k_ablate_L19.json
  python eval_gsm8k.py --mode ablate --layers all --out results/gsm8k_ablate_all.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    load_model, check_correct_numeric, format_chat, batched_generate, MODEL_NAME,
)
from intervention import DirectionIntervention  # noqa: E402
from eval_math500 import parse_layers  # noqa: E402

N_SUB = 500
SUB_SEED = 1234


def load_gsm8k(n: int = N_SUB):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = [{"q": x["question"], "gold": x["answer"].split("####")[-1].strip()} for x in ds]
    rng = np.random.RandomState(SUB_SEED)
    idx = rng.choice(len(items), size=min(n, len(items)), replace=False)
    idx.sort()
    return [items[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["off", "ablate", "add"], required=True)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--alpha-mult", type=float, default=0.0)
    ap.add_argument("--rhat", default=str(HERE / "r_hat.npy"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--n", type=int, default=N_SUB)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r_hat = np.load(args.rhat)
    prov = json.loads((HERE / "rhat_provenance.json").read_text())
    r_norm = float(prov["r_norm"])
    alpha = args.alpha_mult * r_norm if args.mode == "add" else 0.0

    model, tok = load_model()
    layers = parse_layers(args.layers, len(model.model.layers))
    items = load_gsm8k(args.n)
    prompts = [format_chat(it["q"], tok) for it in items]

    print(f"[gsm8k] mode={args.mode} layers={args.layers}({len(layers)}) n={len(items)}", flush=True)
    with DirectionIntervention(model, r_hat, mode=args.mode, alpha=alpha, layers=layers):
        texts = batched_generate(model, tok, prompts,
                                 max_new_tokens=args.max_new_tokens, batch_size=args.batch)

    correct = np.array([check_correct_numeric(t, it["gold"]) for t, it in zip(texts, items)])
    acc = float(correct.mean())
    out = {
        "benchmark": "gsm8k", "model": MODEL_NAME, "mode": args.mode,
        "layers_arg": args.layers, "n_layers_hooked": len(layers),
        "alpha_mult": args.alpha_mult, "alpha": alpha,
        "subsample_seed": SUB_SEED, "n": len(items),
        "n_correct": int(correct.sum()), "accuracy": acc,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"  acc = {acc:.4f} ({int(correct.sum())}/{len(items)})  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
