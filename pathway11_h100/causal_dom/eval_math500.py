#!/usr/bin/env python3
"""FE269 — MATH-500 K=1 under a directional intervention (off/ablate/add).

Reports accuracy and per-problem correctness. The per-problem `orig_correct` (from
the cached baseline labels, 243 True) lets the addition arm compute both:
  - flip rate  = fraction of originally-INCORRECT now correct
  - retention  = fraction of originally-CORRECT still correct

Usage:
  python eval_math500.py --mode off    --layers all --out results/math_off.json
  python eval_math500.py --mode ablate --layers 18  --out results/math_ablate_L19.json
  python eval_math500.py --mode ablate --layers all --out results/math_ablate_all.json
  python eval_math500.py --mode add    --layers 18  --alpha-mult 2.0 --out results/math_add_a2.json
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
from common import (  # noqa: E402
    load_model, check_correct, format_chat, batched_generate, MODEL_NAME,
)
from intervention import DirectionIntervention  # noqa: E402

N_PROBLEMS = 500


def parse_layers(s: str, n_layers: int) -> list[int]:
    if s == "all":
        return list(range(n_layers))
    return [int(x) for x in s.split(",")]


def load_math500():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [{"problem": x["problem"], "answer": x["answer"]} for x in ds][:N_PROBLEMS]


def load_orig_labels() -> np.ndarray | None:
    """Baseline K=1 correctness (243 True) from the prefill cache, for flip/retention."""
    npz = HERE.parent / "prefill_inversion/cache/m15b_prefill.npz"
    if not npz.exists():
        return None
    with np.load(npz, allow_pickle=True) as d:
        return d["correct"].astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["off", "ablate", "add"], required=True)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--alpha-mult", type=float, default=0.0,
                    help="add mode: alpha = alpha_mult * ||r|| (||r|| from provenance)")
    ap.add_argument("--rhat", default=str(HERE / "r_hat.npy"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r_hat = np.load(args.rhat)
    prov = json.loads((HERE / "rhat_provenance.json").read_text())
    r_norm = float(prov["r_norm"])
    alpha = args.alpha_mult * r_norm if args.mode == "add" else 0.0

    model, tok = load_model()
    n_layers = len(model.model.layers)
    layers = parse_layers(args.layers, n_layers)

    probs = load_math500()
    prompts = [format_chat(p["problem"], tok) for p in probs]
    orig = load_orig_labels()

    print(f"[math500] mode={args.mode} layers={args.layers}({len(layers)}) "
          f"alpha_mult={args.alpha_mult} alpha={alpha:.4f} batch={args.batch}", flush=True)

    with DirectionIntervention(model, r_hat, mode=args.mode, alpha=alpha, layers=layers):
        texts = batched_generate(model, tok, prompts,
                                 max_new_tokens=args.max_new_tokens, batch_size=args.batch)

    correct = np.array([check_correct(t, p["answer"]) for t, p in zip(texts, probs)])
    acc = float(correct.mean())
    n_correct = int(correct.sum())

    out = {
        "benchmark": "math500", "model": MODEL_NAME, "mode": args.mode,
        "layers_arg": args.layers, "n_layers_hooked": len(layers),
        "alpha_mult": args.alpha_mult, "alpha": alpha, "r_norm": r_norm,
        "batch": args.batch, "max_new_tokens": args.max_new_tokens,
        "n": len(probs), "n_correct": n_correct, "accuracy": acc,
        "per_problem_correct": correct.astype(int).tolist(),
    }
    if orig is not None and len(orig) == len(correct):
        out["orig_correct"] = orig.astype(int).tolist()
        inc = ~orig; cor = orig
        out["flip_rate_on_incorrect"] = float(correct[inc].mean()) if inc.any() else None
        out["retention_on_correct"] = float(correct[cor].mean()) if cor.any() else None
        out["n_orig_incorrect"] = int(inc.sum()); out["n_orig_correct"] = int(cor.sum())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    msg = f"  acc = {acc:.4f} ({n_correct}/{len(probs)})"
    if "flip_rate_on_incorrect" in out:
        msg += f" | flip={out['flip_rate_on_incorrect']:.3f} retain={out['retention_on_correct']:.3f}"
    print(msg + f"  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
