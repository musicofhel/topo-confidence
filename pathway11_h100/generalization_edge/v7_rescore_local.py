#!/usr/bin/env python3
"""v7_rescore_local.py — SPEC v7 Phase 1a (EXP-84): teacher-forced rescoring.

Recovers per-token features from cached generated TEXTS (no generation):
re-build the exact original prompt, concat re-tokenized generation, one
forward pass, read per-position logprob/entropy/top-2 margin over the
generation span + the answer-span mean logprob.

Cells (local 2060, Qwen-1.5B fp32 — bf16 unsupported on Turing, fp16 overflows
Qwen2 MLP; lm_head applied in row chunks to avoid the full (T,V) allocation):
    --cell math : pathway8_layerwise/data/math500/problem_*.npz   (n=500)
    --cell bbh  : pathway8_layerwise/data/bbh/<subset>/problem_*.npz (n=750)

Output sidecar: results/v7_rescore_qwen1.5b_<cell>.npz with per-problem
    min_lp, p10_lp, mean_entropy, mean_top2_margin, ans_span_lp (NaN if no
    answer span), n_gen_tokens_rescored, mean_lp_rescored, y
plus alignment sanity: corr(mean_lp_rescored, cached mean_logprob) — printed
and stored; the run FAILS (exit 1) if r < 0.90 (prompt reconstruction drift).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/musicofhel/topo-confidence")
sys.path.insert(0, str(ROOT))

from pathway8_layerwise.extract_math500 import (  # noqa: E402
    SYSTEM_PROMPT, load_math500,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MATH_DIR = ROOT / "pathway8_layerwise" / "data" / "math500"
BBH_DIR = ROOT / "pathway8_layerwise" / "data" / "bbh"
BBH_SUBSETS = [  # manifest order (idx 0..749)
    "tracking_shuffled_objects_seven_objects",
    "logical_deduction_seven_objects",
    "web_of_lies",
]
COT_PROMPTS_URL = ("https://raw.githubusercontent.com/suzgunmirac/"
                   "BIG-Bench-Hard/main/cot-prompts/{subset}.txt")
COT_CACHE = HERE / "cache" / "bbh_cot_prompts"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# extract_bbh.py answer regexes (verbatim) — for the BBH answer span.
_BBH_ANSWER_PATS = [
    re.compile(r"(?:the\s+answer\s+is)\s*\(([A-Z])\)", re.I),
    re.compile(r"answer\s*(?:is|:)\s*\(([A-Z])\)", re.I),
    re.compile(r"(?:the\s+answer\s+is)\s*([A-Z])\b", re.I),
    re.compile(r"answer\s*(?:is|:)\s*([A-Z])\b", re.I),
    re.compile(r"\\boxed\{\(?([A-Z])\)?\}", re.I),
    re.compile(r"\(([A-Z])\)\s*\.?\s*$", re.M),
    re.compile(r"(?:the\s+answer\s+is)\s*(Yes|No)\b", re.I),
    re.compile(r"answer\s*(?:is|:)\s*(Yes|No)\b", re.I),
]


def math_answer_span(text: str) -> tuple[int, int] | None:
    """Char span of the \\boxed{...} CONTENT (balanced braces), or None."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return (start, i - 1)


def bbh_answer_span(text: str) -> tuple[int, int] | None:
    """Char span of the answer token (group 1 of the first matching tier)."""
    for pat in _BBH_ANSWER_PATS:
        matches = list(pat.finditer(text))
        if matches:
            return matches[-1].span(1)
    return None


def load_cot_prompt(subset: str) -> str:
    COT_CACHE.mkdir(parents=True, exist_ok=True)
    local = COT_CACHE / f"{subset}.txt"
    if local.exists():
        return local.read_text()
    sysdir = Path("/opt/bbh-prompts") / f"{subset}.txt"
    if sysdir.exists():
        return sysdir.read_text()
    with urllib.request.urlopen(COT_PROMPTS_URL.format(subset=subset)) as r:
        txt = r.read().decode()
    local.write_text(txt)
    return txt


def build_math_prompts(tok) -> list[dict]:
    problems = load_math500()
    assert len(problems) == 500
    out = []
    for i, p in enumerate(problems):
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p["problem"]}]
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
        d = np.load(MATH_DIR / f"problem_{i:03d}.npz", allow_pickle=True)
        out.append({"prompt": prompt, "text": str(d["text"]),
                    "y": bool(d["correct"]),
                    "cached_mlp": float(d["mean_logprob"]),
                    "span_fn": math_answer_span})
    return out


def build_bbh_prompts(tok) -> list[dict]:
    from datasets import load_dataset
    out = []
    for subset in BBH_SUBSETS:
        ds = load_dataset("lukaemon/bbh", subset, split="test")
        cot = load_cot_prompt(subset)
        sub_dir = BBH_DIR / subset
        n = len(list(sub_dir.glob("problem_*.npz")))
        for i in range(n):
            user = f"{cot}\n\nQ: {ds[i]['input']}\nA: Let's think step by step."
            messages = [{"role": "user", "content": user}]
            prompt = tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
            d = np.load(sub_dir / f"problem_{i:03d}.npz", allow_pickle=True)
            out.append({"prompt": prompt, "text": str(d["text"]),
                        "y": bool(d["correct"]),
                        "cached_mlp": float(d["mean_logprob"]),
                        "span_fn": bbh_answer_span})
    assert len(out) == 750, len(out)
    return out


@torch.inference_mode()
def rescore_one(model, tok, item, chunk: int = 256) -> dict:
    dev = model.device
    p_ids = tok(item["prompt"], return_tensors="pt").input_ids[0]
    enc = tok(item["text"], add_special_tokens=False,
              return_offsets_mapping=True)
    t_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
    offsets = enc["offset_mapping"]
    T = len(t_ids)
    if T == 0:
        return {k: np.nan for k in ("min_lp", "p10_lp", "mean_entropy",
                                    "mean_top2_margin", "ans_span_lp",
                                    "mean_lp_rescored")} | {"n_gen": 0}
    ids = torch.cat([p_ids, t_ids]).unsqueeze(0).to(dev)
    P = len(p_ids)

    hidden = model.model(input_ids=ids).last_hidden_state[0]  # (P+T, H)
    # positions predicting generation tokens: P-1 .. P+T-2
    pred_h = hidden[P - 1: P + T - 1]                          # (T, H)
    targets = t_ids.to(dev)                                    # (T,)

    lps = torch.empty(T, dtype=torch.float32)
    ents = torch.empty(T, dtype=torch.float32)
    margins = torch.empty(T, dtype=torch.float32)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        logits = model.lm_head(pred_h[s:e]).float()            # (c, V)
        logp = torch.log_softmax(logits, dim=-1)
        lps[s:e] = logp.gather(1, targets[s:e, None])[:, 0].cpu()
        prob = logp.exp()
        ents[s:e] = -(prob * logp).sum(-1).cpu()
        top2 = logits.topk(2, dim=-1).values
        margins[s:e] = (top2[:, 0] - top2[:, 1]).cpu()
        del logits, logp, prob, top2
    lps_np = lps.numpy()

    span = item["span_fn"](item["text"])
    ans_lp = np.nan
    if span is not None:
        a, b = span
        tok_idx = [j for j, (cs, ce) in enumerate(offsets)
                   if cs < b and ce > a]
        if tok_idx:
            ans_lp = float(lps_np[tok_idx].mean())

    k10 = max(1, int(np.ceil(0.10 * T)))
    return {
        "min_lp": float(lps_np.min()),
        "p10_lp": float(np.sort(lps_np)[:k10].mean()),
        "mean_entropy": float(ents.numpy().mean()),
        "mean_top2_margin": float(margins.numpy().mean()),
        "ans_span_lp": ans_lp,
        "mean_lp_rescored": float(lps_np.mean()),
        "n_gen": T,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["math", "bbh"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Loading {MODEL_NAME} fp32 on {args.device} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
        attn_implementation="sdpa").to(args.device).eval()

    items = (build_math_prompts(tok) if args.cell == "math"
             else build_bbh_prompts(tok))
    if args.limit:
        items = items[: args.limit]

    rows = []
    t0 = time.time()
    for i, item in enumerate(items):
        rows.append(rescore_one(model, tok, item))
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(items)}  ({el/(i+1):.2f}s/prob, "
                  f"ETA {(len(items)-i-1)*el/(i+1)/60:.1f} min)", flush=True)

    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["y"] = np.array([it["y"] for it in items])
    cached = np.array([it["cached_mlp"] for it in items])
    out["cached_mean_logprob"] = cached

    ok = np.isfinite(out["mean_lp_rescored"])
    r = float(np.corrcoef(out["mean_lp_rescored"][ok], cached[ok])[0, 1])
    out["alignment_corr"] = np.array(r)
    n_span = int(np.isfinite(out["ans_span_lp"]).sum())
    print(f"\nalignment corr(rescored, cached mean_logprob) = {r:.4f}")
    print(f"answer-span found: {n_span}/{len(items)}")

    RESULTS.mkdir(exist_ok=True)
    f = RESULTS / f"v7_rescore_qwen1.5b_{args.cell}.npz"
    np.savez(f, **out)
    print(f"-> {f}")
    if r < 0.90:
        print("FAIL: alignment corr < 0.90 — prompt reconstruction drifted.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
