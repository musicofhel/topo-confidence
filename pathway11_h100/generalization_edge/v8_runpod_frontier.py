#!/usr/bin/env python3
"""v8_runpod_frontier.py — SPEC v8 Phase 1 pod workload (H100 SXM, bf16).

One task, four models: greedy MATH-500 with gen-time feature capture (v7 D-4
house method: features from the decoder's own output_scores, never
decode->re-encode) plus the pre-registered banked side-captures:
  (i)  per-token logprob AND entropy ARRAYS for every model (flat + offsets,
       npz) — unblocks entropy-trajectory-shape follow-ups on cached data;
  (ii) mean-pooled prefill hidden states — L19 + last layer for the
       1.5B-class arms (28-layer Qwen2 architecture, the FE429 comparison
       cell), last layer only for the 7B-class arms.

Models (spec-frozen):
  math7b    Qwen/Qwen2.5-Math-7B-Instruct        1024 max  (H-N)
  mathstral mistralai/Mathstral-7B-v0.1          1024 max  (H-O)
  math1.5b  Qwen/Qwen2.5-Math-1.5B-Instruct     1024 max  (H-P/H-Q)
  r1distill deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  4096 max (H-P/H-Q;
            pre-registered fallback: if >20%% of greedy outputs hit max
            tokens unparseable, ONE re-run --temp 0.6 --seed 9999, D-n)

Grading is NOT done pod-side: texts are pulled and graded locally with the
pinned harness grader. Outputs per model in pod_results/:
  v8_frontier_{key}.json.gz   texts + scalar features + token counts + config
  v8_frontier_{key}_arrays.npz  per-token arrays + prefill states

Payload: pod_payload/v8_math500.jsonl.gz ({idx, problem, answer}).
Pull results after EACH task; pod has no volume.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PAYLOAD = HERE / "pod_payload"
OUT = HERE / "pod_results"

SYSTEM_PROMPT = ("Please reason step by step, and put your final answer "
                 "within \\boxed{}.")

MODELS = {
    "math7b": dict(name="Qwen/Qwen2.5-Math-7B-Instruct", max_new=1024,
                   bs=8, klass="7b", prefill_layers=("last",)),
    "mathstral": dict(name="mistralai/Mathstral-7B-v0.1", max_new=1024,
                      bs=8, klass="7b", prefill_layers=("last",)),
    "math1.5b": dict(name="Qwen/Qwen2.5-Math-1.5B-Instruct", max_new=1024,
                     bs=8, klass="1.5b", prefill_layers=(19, "last")),
    "r1distill": dict(name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                      max_new=4096, bs=4, klass="1.5b",
                      prefill_layers=(19, "last")),
}


def chat(tok, messages):
    """apply_chat_template; fold system into user when unsupported (Mistral)."""
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
    except Exception:
        sys_txt = "\n\n".join(m["content"] for m in messages
                              if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        if sys_txt and rest and rest[0]["role"] == "user":
            rest = ([{"role": "user",
                      "content": f"{sys_txt}\n\n{rest[0]['content']}"}]
                    + rest[1:])
        return tok.apply_chat_template(rest, tokenize=False,
                                       add_generation_prompt=True)


def load_payload():
    rows = []
    with gzip.open(PAYLOAD / "v8_math500.jsonl.gz", "rt") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _dtype_kw():
    import transformers
    return ("dtype" if int(transformers.__version__.split(".")[0]) >= 5
            else "torch_dtype")


def load_model(key: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODELS[key]["name"]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, attn_implementation="sdpa",
        device_map="cuda", **{_dtype_kw(): torch.bfloat16}).eval()
    print(f"loaded {name} (bf16, sdpa, {model.config.num_hidden_layers} "
          f"layers)", flush=True)
    return model, tok


def eos_ids(tok):
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>", "</s>",
              "<|end_of_text|>", "<|eot_id|>"):
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i != tok.unk_token_id and i >= 0:
            ids.add(i)
    return sorted(i for i in ids if i is not None and i >= 0)


def dump_json(obj: dict, name: str):
    OUT.mkdir(exist_ok=True)
    f = OUT / name
    with gzip.open(f, "wt") as fh:
        json.dump(obj, fh)
    print(f"-> {f}", flush=True)


@torch.inference_mode()
def token_features(logits_steps, chosen_ids):
    """Per-token (lp, entropy, top2 margin) from a (T, V) logits tensor,
    computed in 512-step chunks to bound the float32 footprint."""
    T = logits_steps.shape[0]
    lps = np.empty(T, dtype=np.float32)
    ents = np.empty(T, dtype=np.float32)
    margs = np.empty(T, dtype=np.float32)
    for s in range(0, T, 512):
        lg = logits_steps[s:s + 512].float()
        lp = torch.log_softmax(lg, dim=-1)
        lps[s:s + 512] = (lp.gather(1, chosen_ids[s:s + 512, None])[:, 0]
                          .cpu().numpy())
        ents[s:s + 512] = (-(lp.exp() * lp).sum(-1)).cpu().numpy()
        top2 = lg.topk(2, dim=-1).values
        margs[s:s + 512] = (top2[:, 0] - top2[:, 1]).cpu().numpy()
        del lg, lp, top2
    return lps, ents, margs


@torch.inference_mode()
def task_frontier(key: str, temp: float = 0.0, seed: int = 0,
                  limit=None):
    cfg = MODELS[key]
    model, tok = load_model(key)
    tok.padding_side = "left"
    items = load_payload()[:limit]
    eos = eos_ids(tok)
    eos_set = set(eos)
    bs, max_new = cfg["bs"], cfg["max_new"]
    sample = temp > 0
    if sample:
        torch.manual_seed(seed)

    n_layers = model.config.num_hidden_layers
    pf_layers = [n_layers if l == "last" else int(l)
                 for l in cfg["prefill_layers"]]
    hid = model.config.hidden_size

    cols = {k: [] for k in ("text", "n_gen", "n_prompt", "hit_max",
                            "min_lp", "p10_lp", "mean_lp", "mean_entropy",
                            "mean_top2_margin")}
    tok_lp, tok_ent, tok_marg = [], [], []
    prefill = {l: np.zeros((len(items), hid), dtype=np.float32)
               for l in pf_layers}

    t0 = time.time()
    for s in range(0, len(items), bs):
        batch = items[s:s + bs]
        prompts = [chat(tok, [{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": it["problem"]}])
                   for it in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        plen = enc.input_ids.shape[1]

        # prefill side-capture (mean-pooled over real prompt tokens)
        pf = model(**enc, output_hidden_states=True).hidden_states
        mask = enc.attention_mask[:, :, None].float()
        for l in pf_layers:
            pooled = (pf[l].float() * mask).sum(1) / mask.sum(1)
            prefill[l][s:s + len(batch)] = pooled.cpu().numpy()
        del pf

        gen_kw = (dict(do_sample=True, temperature=temp, top_p=1.0)
                  if sample else dict(do_sample=False))
        out = model.generate(**enc, max_new_tokens=max_new,
                             pad_token_id=tok.pad_token_id, eos_token_id=eos,
                             output_scores=True, return_dict_in_generate=True,
                             **gen_kw)
        steps = torch.stack(out.scores, dim=1)        # (b, T, V) logits
        new = out.sequences[:, plen:]
        for b, it in enumerate(batch):
            ids = new[b].tolist()
            cut = len(ids)
            for j, t in enumerate(ids):
                if t in eos_set:
                    cut = j
                    break
            cols["n_prompt"].append(int(enc.attention_mask[b].sum()))
            cols["n_gen"].append(cut)
            cols["hit_max"].append(cut == max_new)
            cols["text"].append(tok.decode(new[b, :cut],
                                           skip_special_tokens=True))
            if cut == 0:
                for k in ("min_lp", "p10_lp", "mean_lp", "mean_entropy",
                          "mean_top2_margin"):
                    cols[k].append(float("nan"))
                tok_lp.append(np.zeros(0, np.float32))
                tok_ent.append(np.zeros(0, np.float32))
                tok_marg.append(np.zeros(0, np.float32))
                continue
            lps, ents, margs = token_features(steps[b, :cut], new[b, :cut])
            tok_lp.append(lps); tok_ent.append(ents); tok_marg.append(margs)
            k10 = max(1, int(np.ceil(0.10 * cut)))
            cols["min_lp"].append(float(lps.min()))
            cols["p10_lp"].append(float(np.sort(lps)[:k10].mean()))
            cols["mean_lp"].append(float(lps.mean()))
            cols["mean_entropy"].append(float(ents.mean()))
            cols["mean_top2_margin"].append(float(margs.mean()))
        del steps, out
        done = min(s + bs, len(items))
        el = time.time() - t0
        print(f"  {done}/{len(items)} ({el/done:.1f}s/p, "
              f"ETA {(len(items)-done)*el/done/60:.0f} min, "
              f"hit_max so far {sum(cols['hit_max'])})", flush=True)

    suffix = "" if not sample else f"_t{temp}_s{seed}"
    dump_json({"task": "frontier", "model_key": key,
               "model_name": cfg["name"], "max_new_tokens": max_new,
               "batch_size": bs, "decoding": ("greedy" if not sample else
                                              f"T={temp} seed={seed}"),
               "idx": [it["idx"] for it in items],
               "answer": [it["answer"] for it in items],
               "hit_max_frac": float(np.mean(cols["hit_max"])),
               "cost_tokens_generated": int(np.sum(cols["n_gen"])),
               "cost_tokens_prompt": int(np.sum(cols["n_prompt"])),
               **cols}, f"v8_frontier_{key}{suffix}.json.gz")

    OUT.mkdir(exist_ok=True)
    lens = np.array([len(a) for a in tok_lp], dtype=np.int64)
    npz = {"token_lens": lens,
           "token_lp_flat": (np.concatenate(tok_lp) if tok_lp
                             else np.zeros(0, np.float32)),
           "token_entropy_flat": (np.concatenate(tok_ent) if tok_ent
                                  else np.zeros(0, np.float32)),
           "token_top2margin_flat": (np.concatenate(tok_marg) if tok_marg
                                     else np.zeros(0, np.float32))}
    for l in pf_layers:
        tag = "last" if l == n_layers else f"l{l}"
        npz[f"prefill_mean_{tag}"] = prefill[l]
    f = OUT / f"v8_frontier_{key}{suffix}_arrays.npz"
    np.savez_compressed(f, **npz)
    print(f"-> {f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--temp", type=float, default=0.0,
                    help="0 = greedy (default); 0.6 only for the "
                         "pre-registered r1distill fallback")
    ap.add_argument("--seed", type=int, default=9999)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    task_frontier(args.model, temp=args.temp, seed=args.seed,
                  limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
