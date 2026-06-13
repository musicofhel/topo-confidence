#!/usr/bin/env python3
"""v8_runpod_distill.py — SPEC v8 Phase 2 pod workload (H100 SXM, bf16).

Three tasks (run in order; pull pod_results/ after each):

  teacher --model {math7b,qwen7b}
      Greedy 1024-token traces on pod_payload/v8_math_train_4000.jsonl.gz
      with gen-time scalar capture (n_gen, mean_lp — the gate filter's
      features). Teacher identity is FROZEN at Gate G1 (H-N conditional),
      passed here as an argument, never chosen pod-side.

  sft --arm {a,b,c,d}
      LoRA SFT of Qwen/Qwen2.5-1.5B-Instruct on
      pod_payload/v8_traces_{arm}.jsonl.gz (filter arms built LOCALLY from
      teacher output; matched trace count N). Frozen recipe: rank 16, alpha
      32, dropout 0.05, LR 1e-4 cosine, 2 epochs, effective batch 32 (8x4),
      max_seq 1024, bf16, seed 0, loss on completion tokens only, FINAL
      checkpoint (no selection). Adapter saved to /workspace/adapters/{arm}.

  evalsft --arm {a,b,c,d}
      G2 single-shot eval of the arm's final checkpoint: greedy MATH-500
      (1024) with gen-time features (post-SFT gate refit) + greedy BBH-750
      (forgetting guardrail). Texts stored; MATH grading happens locally
      with the pinned grader. BBH correctness computed pod-side with the
      pinned v7 extraction patterns (same as v7_runpod_arms.py).
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PAYLOAD = HERE / "pod_payload"
OUT = HERE / "pod_results"
ADAPTERS = HERE / "adapters"

SYSTEM_PROMPT = ("Please reason step by step, and put your final answer "
                 "within \\boxed{}.")
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
TEACHERS = {"math7b": "Qwen/Qwen2.5-Math-7B-Instruct",
            "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}

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


def bbh_extract_answer(response: str) -> str:
    for pat in _BBH_ANSWER_PATS:
        m = pat.findall(response)
        if m:
            ans = m[-1].strip()
            if len(ans) == 1 and ans.isalpha() and ans.upper() != "N":
                return f"({ans.upper()})"
            return ans.capitalize() if ans.lower() in ("yes", "no") else ans
    return ""


def chat(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True)


def load_jsonl(path: Path):
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _dtype_kw():
    import transformers
    return ("dtype" if int(transformers.__version__.split(".")[0]) >= 5
            else "torch_dtype")


def load_model(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, attn_implementation="sdpa",
        device_map="cuda", **{_dtype_kw(): torch.bfloat16}).eval()
    print(f"loaded {name}", flush=True)
    return model, tok


def eos_ids(tok):
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
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
def greedy_capture(model, tok, prompts, max_new, bs):
    """Greedy batch generation with gen-time scalar capture (D-4 method)."""
    eos = eos_ids(tok)
    eos_set = set(eos)
    res = {k: [] for k in ("text", "n_gen", "n_prompt", "mean_lp", "hit_max")}
    t0 = time.time()
    for s in range(0, len(prompts), bs):
        chunk = prompts[s:s + bs]
        enc = tok(chunk, return_tensors="pt", padding=True).to("cuda")
        plen = enc.input_ids.shape[1]
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=eos,
                             output_scores=True, return_dict_in_generate=True)
        steps = torch.stack(out.scores, dim=1)
        new = out.sequences[:, plen:]
        for b in range(len(chunk)):
            ids = new[b].tolist()
            cut = len(ids)
            for j, t in enumerate(ids):
                if t in eos_set:
                    cut = j
                    break
            res["n_prompt"].append(int(enc.attention_mask[b].sum()))
            res["n_gen"].append(cut)
            res["hit_max"].append(cut == max_new)
            res["text"].append(tok.decode(new[b, :cut],
                                          skip_special_tokens=True))
            if cut == 0:
                res["mean_lp"].append(float("nan"))
                continue
            lp = torch.log_softmax(steps[b, :cut].float(), dim=-1)
            lps = lp.gather(1, new[b, :cut, None])[:, 0]
            res["mean_lp"].append(float(lps.mean()))
            del lp
        del steps, out
        done = min(s + bs, len(prompts))
        el = time.time() - t0
        print(f"  {done}/{len(prompts)} ({el/done:.1f}s/p, "
              f"ETA {(len(prompts)-done)*el/done/60:.0f} min)", flush=True)
    return res


def task_teacher(model_key: str, limit=None, bs=16):
    name = TEACHERS[model_key]
    model, tok = load_model(name)
    tok.padding_side = "left"
    items = load_jsonl(PAYLOAD / "v8_math_train_4000.jsonl.gz")[:limit]
    prompts = [chat(tok, [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": it["problem"]}])
               for it in items]
    res = greedy_capture(model, tok, prompts, max_new=1024, bs=bs)
    dump_json({"task": "teacher", "model_key": model_key, "model_name": name,
               "idx": [it["idx"] for it in items],
               "level": [it["level"] for it in items],
               "cost_tokens_generated": int(np.sum(res["n_gen"])), **res},
              f"v8_teacher_{model_key}.json.gz")


def task_sft(arm: str):
    import random
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, attn_implementation="sdpa", device_map="cuda",
        **{_dtype_kw(): torch.bfloat16})
    model.config.use_cache = False

    lconf = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                       task_type="CAUSAL_LM",
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lconf)
    model.print_trainable_parameters()

    traces = load_jsonl(PAYLOAD / f"v8_traces_{arm}.jsonl.gz")
    print(f"arm {arm}: {len(traces)} traces", flush=True)

    class SFTSet(Dataset):
        def __len__(self):
            return len(traces)

        def __getitem__(self, i):
            it = traces[i]
            prompt = chat(tok, [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": it["problem"]}])
            p_ids = tok(prompt, add_special_tokens=False).input_ids
            c_ids = tok(it["text"] + tok.eos_token,
                        add_special_tokens=False).input_ids
            ids = (p_ids + c_ids)[:1024]
            labels = ([-100] * len(p_ids) + c_ids)[:1024]
            return {"input_ids": ids, "labels": labels}

    def collate(batch):
        mx = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        return {"input_ids": torch.tensor(
                    [b["input_ids"] + [pad] * (mx - len(b["input_ids"]))
                     for b in batch]),
                "attention_mask": torch.tensor(
                    [[1] * len(b["input_ids"]) +
                     [0] * (mx - len(b["input_ids"])) for b in batch]),
                "labels": torch.tensor(
                    [b["labels"] + [-100] * (mx - len(b["labels"]))
                     for b in batch])}

    args = TrainingArguments(
        output_dir=str(HERE / f"sft_tmp_{arm}"),
        per_device_train_batch_size=8, gradient_accumulation_steps=4,
        num_train_epochs=2, learning_rate=1e-4, lr_scheduler_type="cosine",
        warmup_ratio=0.03, bf16=True, logging_steps=10, save_strategy="no",
        seed=0, report_to=[])
    Trainer(model=model, args=args, train_dataset=SFTSet(),
            data_collator=collate).train()

    ADAPTERS.mkdir(exist_ok=True)
    model.save_pretrained(str(ADAPTERS / arm))
    print(f"-> adapter saved: {ADAPTERS / arm}", flush=True)


def _eval_math_bbh(model, tok, tag: str, limit=None, bs=8):
    """Shared MATH-500 + BBH-750 greedy eval (used by evalsft and evalbase)."""
    tok.padding_side = "left"
    math = load_jsonl(PAYLOAD / "v8_math500.jsonl.gz")[:limit]
    prompts = [chat(tok, [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": it["problem"]}])
               for it in math]
    res = greedy_capture(model, tok, prompts, max_new=1024, bs=bs)
    dump_json({"task": "evalsft_math", "arm": tag,
               "idx": [it["idx"] for it in math],
               "answer": [it["answer"] for it in math],
               "cost_tokens_generated": int(np.sum(res["n_gen"])), **res},
              f"v8_eval{tag}_math.json.gz" if tag == "base"
              else f"v8_evalsft_{tag}_math.json.gz")

    bbh = load_jsonl(PAYLOAD / "qwen1.5b_bbh.jsonl.gz")[:limit]
    bprompts = []
    for it in bbh:
        cot = (PAYLOAD / "bbh_cot_prompts" / f"{it['subset']}.txt").read_text()
        user = f"{cot}\n\nQ: {it['question']}\nA: Let's think step by step."
        bprompts.append(chat(tok, [{"role": "user", "content": user}]))
    bres = greedy_capture(model, tok, bprompts, max_new=1024, bs=bs)
    answers = [bbh_extract_answer(t) for t in bres["text"]]
    correct = [a != "" and a == it["target"] for a, it in zip(answers, bbh)]
    print(f"[{tag}] BBH acc: {np.mean(correct):.4f}", flush=True)
    dump_json({"task": "evalsft_bbh", "arm": tag,
               "idx": [it["idx"] for it in bbh],
               "subset": [it["subset"] for it in bbh],
               "answers": answers, "correct": correct,
               "cost_tokens_generated": int(np.sum(bres["n_gen"])),
               **{k: v for k, v in bres.items() if k != "text"}},
              f"v8_eval{tag}_bbh.json.gz" if tag == "base"
              else f"v8_evalsft_{tag}_bbh.json.gz")


def task_evalbase(limit=None, bs=8):
    """Un-adapted BASE reference: BBH-750 forgetting baseline + MATH-500
    re-confirmation of the pinned 0.486. Not part of the SFT recipe."""
    model, tok = load_model(BASE)
    _eval_math_bbh(model, tok, "base", limit, bs)


def task_evalsft(arm: str, limit=None, bs=8):
    from peft import PeftModel
    base, tok = load_model(BASE)
    model = PeftModel.from_pretrained(base, str(ADAPTERS / arm)).eval()
    _eval_math_bbh(model, tok, arm, limit, bs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["teacher", "sft", "evalsft", "evalbase"])
    ap.add_argument("--model", default=None, choices=sorted(TEACHERS))
    ap.add_argument("--arm", default=None, choices=["a", "b", "c", "d"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.task == "teacher":
        assert args.model, "--model required (frozen at G1)"
        task_teacher(args.model, args.limit)
    elif args.task == "sft":
        assert args.arm
        task_sft(args.arm)
    elif args.task == "evalbase":
        task_evalbase(args.limit)
    else:
        assert args.arm
        task_evalsft(args.arm, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
