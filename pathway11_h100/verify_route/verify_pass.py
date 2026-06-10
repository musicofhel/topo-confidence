#!/usr/bin/env python3
"""S2 — Verification pass + PANL activation capture (GPU).

For each problem we present the model with its OWN K=1 greedy answer and ask it to
self-verify. We capture two things:

  (1) verbalized verdict: greedy "Yes"/"No" to "Is the final answer correct?"
  (2) PANL-equivalent activation: the residual stream at ALL layers at the LAST prompt
      token (the post-answer judgment position whose output predicts the first verdict
      token). This is our post-hoc analog of Kumaran 2604.22271's post-answer-newline
      signal. Shape per problem: (n_layers+1, 1536) = (29, 1536).

Capture is memory-light: a forward hook stores ONLY the last-token slice of each layer's
PREFILL output (seq_len>1 forward), never the full hidden-state tensor — critical on 8 GB.
Same hidden_states[k] = output of model.model.layers[k-1] indexing as FE269; we additionally
keep the embedding output via a hook on model.model.embed_tokens -> index 0.

Outputs:
  results/verify.json     : {idx, verdict_text, verdict_bool} per problem
  results/verify_acts.npz : acts (N, n_layers+1, 1536) f16, idx (N,), correct_k1 (N,) bool
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
from common import load_model, format_chat, _extract_boxed  # noqa: E402

ROOT = Path("/home/musicofhel/topo-confidence")
N = 500


def build_verify_prompt(problem: str, k1_answer_text: str, tokenizer) -> str:
    """Present problem + the model's own boxed answer, ask Yes/No."""
    boxed = _extract_boxed(k1_answer_text)
    user = (
        f"Problem:\n{problem}\n\n"
        f"A proposed solution gives the final answer: \\boxed{{{boxed}}}\n\n"
        f"Is this final answer correct? Reply with exactly one word: Yes or No."
    )
    # plain system prompt (no \boxed instruction needed for a yes/no)
    return format_chat(user, tokenizer, system="You are a careful math grader.")


class LastTokenCapture:
    """Forward hooks storing the last-token activation of the PREFILL forward only.

    Captures embed_tokens output (index 0) + each decoder layer output (1..n_layers),
    matching hidden_states[k] = layers[k-1]. Only fires on seq_len>1 (prefill); decode
    steps (seq_len==1) are skipped. Re-armed per batch via .reset().
    """

    def __init__(self, model):
        self.model = model
        self.n_layers = len(model.model.layers)
        self.buf: dict[int, torch.Tensor] = {}
        self._handles = []

    def _mk(self, idx):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if t.dim() == 3 and t.shape[1] > 1 and idx not in self.buf:
                self.buf[idx] = t[:, -1, :].detach().float().cpu()
        return hook

    def __enter__(self):
        self._handles.append(self.model.model.embed_tokens.register_forward_hook(self._mk(0)))
        for k, layer in enumerate(self.model.model.layers):
            self._handles.append(layer.register_forward_hook(self._mk(k + 1)))
        return self

    def reset(self):
        self.buf = {}

    def collect(self, batch_size: int) -> np.ndarray:
        """Return (batch, n_layers+1, 1536) for the captured prefill."""
        layers = [self.buf[k] for k in range(self.n_layers + 1)]
        return torch.stack(layers, dim=1).numpy().astype(np.float16)  # (B, L+1, D)

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=6)
    args = ap.parse_args()

    # K=1 answers (from S1) + the MATH-500 problems (from stage2 cache via datasets)
    k1 = json.loads((HERE / "results" / "k1_greedy.json").read_text())["rows"]
    # problem statements: load MATH-500 in the SAME order pathway8 used (HF test split)
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = [ds[i]["problem"] for i in range(N)]

    model, tok = load_model()

    prompts = [build_verify_prompt(problems[r["idx"]], r["text"], tok) for r in k1]
    eos_ids = [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]

    verdict_texts: list[str] = []
    acts_all: list[np.ndarray] = []

    cap = LastTokenCapture(model)
    with cap:
        for s in range(0, len(prompts), args.batch):
            chunk = prompts[s : s + args.batch]
            enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
            cap.reset()
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos_ids,
            )
            new = gen[:, enc.input_ids.shape[1] :]
            verdict_texts.extend(tok.batch_decode(new, skip_special_tokens=True))
            acts_all.append(cap.collect(len(chunk)))
            print(f"  verify {s + len(chunk)}/{len(prompts)}", flush=True)

    acts = np.concatenate(acts_all, axis=0)  # (N, L+1, D)

    def parse_verdict(t: str) -> bool:
        tl = t.strip().lower()
        # True == model says correct ("yes"); be robust to leading punctuation
        if tl.startswith("yes") or "yes" in tl[:8]:
            return True
        if tl.startswith("no") or "no" in tl[:8]:
            return False
        return False  # default: treat ambiguous as "not verified correct"

    rows = []
    for r, t in zip(k1, verdict_texts):
        rows.append({"idx": r["idx"], "verdict_text": t, "verdict_bool": parse_verdict(t)})

    idx = np.array([r["idx"] for r in k1], dtype=int)
    correct_k1 = np.array([r["correct"] for r in k1], dtype=bool)

    # sanity
    assert acts.shape == (len(k1), len(model.model.layers) + 1, 1536), acts.shape
    assert np.isfinite(acts).all(), "non-finite activations"
    stds = acts.astype(np.float32).std(axis=0).mean(axis=1)  # per-layer mean std
    # slot 0 = embed_tokens of the final (fixed) generation-prompt token -> constant by
    # construction (every verification prompt ends with the same chat-template suffix).
    # That is expected and harmless; only the decoder-layer slots (1..n) carry signal.
    degenerate = np.where(stds <= 0)[0].tolist()
    if degenerate:
        print(f"  note: zero-variance layer slots {degenerate} (slot 0 = constant embedding, expected)")
    assert all(s == 0 for s in degenerate), \
        f"a DECODER layer (slot>0) is degenerate: {degenerate}"  # only slot 0 may be constant

    (HERE / "results" / "verify.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False))
    np.savez_compressed(
        HERE / "results" / "verify_acts.npz",
        acts=acts, idx=idx, correct_k1=correct_k1,
    )

    vb = np.array([r["verdict_bool"] for r in rows])
    # verbalized-verdict diagnostic: does "Yes" track actual K=1 correctness?
    agree = (vb == correct_k1).mean()
    print(f"\n  verbalized verdict says-correct rate: {vb.mean():.3f}  (true K=1 acc {correct_k1.mean():.3f})")
    print(f"  verbalized verdict accuracy vs truth: {agree:.3f}")
    print(f"  acts shape {acts.shape}, per-layer std range [{stds.min():.3f}, {stds.max():.3f}]")
    print("  wrote results/verify.json + results/verify_acts.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
