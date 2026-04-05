"""Extract hidden states from HuggingFace causal language models."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from topo_confidence.utils import resolve_device, resolve_layers, timer

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class HiddenStateExtractor:
    """Extract hidden states from any HuggingFace causal LM.

    Handles model loading, tokenization, forward pass, and hidden state
    collection. Returns per-token hidden states at specified layers.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "float16",
        layers: list[int] | str = "last",
    ):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.dtype = DTYPE_MAP.get(dtype, torch.float16)
        self._layer_spec = layers

        with timer(f"Loading {model_name}"):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=self.dtype,
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True,
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()

        n_layers = self.model.config.num_hidden_layers
        self.layers = resolve_layers(self._layer_spec, n_layers)
        logger.info(
            "Loaded %s (%d layers, extracting %s)", model_name, n_layers, self.layers
        )

    @torch.no_grad()
    def extract(
        self, prompts: list[str], max_length: int = 512, batch_size: int = 8
    ) -> dict[str, Any]:
        """Extract hidden states for a batch of prompts.

        Returns the hidden state at each token position for each specified layer.
        The primary output is the per-prompt token trajectories (used for PH).

        Returns:
            {
                "hidden_states": np.ndarray of shape (n_prompts, hidden_dim)
                    — last-token hidden state at the final extracted layer.
                "token_trajectories": list[np.ndarray]
                    — per-prompt list of (n_tokens, hidden_dim) arrays.
                "token_counts": np.ndarray of shape (n_prompts,)
            }
        """
        all_last_hidden = []
        all_trajectories = []
        all_token_counts = []

        for i in range(0, len(prompts), batch_size):
            if i % 50 == 0:
                logger.info("Extracting hidden states %d/%d...", i, len(prompts))
            batch = prompts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.model.device)

            outputs = self.model(**inputs, output_hidden_states=True)

            # hidden_states is a tuple of (n_layers+1,) tensors of (batch, seq, hidden)
            # Index 0 is embeddings, 1..n_layers are layer outputs
            layer_idx = self.layers[-1] + 1  # +1 because index 0 is embeddings
            hidden = outputs.hidden_states[layer_idx]  # (batch, seq, hidden)

            attention_mask = inputs["attention_mask"]  # (batch, seq)

            for j in range(len(batch)):
                mask = attention_mask[j].bool()
                tokens_hidden = hidden[j][mask].cpu().float().numpy()  # (n_real_tokens, hidden)
                all_trajectories.append(tokens_hidden)
                all_last_hidden.append(tokens_hidden[-1])  # last real token
                all_token_counts.append(len(tokens_hidden))

        return {
            "hidden_states": np.stack(all_last_hidden),
            "token_trajectories": all_trajectories,
            "token_counts": np.array(all_token_counts),
        }

    @torch.no_grad()
    def extract_with_output(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        max_length: int = 512,
        batch_size: int = 8,
        collect_logits: bool = False,
    ) -> dict[str, Any]:
        """Extract hidden states AND generate output text.

        Uses two passes to avoid storing hidden states during generation:
        1. Forward pass: extract prompt hidden states (fast, batched)
        2. Generation pass: generate text (+ optionally output logits)

        Returns:
            {
                "hidden_states": np.ndarray (n_prompts, hidden_dim),
                "token_trajectories": list[np.ndarray],
                "token_counts": np.ndarray,
                "generated_texts": list[str],
                "output_logits": list[np.ndarray],  # only if collect_logits=True
            }
        """
        # Pass 1: extract hidden states (batched)
        extraction = self.extract(prompts, max_length=max_length, batch_size=batch_size)

        # Pass 2: generate outputs one at a time
        all_texts = []
        all_logits = []

        for i, prompt in enumerate(prompts):
            if i % 50 == 0:
                logger.info("Generating %d/%d...", i, len(prompts))

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self.model.device)

            prompt_len = inputs["input_ids"].shape[1]

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            }
            if collect_logits:
                gen_kwargs["output_scores"] = True
                gen_kwargs["return_dict_in_generate"] = True

            outputs = self.model.generate(**inputs, **gen_kwargs)

            if collect_logits:
                generated_ids = outputs.sequences[0][prompt_len:]
                if outputs.scores:
                    logits = torch.stack(outputs.scores, dim=0)
                    all_logits.append(logits.cpu().float().numpy())
                else:
                    all_logits.append(np.array([]))
            else:
                generated_ids = outputs[0][prompt_len:]

            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            all_texts.append(text)

        return {
            "hidden_states": extraction["hidden_states"],
            "token_trajectories": extraction["token_trajectories"],
            "token_counts": extraction["token_counts"],
            "generated_texts": all_texts,
            "output_logits": all_logits,
        }
