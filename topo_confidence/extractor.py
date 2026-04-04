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
                torch_dtype=self.dtype,
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
    ) -> dict[str, Any]:
        """Extract hidden states AND generate output text.

        Processes prompts one at a time for generation (batch generation
        with variable-length outputs is fragile).

        Returns:
            {
                "hidden_states": np.ndarray (n_prompts, hidden_dim),
                "token_trajectories": list[np.ndarray],
                "token_counts": np.ndarray,
                "generated_texts": list[str],
                "output_logits": list[np.ndarray],  # per-token logits for baselines
            }
        """
        all_last_hidden = []
        all_trajectories = []
        all_token_counts = []
        all_texts = []
        all_logits = []

        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self.model.device)

            prompt_len = inputs["input_ids"].shape[1]

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_hidden_states=True,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

            # Extract prompt hidden states from the first forward pass
            # outputs.hidden_states[0] is the prefill step: tuple of (n_layers+1,) tensors
            layer_idx = self.layers[-1] + 1
            prompt_hidden = (
                outputs.hidden_states[0][layer_idx][0].cpu().float().numpy()
            )  # (prompt_len, hidden)

            all_trajectories.append(prompt_hidden)
            all_last_hidden.append(prompt_hidden[-1])
            all_token_counts.append(len(prompt_hidden))

            # Decode generated text (excluding prompt)
            generated_ids = outputs.sequences[0][prompt_len:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            all_texts.append(text)

            # Collect output logits for baseline methods
            if outputs.scores:
                logits = torch.stack(outputs.scores, dim=0)  # (gen_len, vocab)
                all_logits.append(logits.cpu().float().numpy())
            else:
                all_logits.append(np.array([]))

        return {
            "hidden_states": np.stack(all_last_hidden),
            "token_trajectories": all_trajectories,
            "token_counts": np.array(all_token_counts),
            "generated_texts": all_texts,
            "output_logits": all_logits,
        }
