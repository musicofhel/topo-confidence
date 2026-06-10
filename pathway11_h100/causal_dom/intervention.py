#!/usr/bin/env python3
"""FE269 — directional intervention via forward hooks (Arditi et al. 2406.11717).

A direction r_hat "encodes" a behaviour iff ablating it removes the behaviour AND
adding it induces the behaviour, with general capability preserved.

  ablate:  h' = h - (h·r_hat) r_hat      # project the DoM component OUT, per position
  add:     h' = h + alpha * r_hat        # inject the DoM component
  off:     h' = h                        # identity (plumbing gate)

LAYER SEMANTICS (verified from source): hidden_states[19] = output of
model.model.layers[18]. So "ablate/add at L19" => hook model.model.layers[18].
"every layer" (Arditi ablation recipe) => hook model.model.layers[0..27].

TRANSFORMERS 4.57.6 QUIRK (verified from installed source): Qwen2DecoderLayer.forward
returns a BARE TENSOR, not a tuple. The hook is shape-defensive: it edits out[0] when
out is a tuple, else edits the bare tensor.

KV-cache consistency: hooking layer OUTPUTS means the edited residual stream feeds the
next layer's K/V on every forward (prefill + each decode step) — the cache reflects
edited activations, no manual cache surgery needed.
"""
from __future__ import annotations

import torch


class DirectionIntervention:
    """Context manager that registers shape-defensive forward hooks.

    Args:
        model: HF causal LM with .model.layers
        r_hat: (d,) unit direction (np array or tensor)
        mode:  "off" | "ablate" | "add"
        alpha: scalar for "add" (raw units; caller scales by ||r|| externally)
        layers: iterable of decoder-layer indices to hook. For "L19" pass [18];
                for "all" pass range(n_layers).
    """

    def __init__(self, model, r_hat, mode="off", alpha=0.0, layers=None):
        self.model = model
        self.r = torch.as_tensor(r_hat).float().flatten()
        assert mode in ("off", "ablate", "add"), mode
        self.mode = mode
        self.alpha = float(alpha)
        n_layers = len(model.model.layers)
        self.layers = list(range(n_layers)) if layers is None else list(layers)
        self._handles: list = []

    # --- core edit ---
    def _edit(self, h: torch.Tensor) -> torch.Tensor:
        r = self.r.to(device=h.device, dtype=h.dtype)
        if self.mode == "ablate":
            coeff = torch.einsum("...d,d->...", h, r)        # (..., seq)
            return h - coeff.unsqueeze(-1) * r
        if self.mode == "add":
            return h + self.alpha * r
        return h

    def _hook(self, module, inp, out):
        if self.mode == "off":
            return out
        if isinstance(out, tuple):
            return (self._edit(out[0]),) + tuple(out[1:])
        return self._edit(out)                                # bare Tensor (4.57.6)

    def __enter__(self):
        layers_mod = self.model.model.layers
        for k in self.layers:
            self._handles.append(layers_mod[k].register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def _self_test():
    """Verification #3: ablate removes the projection; add shifts it by alpha."""
    torch.manual_seed(0)
    d = 1536
    r = torch.randn(d); r = r / r.norm()
    h = torch.randn(4, 7, d)

    iv = DirectionIntervention.__new__(DirectionIntervention)
    iv.r = r; iv.mode = "ablate"; iv.alpha = 0.0
    h_ab = iv._edit(h)
    resid = torch.einsum("...d,d->...", h_ab, r).abs().max().item()
    print(f"ablate: max |h_edited · r_hat| = {resid:.3e}  (expect ~0)")
    assert resid < 1e-4

    iv.mode = "add"; iv.alpha = 3.0
    h_add = iv._edit(h)
    delta = (torch.einsum("...d,d->...", h_add, r) - torch.einsum("...d,d->...", h, r))
    err = (delta - 3.0).abs().max().item()
    print(f"add:    max |Δ(h·r_hat) - alpha| = {err:.3e}  (expect ~0)")
    assert err < 1e-4
    print("intervention self-test PASSED")


if __name__ == "__main__":
    _self_test()
