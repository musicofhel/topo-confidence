"""P11-FE233 — Label-free kernel score KS(Ker^(l)) per layer (weights-only).

Computes a structural, activation-free ranking of layers for
Qwen-2.5-1.5B-Instruct, following the Wang et al. App. G multi-head kernel
recipe:

    Ker^(l,h) = W_vo^(l-1,h) @ W_qk^(l,h) @ W_vo^(0),T

where, per attention head h,
    W_qk^(l,h) = W_q^(l,h).T @ W_k^(l,h)      (QK circuit, hidden x hidden)
    W_vo^(l,h) = W_o^(l,h)   @ W_v^(l,h)      (OV/VO circuit, hidden x hidden)

Heads are aggregated with the proposed mean-of-Trace formula:
    KS(l) = mean_h Trace(Ker^(l,h)).

GQA (Qwen-2.5 has 12 query heads / 2 KV heads) is handled by mapping each
query head h to its KV group g(h) = h // (n_heads // n_kv) for the K and V
weights. Layer indices are 0-based to match the residual-stream convention
(L19 == index 19), and we sweep L = 1 .. n_layers-1 (layer 0 has no
predecessor for the W_vo^(l-1) factor).

Rationale: a label-free, structural alternative to the AUROC sweep that fixed
L19 in F-2. If KS peaks at L19, F-2 gains a mechanistic anchor; if it peaks
elsewhere, re-extracting prefill DoM at the KS-peak layer is the falsifiable
follow-up.

No activations, no torch/transformers: weights are read directly from the
model's safetensors via a minimal numpy reader (BF16 -> float32 by bit shift).
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")

# Candidate locations for the Qwen-2.5-1.5B-Instruct weights (must be under ROOT).
MODEL_CANDIDATES = [
    ROOT / "models/Qwen2.5-1.5B-Instruct",
    ROOT / "models/qwen2.5-1.5b-instruct",
    ROOT / "pathway11_h100/models/Qwen2.5-1.5B-Instruct",
    ROOT / "data/models/Qwen2.5-1.5B-Instruct",
]

OUT_JSON = ROOT / "pathway11_h100/kernel_score/results.json"

L19 = 19  # the F-2 layer, 0-based index


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


# --- minimal safetensors reader (numpy-only, BF16-aware) --------------------

_DTYPE_MAP = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
}


def _load_header(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
    return header, 8 + n


def _read_tensor(path: Path, info: dict, data_start: int) -> np.ndarray:
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + start)
        buf = f.read(end - start)
    dtype = info["dtype"]
    if dtype == "BF16":
        raw = (np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16)
        arr = raw.view(np.float32)
    else:
        arr = np.frombuffer(buf, dtype=_DTYPE_MAP[dtype])
    return np.ascontiguousarray(arr.reshape(info["shape"]).astype(np.float32))


class SafeStore:
    """Resolves tensor names across one or more safetensors shards."""

    def __init__(self, model_dir: Path):
        idx = model_dir / "model.safetensors.index.json"
        single = model_dir / "model.safetensors"
        self.file_for_name: dict[str, Path] = {}
        if idx.exists():
            wm = json.loads(idx.read_text())["weight_map"]
            for name, fn in wm.items():
                self.file_for_name[name] = model_dir / fn
        elif single.exists():
            header, _ = _load_header(single)
            for name in header:
                if name != "__metadata__":
                    self.file_for_name[name] = single
        else:
            raise FileNotFoundError("no safetensors found")
        self._hdr_cache: dict[Path, tuple] = {}

    def _hdr(self, path: Path):
        if path not in self._hdr_cache:
            self._hdr_cache[path] = _load_header(path)
        return self._hdr_cache[path]

    def get(self, name: str) -> np.ndarray:
        path = self.file_for_name[name]
        header, data_start = self._hdr(path)
        return _read_tensor(path, header[name], data_start)


# --- per-layer circuit construction -----------------------------------------

def vo_heads(store: SafeStore, layer: int, n_heads: int, n_kv: int, hd: int):
    """Return list of W_vo^(l,h) = W_o^(l,h) @ W_v^(l,g(h)), each (hidden, hidden)."""
    p = f"model.layers.{layer}.self_attn"
    Wv = store.get(f"{p}.v_proj.weight")  # (n_kv*hd, hidden)
    Wo = store.get(f"{p}.o_proj.weight")  # (hidden, n_heads*hd)
    group = n_heads // n_kv
    out = []
    for h in range(n_heads):
        g = h // group
        Wv_h = Wv[g * hd:(g + 1) * hd, :]          # (hd, hidden)
        Wo_h = Wo[:, h * hd:(h + 1) * hd]          # (hidden, hd)
        out.append(Wo_h @ Wv_h)                    # (hidden, hidden)
    return out


def qk_heads(store: SafeStore, layer: int, n_heads: int, n_kv: int, hd: int):
    """Return list of W_qk^(l,h) = W_q^(l,h).T @ W_k^(l,g(h)), each (hidden, hidden)."""
    p = f"model.layers.{layer}.self_attn"
    Wq = store.get(f"{p}.q_proj.weight")  # (n_heads*hd, hidden)
    Wk = store.get(f"{p}.k_proj.weight")  # (n_kv*hd, hidden)
    group = n_heads // n_kv
    out = []
    for h in range(n_heads):
        g = h // group
        Wq_h = Wq[h * hd:(h + 1) * hd, :]          # (hd, hidden)
        Wk_h = Wk[g * hd:(g + 1) * hd, :]          # (hd, hidden)
        out.append(Wq_h.T @ Wk_h)                  # (hidden, hidden)
    return out


def main() -> int:
    model_dir = next(
        (d for d in MODEL_CANDIDATES
         if (d / "config.json").exists()
         and ((d / "model.safetensors").exists()
              or (d / "model.safetensors.index.json").exists())),
        None,
    )
    if model_dir is None:
        print("MISSING_REGEN_INPUT", "Qwen2.5-1.5B-Instruct weights", file=sys.stderr)
        for d in MODEL_CANDIDATES:
            print("  tried:", d, file=sys.stderr)
        return 2

    cfg = json.loads((model_dir / "config.json").read_text())
    hidden = int(cfg["hidden_size"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv = int(cfg.get("num_key_value_heads", n_heads))
    n_layers = int(cfg["num_hidden_layers"])
    hd = int(cfg.get("head_dim", hidden // n_heads))

    try:
        store = SafeStore(model_dir)
    except FileNotFoundError:
        print("MISSING_REGEN_INPUT", "safetensors", file=sys.stderr)
        return 2

    # Fixed reference factor: layer-0 VO circuit per head, transposed.
    vo0 = vo_heads(store, 0, n_heads, n_kv, hd)
    vo0T = [m.T.copy() for m in vo0]

    # Sweep L = 1 .. n_layers-1. vo_prev starts as the layer-0 VO circuit.
    vo_prev = vo0
    per_layer = []
    for L in range(1, n_layers):
        qk = qk_heads(store, L, n_heads, n_kv, hd)
        traces = np.empty(n_heads, dtype=np.float64)
        for h in range(n_heads):
            # Ker = vo_prev[h] @ qk[h] @ vo0T[h]; Trace(A@B@C) = sum(A * (B@C).T).
            BC = qk[h] @ vo0T[h]                    # (hidden, hidden), float32
            traces[h] = np.sum(vo_prev[h] * BC.T, dtype=np.float64)
        per_layer.append({
            "layer": L,
            "ks_mean_trace": float(traces.mean()),
            "ks_mean_abs_trace": float(np.abs(traces).mean()),
            "ks_mean_trace_norm": float(traces.mean() / hidden),
            "ks_std_trace": float(traces.std()),
        })
        # advance: this layer's VO becomes the predecessor for the next layer
        vo_prev = vo_heads(store, L, n_heads, n_kv, hd)

    layers = np.array([d["layer"] for d in per_layer])
    ks = np.array([d["ks_mean_trace"] for d in per_layer])
    order = np.argsort(-ks)
    ranked = layers[order].tolist()
    peak_layer = int(layers[int(np.argmax(ks))])

    l19_entry = next((d for d in per_layer if d["layer"] == L19), None)
    l19_rank = (ranked.index(L19) + 1) if L19 in ranked else None

    out = {
        "experiment": "P11-FE233",
        "model_dir": str(model_dir),
        "hidden": hidden,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "head_dim": hd,
        "num_layers": n_layers,
        "aggregation": "mean_of_Trace over heads (Wang et al. App. G)",
        "swept_layers": layers.tolist(),
        "per_layer": per_layer,
        "ranked_layers_by_ks": ranked,
        "peak_layer": peak_layer,
        "peak_ks": float(ks.max()),
        "L19_index": L19,
        "L19_ks_mean_trace": (float(l19_entry["ks_mean_trace"]) if l19_entry else None),
        "L19_rank": l19_rank,
        "L19_is_peak": bool(peak_layer == L19),
        "note": (
            "Label-free structural ranking. If peak_layer==19, F-2's L19 gains a "
            "mechanistic anchor; otherwise re-extract prefill DoM at peak_layer "
            "as a falsifiable extension."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())