"""P11-FE693 — Logit-lens JSD between prefill (L19) and final-token (L19) states.

F-3 reports cos(prefill_DoM, final_DoM)=0.046 — the two correctness directions
are near-orthogonal at the per-class direction level. DistillLens-style work
aligns early-and-late layers in *vocab* space, which suggests the orthogonality
may be a DoM-direction artifact rather than a property of the full distributional
structure. This script projects both prefill (L19) and final-token (L19) hidden
states through the model's tied embedding (used as the unembedding W_U), softmaxes
to the vocab simplex, and computes a per-problem Jensen-Shannon divergence.

Decision rule (per the FE rationale): if mean JSD < 0.3 nats (well below the
ceiling ln 2 ≈ 0.693), the prefill and final distributions are broadly shared
and F-3's 0.046 cosine is a DoM-direction artifact, not distributional
orthogonality.

CPU-only / no-network. The tied embedding matrix and the final-token hidden
states are not produced by the documented cache; they must be supplied as cached
NPZs (no model is loaded here). If absent, the script prints MISSING_REGEN_INPUT
and returns 2.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/logit_lens_jsd/results.json"

# Candidate locations for the final-token L19 hidden states (500, 1536).
FINAL_CANDIDATES = [
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz", ("final", "hidden", "final_hidden", "h_final")),
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz", ("final", "final_hidden", "h_final")),
    (ROOT / "pathway11_h100/data/m15b_final_l19.npz", ("final", "hidden")),
]

# Candidate locations for the tied embedding / unembedding matrix (V, 1536).
EMBED_CANDIDATES = [
    (ROOT / "pathway11_h100/data/qwen15b_embed.npz", ("embed", "WU", "unembed", "embed_tokens", "lm_head", "weight")),
    (ROOT / "pathway11_h100/data/m15b_embed.npz", ("embed", "WU", "unembed", "embed_tokens", "lm_head", "weight")),
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_embed.npz", ("embed", "WU", "unembed", "weight")),
]

# Candidate locations for the final-RMSNorm weight (1536,) — optional.
NORM_CANDIDATES = [
    (ROOT / "pathway11_h100/data/qwen15b_embed.npz", ("norm", "final_norm", "ln_f", "norm_weight")),
    (ROOT / "pathway11_h100/data/m15b_norm.npz", ("norm", "final_norm", "ln_f", "norm_weight")),
]

JSD_THRESHOLD = 0.3
COSINE_F3 = 0.046
EPS = 1e-12


def auroc(scores, labels):
    pos = scores[labels]; neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def _find(candidates, want_shape=None):
    """Return (array, path, key) for the first candidate file/key that exists."""
    for path, keys in candidates:
        if not path.exists():
            continue
        try:
            blob = np.load(path)
        except Exception:
            continue
        for k in keys:
            if k in blob.files:
                arr = blob[k]
                if want_shape is None or arr.shape == want_shape:
                    return np.asarray(arr), path, k
    return None, None, None


def rms_norm(H, weight, eps=1e-6):
    var = np.mean(H * H, axis=1, keepdims=True)
    return (H / np.sqrt(var + eps)) * weight[None, :]


def softmax_rows(logits):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def logit_lens_probs(H, WU, norm_weight):
    """H (n, d) -> vocab-simplex probabilities (n, V) via tied embedding W_U."""
    if norm_weight is not None:
        H = rms_norm(H, norm_weight)
    logits = H.astype(np.float32) @ WU.T.astype(np.float32)
    return softmax_rows(logits.astype(np.float64))


def jsd_rows(P, Q):
    """Per-row Jensen-Shannon divergence in nats. ceiling = ln 2 ≈ 0.693."""
    M = 0.5 * (P + Q)
    kl_pm = np.sum(np.where(P > EPS, P * (np.log(P + EPS) - np.log(M + EPS)), 0.0), axis=1)
    kl_qm = np.sum(np.where(Q > EPS, Q * (np.log(Q + EPS) - np.log(M + EPS)), 0.0), axis=1)
    return 0.5 * kl_pm + 0.5 * kl_qm


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2

    cache = np.load(CACHE)
    prefill = cache["prefill"].astype(np.float64)
    correct = cache["correct"].astype(bool)
    assert prefill.shape == (500, 1536) and correct.shape == (500,)

    final, final_path, final_key = _find(FINAL_CANDIDATES, want_shape=(500, 1536))
    if final is None:
        print("MISSING_REGEN_INPUT", "final-token L19 hidden states (500,1536)", file=sys.stderr)
        return 2
    final = final.astype(np.float64)

    WU, embed_path, embed_key = _find(EMBED_CANDIDATES)
    if WU is None:
        print("MISSING_REGEN_INPUT", "tied embedding / unembedding matrix (V,1536)", file=sys.stderr)
        return 2
    WU = np.asarray(WU)
    if WU.ndim != 2:
        print("MISSING_REGEN_INPUT", f"embedding matrix has ndim={WU.ndim}", file=sys.stderr)
        return 2
    # Orient so columns match the hidden dim (1536); rows are vocab.
    if WU.shape[1] != 1536 and WU.shape[0] == 1536:
        WU = WU.T
    if WU.shape[1] != 1536:
        print("MISSING_REGEN_INPUT", f"embedding hidden dim mismatch {WU.shape}", file=sys.stderr)
        return 2

    norm_weight, norm_path, norm_key = _find(NORM_CANDIDATES, want_shape=(1536,))
    if norm_weight is not None:
        norm_weight = norm_weight.astype(np.float64)

    P_prefill = logit_lens_probs(prefill, WU, norm_weight)
    P_final = logit_lens_probs(final, WU, norm_weight)

    jsd = jsd_rows(P_prefill, P_final)

    # Direction-level reference: per-class DoM cosine in raw hidden space.
    d_prefill = prefill[correct].mean(0) - prefill[~correct].mean(0)
    d_final = final[correct].mean(0) - final[~correct].mean(0)
    cos_dom = float(d_prefill @ d_final / (np.linalg.norm(d_prefill) * np.linalg.norm(d_final) + EPS))

    # Does the distributional JSD itself carry correctness signal?
    auroc_jsd = auroc(jsd, correct)

    mean_jsd = float(np.mean(jsd))
    out = {
        "experiment": "P11-FE693",
        "vocab_size": int(WU.shape[0]),
        "final_source": f"{final_path.name}:{final_key}",
        "embed_source": f"{embed_path.name}:{embed_key}",
        "rms_norm_applied": norm_weight is not None,
        "jsd_ceiling_nats": float(np.log(2.0)),
        "mean_jsd_nats": mean_jsd,
        "median_jsd_nats": float(np.median(jsd)),
        "std_jsd_nats": float(np.std(jsd)),
        "min_jsd_nats": float(np.min(jsd)),
        "max_jsd_nats": float(np.max(jsd)),
        "jsd_threshold": JSD_THRESHOLD,
        "jsd_below_threshold": bool(mean_jsd < JSD_THRESHOLD),
        "cos_prefill_final_dom_raw": cos_dom,
        "cos_f3_reference": COSINE_F3,
        "auroc_jsd_vs_correct": float(auroc_jsd),
        "verdict": (
            "DOM_DIRECTION_ARTIFACT" if mean_jsd < JSD_THRESHOLD
            else "DISTRIBUTIONAL_ORTHOGONALITY_SURVIVES"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())