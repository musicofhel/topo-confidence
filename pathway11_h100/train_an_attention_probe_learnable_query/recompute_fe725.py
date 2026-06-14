"""P11-FE725 — Attention Probe (ReLope-style) over preceding token-positions and layers.

ReLope's central claim is that correctness signal is distributed across preceding
prompt tokens and adjacent layers, and that a learnable soft-aggregation lifts AUROC
~1pt over a single-token baseline. This script trains an Attention Probe — a learnable
query q producing softmax weights beta_i over the (layer x position) token grid — whose
aggregated vector feeds a 5-layer MLP head. Logit = MLP( sum_i beta_i * h_i ).

It aggregates across layers L18-L20 and the last K in {8, 16, 32} prompt positions,
scores 5-fold OOF AUROC, and compares against F-2's single-token L19 DoM (0.7731) and,
if available, against P11-FE724.

Direct test of F-3's orthogonality framing (cos(prefill_DoM, final_DoM) = 0.046): we
compute the aggregated discriminative direction (attention-pooled DoM) and measure its
cosine with the L19 prefill DoM and, if a final-token cache exists, the final-token DoM.
If the aggregated probe beats the single-token baseline AND its direction lies near both
prefill and final DoM (cos > 0.4 with each), F-3 collapses to "one signal aliased
differently by single-token probes."

Required input is a per-layer / per-position prefill cache (multi-token, multi-layer).
The documented single-token L19 cache (m15b_prefill.npz) does NOT carry the preceding
token positions or adjacent layers this experiment needs, so if the richer cache is
absent we emit MISSING_REGEN_INPUT and exit 2.
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
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
# Richer per-(layer, position) prefill cache required by the attention probe.
TOKEN_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_tokens.npz"
# Optional final-token cache for the F-3 orthogonality test.
FINAL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz"
# Optional sibling experiment for head-to-head reference.
FE724_JSON = ROOT / "pathway11_h100/attention_probe_fe724/results.json"
OUT_JSON = ROOT / "pathway11_h100/attention_probe/results.json"

N_FOLDS = 5
SEED = 9999
TARGET_LAYERS = (18, 19, 20)
K_VALUES = (8, 16, 32)
HIDDEN = 48          # MLP head width
N_MLP_LAYERS = 5     # 5 weight matrices: d->h->h->h->h->1
EPOCHS = 120
LR = 2e-3
WD = 1e-4
F2_BASELINE = 0.7731


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _softmax_rows(s: np.ndarray) -> np.ndarray:
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def load_token_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (hidden, layers, positions, correct).

    hidden is (N, L, P, d): N examples, L stored layers, P stored preceding
    positions, d=1536 features. Defensive about key names / axis order.
    """
    blob = np.load(TOKEN_CACHE, allow_pickle=False)
    keys = set(blob.files)
    hkey = next((k for k in ("hidden", "hidden_states", "tokens", "grid") if k in keys), None)
    if hkey is None:
        print("MISSING_REGEN_INPUT", TOKEN_CACHE, "(no hidden grid key)", file=sys.stderr)
        return None
    H = blob[hkey].astype(np.float32)
    if H.ndim != 4:
        print("MISSING_REGEN_INPUT", TOKEN_CACHE, f"(hidden ndim={H.ndim}, want 4)", file=sys.stderr)
        return None
    # Orient so the feature axis (d=1536) is last.
    d_axis = int(np.argmax([1 if H.shape[ax] == 1536 else 0 for ax in range(4)]))
    if H.shape[d_axis] != 1536:
        print("MISSING_REGEN_INPUT", TOKEN_CACHE, "(no 1536 feature axis)", file=sys.stderr)
        return None
    if d_axis != 3:
        H = np.moveaxis(H, d_axis, 3)
    # Axis 0 is N (==500); remaining are (L, P) in stored order.
    correct = blob["correct"].astype(bool) if "correct" in keys else None
    if correct is None:
        print("MISSING_REGEN_INPUT", TOKEN_CACHE, "(no correct labels)", file=sys.stderr)
        return None
    layers = blob["layers"].astype(int) if "layers" in keys else np.array([], dtype=int)
    positions = blob["positions"].astype(int) if "positions" in keys else np.arange(H.shape[2])
    return H, layers, positions, correct


def select_grid(H: np.ndarray, layers: np.ndarray, K: int) -> np.ndarray | None:
    """Slice to TARGET_LAYERS and the last K positions, return (N, T, d), T=L*K."""
    N, L, P, d = H.shape
    if layers.size == L:
        keep = [i for i, lyr in enumerate(layers) if int(lyr) in TARGET_LAYERS]
    else:
        # No layer metadata: assume the stored layers are centered on L19; take up to 3.
        mid = L // 2
        keep = sorted({max(0, mid - 1), mid, min(L - 1, mid + 1)})
    if not keep:
        return None
    if K > P:
        return None
    sub = H[:, keep, :, :][:, :, P - K:, :]          # (N, |keep|, K, d)
    return sub.reshape(N, len(keep) * K, d)            # (N, T, d)


def init_params(d: int, rng: np.random.Generator) -> dict:
    q = rng.standard_normal(d).astype(np.float64) * (1.0 / np.sqrt(d))
    dims = [d] + [HIDDEN] * (N_MLP_LAYERS - 1) + [1]
    Ws, bs = [], []
    for a, b in zip(dims[:-1], dims[1:]):
        Ws.append(rng.standard_normal((a, b)) * np.sqrt(2.0 / a))
        bs.append(np.zeros(b))
    return {"q": q, "Ws": Ws, "bs": bs}


def attention_pool(X: np.ndarray, q: np.ndarray, scale: float):
    s = (X @ q) / scale                                # (B, T)
    beta = _softmax_rows(s)                             # (B, T)
    a = np.einsum("bt,btd->bd", beta, X)               # (B, d)
    return a, beta, s


def mlp_forward(a: np.ndarray, Ws, bs):
    acts = [a]
    h = a
    for i, (W, b) in enumerate(zip(Ws, bs)):
        z = h @ W + b
        if i < len(Ws) - 1:
            h = np.maximum(z, 0.0)
        else:
            h = z
        acts.append(h)
    return acts[-1].ravel(), acts


def adam_step(params, grads, state, lr, t):
    for key in grads:
        m = state["m"].setdefault(key, np.zeros_like(grads[key]))
        v = state["v"].setdefault(key, np.zeros_like(grads[key]))
        m[:] = 0.9 * m + 0.1 * grads[key]
        v[:] = 0.999 * v + 0.001 * (grads[key] ** 2)
        mhat = m / (1 - 0.9 ** t)
        vhat = v / (1 - 0.999 ** t)
        params_ref[key][:] -= lr * mhat / (np.sqrt(vhat) + 1e-8)


def train_probe(Xtr: np.ndarray, ytr: np.ndarray, d: int, seed: int):
    """Full-batch Adam training of attention query + 5-layer MLP head."""
    rng = np.random.default_rng(seed)
    P = init_params(d, rng)
    q, Ws, bs = P["q"], P["Ws"], P["bs"]
    scale = float(np.sqrt(d))
    yv = ytr.astype(np.float64)
    B = Xtr.shape[0]

    mq = np.zeros_like(q); vq = np.zeros_like(q)
    mW = [np.zeros_like(W) for W in Ws]; vW = [np.zeros_like(W) for W in Ws]
    mb = [np.zeros_like(b) for b in bs]; vb = [np.zeros_like(b) for b in bs]

    def adam(p, g, m, v, t):
        m[:] = 0.9 * m + 0.1 * g
        v[:] = 0.999 * v + 0.001 * (g ** 2)
        mh = m / (1 - 0.9 ** t); vh = v / (1 - 0.999 ** t)
        p[:] -= LR * mh / (np.sqrt(vh) + 1e-8)

    for ep in range(1, EPOCHS + 1):
        a, beta, _ = attention_pool(Xtr, q, scale)
        logit, acts = mlp_forward(a, Ws, bs)
        p = _sigmoid(logit)
        dlogit = ((p - yv) / B)[:, None]               # (B,1)

        # Backprop MLP head.
        gWs = [None] * len(Ws); gbs = [None] * len(bs)
        grad = dlogit
        for i in reversed(range(len(Ws))):
            h_in = acts[i]
            gWs[i] = h_in.T @ grad + WD * Ws[i]
            gbs[i] = grad.sum(axis=0)
            grad = grad @ Ws[i].T
            if i > 0:
                grad = grad * (acts[i] > 0)
        da = grad                                       # (B, d) gradient wrt pooled a

        # Backprop attention pooling.  a = sum_t beta_t * X_t
        dbeta = np.einsum("bd,btd->bt", da, Xtr)        # (B, T)
        bdb = (beta * dbeta).sum(axis=1, keepdims=True)
        ds = beta * (dbeta - bdb)                       # softmax jacobian
        gq = np.einsum("bt,btd->d", ds, Xtr) / scale + WD * q

        adam(q, gq, mq, vq, ep)
        for i in range(len(Ws)):
            adam(Ws[i], gWs[i], mW[i], vW[i], ep)
            adam(bs[i], gbs[i], mb[i], vb[i], ep)

    return {"q": q, "Ws": Ws, "bs": bs, "scale": scale}


def predict(model, X: np.ndarray):
    a, beta, _ = attention_pool(X, model["q"], model["scale"])
    logit, _ = mlp_forward(a, model["Ws"], model["bs"])
    return logit, a


def run_for_K(grid: np.ndarray, y: np.ndarray, dom_l19: np.ndarray,
              dom_final: np.ndarray | None, K: int) -> dict:
    N, T, d = grid.shape
    # Standardize per-feature using all-data stats (cheap, leakage-light for direction;
    # OOF AUROC below refits stats per fold to stay honest).
    folds = stratified_kfold(y, N_FOLDS, SEED)
    oof = np.zeros(N, dtype=np.float64)
    for fi, test_idx in enumerate(folds):
        train_mask = np.ones(N, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = grid[train_mask], grid[test_idx]
        mu = Xtr.reshape(-1, d).mean(axis=0)
        sd = Xtr.reshape(-1, d).std(axis=0) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        model = train_probe(Xtr, y[train_mask], d, SEED + fi)
        logit, _ = predict(model, Xte)
        oof[test_idx] = logit
    auc = auroc(oof, y)

    # Direction test: fit one model on all (standardized) data, pooled DoM direction.
    mu = grid.reshape(-1, d).mean(axis=0)
    sd = grid.reshape(-1, d).std(axis=0) + 1e-6
    Xall = (grid - mu) / sd
    full_model = train_probe(Xall, y, d, SEED)
    _, a_all = predict(full_model, Xall)               # (N, d) pooled reps (standardized space)
    # Map pooled discriminative direction back to raw feature space (undo per-feature std).
    agg_dom_std = a_all[y].mean(axis=0) - a_all[~y].mean(axis=0)
    agg_dom_raw = agg_dom_std / sd                     # chain rule on the standardization
    q_raw = full_model["q"] / sd

    cos_aggdom_l19 = _cos(agg_dom_raw, dom_l19)
    cos_q_l19 = _cos(q_raw, dom_l19)
    cos_aggdom_final = _cos(agg_dom_raw, dom_final) if dom_final is not None else None

    f3_collapse = (
        auc > F2_BASELINE
        and dom_final is not None
        and cos_aggdom_l19 is not None and cos_aggdom_final is not None
        and abs(cos_aggdom_l19) > 0.4 and abs(cos_aggdom_final) > 0.4
    )
    return {
        "K": K,
        "n_tokens_aggregated": int(T),
        "auroc_oof": float(auc),
        "cos_aggdom_prefill_l19_dom": cos_aggdom_l19,
        "cos_query_prefill_l19_dom": cos_q_l19,
        "cos_aggdom_final_dom": cos_aggdom_final,
        "beats_f2_baseline": bool(auc > F2_BASELINE),
        "f3_orthogonality_collapse": bool(f3_collapse),
    }


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not TOKEN_CACHE.exists():
        print("MISSING_REGEN_INPUT", TOKEN_CACHE, file=sys.stderr); return 2

    base = np.load(CACHE)
    Xprefill = base["prefill"].astype(np.float64)
    y_base = base["correct"].astype(bool)
    assert Xprefill.shape == (500, 1536) and y_base.shape == (500,)
    dom_l19 = Xprefill[y_base].mean(axis=0) - Xprefill[~y_base].mean(axis=0)

    dom_final = None
    if FINAL_CACHE.exists():
        fblob = np.load(FINAL_CACHE)
        fkey = next((k for k in ("final", "final_hidden", "hidden") if k in fblob.files), None)
        if fkey is not None and "correct" in fblob.files:
            Xf = fblob[fkey].astype(np.float64)
            yf = fblob["correct"].astype(bool)
            if Xf.shape == (500, 1536):
                dom_final = Xf[yf].mean(axis=0) - Xf[~yf].mean(axis=0)

    loaded = load_token_grid()
    if loaded is None:
        return 2
    H, layers, positions, y = loaded
    if y.shape[0] != H.shape[0]:
        print("MISSING_REGEN_INPUT", TOKEN_CACHE, "(label/grid length mismatch)", file=sys.stderr)
        return 2

    per_k = []
    for K in K_VALUES:
        grid = select_grid(H, layers, K)
        if grid is None:
            per_k.append({"K": K, "auroc_oof": None, "skipped": "insufficient_positions_or_layers"})
            continue
        per_k.append(run_for_K(grid, y, dom_l19, dom_final, K))

    scored = [r for r in per_k if r.get("auroc_oof") is not None]
    best = max(scored, key=lambda r: r["auroc_oof"]) if scored else None

    fe724_auroc = None
    if FE724_JSON.exists():
        try:
            fe724 = json.loads(FE724_JSON.read_text())
            fe724_auroc = fe724.get("auroc_oof") or fe724.get("best_auroc")
        except Exception:
            fe724_auroc = None

    out = {
        "experiment": "P11-FE725",
        "description": "Attention probe over L18-L20 x last-K prompt positions + 5-layer MLP head",
        "n": int(y.shape[0]),
        "target_layers": list(TARGET_LAYERS),
        "k_values": list(K_VALUES),
        "mlp_layers": N_MLP_LAYERS,
        "mlp_hidden": HIDDEN,
        "f2_baseline_auroc": F2_BASELINE,
        "fe724_auroc": fe724_auroc,
        "final_dom_available": dom_final is not None,
        "per_k": per_k,
        "best_auroc_oof": float(best["auroc_oof"]) if best else None,
        "best_K": best["K"] if best else None,
        "best_beats_f2": bool(best["auroc_oof"] > F2_BASELINE) if best else None,
        "best_beats_fe724": (bool(best["auroc_oof"] > fe724_auroc)
                             if best and fe724_auroc is not None else None),
        "f3_orthogonality_collapse_any": bool(any(r.get("f3_orthogonality_collapse") for r in per_k)),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("best_auroc_oof", "best_K", "best_beats_f2",
                                          "f3_orthogonality_collapse_any")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())