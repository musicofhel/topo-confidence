"""P11-FE307 — Damani-style MLP-on-prefill difficulty probe vs F-2 linear logistic.

Trains a 2-layer MLP (one hidden ReLU layer, sigmoid output) on cached L19
prefill activations of Qwen-2.5-1.5B over MATH-500, targeting the Bernoulli
success rate lambda_i estimated from K=8 self-consistency generations
(Damani et al. Eq. 7 cross-entropy on soft targets). The OOF MLP predictions
are scored as AUROC against K=1 correctness and compared head-to-head with
F-2's linear logistic probe (reference 0.7731). If the MLP lifts AUROC by
>= 0.02 over the linear probe, F-2 is probe-suboptimal.

A secondary layer sweep (all 28 layers, top-layer last-token activations) tests
whether F-2's L19 choice is layer-precise or layer-arbitrary. The per-layer
activation cache is optional; if it is absent the sweep is reported as
unavailable rather than failing the run.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
PERLAYER = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_perlayer.npz"
OUT_JSON = ROOT / "pathway11_h100/damani_mlp_probe/results.json"

SEED = 9999
N_FOLDS = 5
REF_F2_LINEAR = 0.7731
HIDDEN = 64
EPOCHS = 400
LR = 0.1
L2 = 1e-3


def auroc(scores, labels):
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def train_mlp(X, y_soft, hidden=HIDDEN, epochs=EPOCHS, lr=LR, l2=L2, seed=SEED):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W1 = rng.standard_normal((d, hidden)) * np.sqrt(2.0 / d)
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden, 1)) * np.sqrt(2.0 / hidden)
    b2 = 0.0
    for _ in range(epochs):
        Z1 = X @ W1 + b1
        H = np.maximum(Z1, 0.0)
        logits = (H @ W2).ravel() + b2
        p = _sigmoid(logits)
        dlogits = (p - y_soft) / n
        gW2 = H.T @ dlogits[:, None] + l2 * W2
        gb2 = dlogits.sum()
        dH = dlogits[:, None] @ W2.T
        dZ1 = dH * (Z1 > 0)
        gW1 = X.T @ dZ1 + l2 * W1
        gb1 = dZ1.sum(axis=0)
        W1 -= lr * gW1
        b1 -= lr * gb1
        W2 -= lr * gW2
        b2 -= lr * gb2
    return W1, b1, W2, b2


def predict_mlp(X, params):
    W1, b1, W2, b2 = params
    H = np.maximum(X @ W1 + b1, 0.0)
    return _sigmoid((H @ W2).ravel() + b2)


def load_k8_soft(n=500):
    if not K8_DIR.exists():
        return None
    lam = np.full(n, np.nan, dtype=np.float64)
    missing = 0
    for i in range(n):
        f = K8_DIR / f"problem_{i:03d}.npz"
        if not f.exists():
            missing += 1
            continue
        c = np.load(f)["correct"].astype(bool)
        lam[i] = float(c.mean())
    if missing > n // 4:
        return None
    # backfill any stragglers with the global mean so the MLP target is defined
    if np.isnan(lam).any():
        lam[np.isnan(lam)] = np.nanmean(lam)
    return lam


def oof_mlp(X, y_soft, correct):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(correct), dtype=np.float64)
    for tr, te in skf.split(X, correct):
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0) + 1e-8
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd
        params = train_mlp(Xtr, y_soft[tr])
        oof[te] = predict_mlp(Xte, params)
    return oof


def oof_linear(X, correct):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(correct), dtype=np.float64)
    for tr, te in skf.split(X, correct):
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0) + 1e-8
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(Xtr, correct[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
    return oof


def layer_sweep(correct):
    """Optional: AUROC of a linear logistic probe on each layer's last-token state."""
    if not PERLAYER.exists():
        return {"available": False, "reason": f"missing {PERLAYER.name}"}
    blob = np.load(PERLAYER)
    # expect an array shaped (n_layers, n, d); accept common key names
    key = next((k for k in ("hidden", "states", "perlayer", "acts") if k in blob), None)
    if key is None:
        return {"available": False, "reason": "no recognized array key"}
    H = blob[key]
    if H.ndim != 3 or H.shape[1] != len(correct):
        return {"available": False, "reason": f"unexpected shape {H.shape}"}
    n_layers = H.shape[0]
    per_layer = []
    for L in range(n_layers):
        oof = oof_linear(H[L].astype(np.float64), correct)
        per_layer.append(float(auroc(oof, correct)))
    best = int(np.argmax(per_layer))
    l19 = float(per_layer[19]) if n_layers > 19 else float("nan")
    return {
        "available": True,
        "n_layers": n_layers,
        "auroc_per_layer": per_layer,
        "best_layer": best,
        "best_auroc": float(per_layer[best]),
        "l19_auroc": l19,
        "l19_minus_best": l19 - float(per_layer[best]) if n_layers > 19 else float("nan"),
        "layer_choice_precise": bool(n_layers > 19 and (float(per_layer[best]) - l19) >= 0.02),
    }


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    correct = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and correct.shape == (500,)

    lam = load_k8_soft(n=len(correct))
    if lam is None:
        print("MISSING_REGEN_INPUT", K8_DIR, file=sys.stderr)
        return 2

    oof_mlp_scores = oof_mlp(X, lam, correct)
    oof_lin_scores = oof_linear(X, correct)

    auroc_mlp = float(auroc(oof_mlp_scores, correct))
    auroc_lin = float(auroc(oof_lin_scores, correct))
    lift_vs_ref = auroc_mlp - REF_F2_LINEAR
    lift_vs_lin = auroc_mlp - auroc_lin

    out = {
        "experiment": "P11-FE307",
        "n": int(len(correct)),
        "lambda_mean": float(lam.mean()),
        "lambda_std": float(lam.std()),
        "mlp_hidden": HIDDEN,
        "mlp_epochs": EPOCHS,
        "auroc_mlp_soft_oof": auroc_mlp,
        "auroc_linear_logistic_oof": auroc_lin,
        "auroc_reference_f2_linear": REF_F2_LINEAR,
        "mlp_lift_over_local_linear": lift_vs_lin,
        "mlp_lift_over_reference": lift_vs_ref,
        "f2_probe_suboptimal": bool(lift_vs_lin >= 0.02),
        "layer_sweep": layer_sweep(correct),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "layer_sweep"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())