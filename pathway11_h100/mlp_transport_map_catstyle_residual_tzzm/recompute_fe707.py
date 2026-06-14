"""FE707 — CAT-style non-linear MLP transport map vs F-2 linear probe.

Trains a residual transport map T(z) = z + MLP(z) on cached L19 prefill
activations of MATH-500 (Qwen-2.5-1.5B), with the MLP's final layer zero-init
(so T = identity at init, reproducing the linear probe) and an RMSNorm+GELU
hidden block. A linear logistic head reads out T(z). Motivation (CAT,
variance-mismatch / XOR synthetics): a linear ActAdd mass-mean direction can be
~zero while a non-linear transport recovers the true class mapping; if the L19
prefill manifold has the same shape mismatch, F-2's linear AUROC (0.7731)
underestimates the available signal.

Compares 5-fold OOF AUROC of:
  - the MLP transport readout,
  - the same head with the MLP frozen at zero (pure linear logistic), and
  - the mass-mean DoM direction,
all against F-2's 0.7731 logistic-probe baseline. Pure numpy / CPU.
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
OUT_JSON = ROOT / "pathway11_h100/mlp_transport/results.json"

N_FOLDS = 5
SEED = 9999
F2_BASELINE = 0.7731

# MLP / optimisation hyperparameters
HIDDEN = 64
EPOCHS = 300
LR = 2e-3
WEIGHT_DECAY = 1e-3
RMS_EPS = 1e-6
ADAM_B1 = 0.9
ADAM_B2 = 0.999
ADAM_EPS = 1e-8
GELU_C = float(np.sqrt(2.0 / np.pi))
GELU_A = 0.044715


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(GELU_C * (x + GELU_A * x ** 3)))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    u = GELU_C * (x + GELU_A * x ** 3)
    t = np.tanh(u)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * GELU_C * (1.0 + 3.0 * GELU_A * x ** 2)


def _rmsnorm(X: np.ndarray) -> np.ndarray:
    s = np.sqrt((X ** 2).mean(axis=1, keepdims=True) + RMS_EPS)
    return X / s


def train_transport(Xtr, ytr, Xte, *, use_mlp: bool, seed: int):
    """Full-batch Adam on a residual transport map; returns test scores (logits)."""
    rng = np.random.default_rng(seed)
    n, d = Xtr.shape
    H = HIDDEN

    # Parameters. W2 zero-init => T == identity at init (linear probe equivalence).
    W1 = (rng.standard_normal((d, H)) * np.sqrt(2.0 / d)) if use_mlp else np.zeros((d, H))
    b1 = np.zeros(H)
    W2 = np.zeros((H, d))
    w = np.zeros(d)
    b_out = 0.0

    params = {"W1": W1, "b1": b1, "W2": W2, "w": w, "b_out": np.array(b_out)}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}

    yf = ytr.astype(np.float64)
    r = _rmsnorm(Xtr)  # input is fixed -> precompute

    for ep in range(1, EPOCHS + 1):
        # ---- forward ----
        if use_mlp:
            h_pre = r @ params["W1"] + params["b1"]
            h = _gelu(h_pre)
            delta = h @ params["W2"]
        else:
            h_pre = h = delta = None
            delta = np.zeros_like(Xtr)
        T = Xtr + delta
        logit = T @ params["w"] + params["b_out"]
        p = _sigmoid(logit)

        # ---- backward (BCE mean loss) ----
        dlogit = (p - yf) / n
        dw = T.T @ dlogit + WEIGHT_DECAY * params["w"]
        db_out = dlogit.sum()
        dT = np.outer(dlogit, params["w"])

        grads = {"w": dw, "b_out": np.array(db_out)}
        if use_mlp:
            ddelta = dT
            dW2 = h.T @ ddelta + WEIGHT_DECAY * params["W2"]
            dh = ddelta @ params["W2"].T
            dh_pre = dh * _gelu_grad(h_pre)
            dW1 = r.T @ dh_pre + WEIGHT_DECAY * params["W1"]
            db1 = dh_pre.sum(axis=0)
            grads.update({"W1": dW1, "b1": db1, "W2": dW2})
        else:
            grads.update({
                "W1": np.zeros_like(params["W1"]),
                "b1": np.zeros_like(params["b1"]),
                "W2": np.zeros_like(params["W2"]),
            })

        # ---- Adam step ----
        for k, g in grads.items():
            m[k] = ADAM_B1 * m[k] + (1 - ADAM_B1) * g
            v[k] = ADAM_B2 * v[k] + (1 - ADAM_B2) * (g ** 2)
            mhat = m[k] / (1 - ADAM_B1 ** ep)
            vhat = v[k] / (1 - ADAM_B2 ** ep)
            params[k] = params[k] - LR * mhat / (np.sqrt(vhat) + ADAM_EPS)

    # ---- test forward ----
    rte = _rmsnorm(Xte)
    if use_mlp:
        delta_te = _gelu(rte @ params["W1"] + params["b1"]) @ params["W2"]
    else:
        delta_te = np.zeros_like(Xte)
    Tte = Xte + delta_te
    return Tte @ params["w"] + float(params["b_out"])


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    folds = stratified_kfold(y, N_FOLDS, SEED)
    n = len(y)
    oof_mlp = np.zeros(n, dtype=np.float64)
    oof_lin = np.zeros(n, dtype=np.float64)
    oof_dom = np.zeros(n, dtype=np.float64)

    for fi, test_idx in enumerate(folds):
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]

        # Per-fold standardisation fit on train.
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0) + 1e-8
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd

        oof_mlp[test_idx] = train_transport(Xtr_s, ytr, Xte_s, use_mlp=True, seed=SEED + fi)
        oof_lin[test_idx] = train_transport(Xtr_s, ytr, Xte_s, use_mlp=False, seed=SEED + fi)

        # Mass-mean DoM baseline on standardised features.
        d_dom = Xtr_s[ytr].mean(axis=0) - Xtr_s[~ytr].mean(axis=0)
        oof_dom[test_idx] = Xte_s @ d_dom

    auroc_mlp = auroc(oof_mlp, y)
    auroc_lin = auroc(oof_lin, y)
    auroc_dom = auroc(oof_dom, y)

    out = {
        "experiment": "P11-FE707",
        "description": "CAT-style residual MLP transport map T(z)=z+MLP(z) vs F-2 linear probe",
        "n": int(n),
        "n_folds": N_FOLDS,
        "hidden": HIDDEN,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "auroc_mlp_transport_oof": float(auroc_mlp),
        "auroc_linear_logistic_oof": float(auroc_lin),
        "auroc_dom_massmean_oof": float(auroc_dom),
        "f2_baseline_auroc": F2_BASELINE,
        "delta_mlp_vs_f2": float(auroc_mlp - F2_BASELINE),
        "delta_mlp_vs_linear": float(auroc_mlp - auroc_lin),
        "mlp_beats_f2": bool(auroc_mlp > F2_BASELINE),
        "mlp_beats_linear": bool(auroc_mlp > auroc_lin),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())