"""P11-FE475 — Multi-layer weighted MLP probe for correctness.

Tests whether concatenating prefill hidden states across layers L19..L27 and
training an MLP(hidden=1024, ReLU, MSE-target=correctness) beats F-2's single-
layer-L19 DoM ceiling of 0.7731 (5-fold OOF AUROC).

Motivation: WAS's layer-selection ablation (Table 6) shows multi-layer steering
beats late-only by ~12 points. For *correctness prediction* we have only ever
probed L19 alone. If the multi-layer MLP gains >= 0.02 AUROC over 0.7731, the
"single direction at L19" framing in F-2 / F-9 must be relaxed.

Inputs:
  - Labels (correct) + L19 prefill from the canonical m15b_prefill.npz.
  - Multi-layer prefill activations (L19..L27) from a Stage-2 multilayer NPZ.
    Supported layouts:
      (a) per-layer keys: prefill_L19 ... prefill_L27, each (500, 1536)
      (b) stacked key 'prefill_layers' (500, n_layers, 1536) + 'layers' index
    If no multilayer cache is found, prints MISSING_REGEN_INPUT and returns 2.

For reference we also train the same MLP on L19-alone so the multi-layer gain is
measured against a matched architecture, not only the linear DoM number.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
MULTILAYER_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_multilayer.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_layers.npz",
    ROOT / "pathway11_h100/data/m15b_prefill_multilayer.npz",
]
OUT_JSON = ROOT / "pathway11_h100/multilayer_mlp_probe/results.json"

LAYERS = list(range(19, 28))  # L19 .. L27 inclusive (9 layers)
F2_DOM_AUROC = 0.7731         # F-2 single-direction-at-L19 ceiling (1024tok)
GAIN_THRESHOLD = 0.02
HIDDEN = 1024
N_FOLDS = 5
SEED = 9999


def auroc(scores, labels):
    labels = labels.astype(bool)
    pos = scores[labels]; neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def load_multilayer(path: Path, layers):
    """Return (X (500, 1536*len(layers)), found_layers) or None if unparseable."""
    blob = np.load(path)
    keys = set(blob.keys())

    # Layout (a): per-layer keys.
    cols = []
    for L in layers:
        hit = None
        for cand in (f"prefill_L{L}", f"prefill_l{L}", f"L{L}", f"layer_{L}", f"h_L{L}"):
            if cand in keys:
                hit = blob[cand]; break
        if hit is None:
            cols = None; break
        cols.append(np.asarray(hit, dtype=np.float64))
    if cols is not None:
        return np.concatenate(cols, axis=1), list(layers)

    # Layout (b): stacked array + layer index.
    stacked = None
    for cand in ("prefill_layers", "prefill_all", "hidden_layers"):
        if cand in keys:
            stacked = np.asarray(blob[cand], dtype=np.float64); break
    if stacked is None or stacked.ndim != 3:
        return None
    idx_key = next((k for k in ("layers", "layer_indices", "layer_idx") if k in keys), None)
    if idx_key is None:
        return None
    layer_ids = list(np.asarray(blob[idx_key]).ravel().astype(int))
    sel = []
    for L in layers:
        if L not in layer_ids:
            return None
        sel.append(layer_ids.index(L))
    cols = [stacked[:, j, :] for j in sel]
    return np.concatenate(cols, axis=1), list(layers)


def make_mlp():
    return MLPRegressor(
        hidden_layer_sizes=(HIDDEN,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=SEED,
    )


def oof_mlp_auroc(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        model = make_mlp()
        model.fit(Xtr, y[train_idx].astype(np.float64))
        oof[test_idx] = model.predict(Xte)
    return float(auroc(oof, y)), oof


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2

    base = np.load(CACHE)
    X19 = base["prefill"].astype(np.float64)
    y = base["correct"].astype(bool)
    assert X19.shape == (500, 1536) and y.shape == (500,)

    multilayer_path = next((p for p in MULTILAYER_CANDIDATES if p.exists()), None)
    if multilayer_path is None:
        print("MISSING_REGEN_INPUT", "multilayer prefill NPZ (L19..L27);",
              "candidates:", *[str(p) for p in MULTILAYER_CANDIDATES], file=sys.stderr)
        return 2

    loaded = load_multilayer(multilayer_path, LAYERS)
    if loaded is None:
        print("MISSING_REGEN_INPUT", "could not parse layers L19..L27 from",
              multilayer_path, file=sys.stderr)
        return 2
    Xmulti, found_layers = loaded
    if Xmulti.shape != (500, 1536 * len(LAYERS)):
        print("MISSING_REGEN_INPUT", "unexpected multilayer shape", Xmulti.shape,
              file=sys.stderr)
        return 2

    auroc_multi, _ = oof_mlp_auroc(Xmulti, y)
    auroc_l19_mlp, _ = oof_mlp_auroc(X19, y)
    auroc_l19_dom_oof = _dom_oof_auroc(X19, y)

    gain_vs_f2 = auroc_multi - F2_DOM_AUROC
    gain_vs_l19_mlp = auroc_multi - auroc_l19_mlp

    out = {
        "experiment": "P11-FE475",
        "description": "Multi-layer (L19..L27) MLP(hidden=1024,ReLU) probe vs F-2 L19 ceiling",
        "multilayer_source": str(multilayer_path),
        "layers": found_layers,
        "n_features_multilayer": int(Xmulti.shape[1]),
        "mlp_hidden": HIDDEN,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "auroc_multilayer_mlp_oof": auroc_multi,
        "auroc_l19_mlp_oof": auroc_l19_mlp,
        "auroc_l19_dom_oof": auroc_l19_dom_oof,
        "f2_dom_reference": F2_DOM_AUROC,
        "gain_vs_f2_dom": float(gain_vs_f2),
        "gain_vs_l19_mlp": float(gain_vs_l19_mlp),
        "gain_threshold": GAIN_THRESHOLD,
        "single_direction_framing_challenged": bool(gain_vs_f2 >= GAIN_THRESHOLD),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


def _dom_oof_auroc(X, y):
    """Matched 5-fold OOF AUROC for the linear L19 DoM, for a fair comparison."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        ytr = y[train_idx]
        d = X[train_idx][ytr].mean(axis=0) - X[train_idx][~ytr].mean(axis=0)
        oof[test_idx] = X[test_idx] @ d
    return float(auroc(oof, y))


if __name__ == "__main__":
    raise SystemExit(main())