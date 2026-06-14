"""P11-FE692 — Multi-layer DoM ensemble vs single-layer L19 DoM.

Tests whether F-2's privileged-layer claim (L19-only prefill DoM AUROC 0.7731)
is a local optimum or a depth fact, by building an interleaved 5-layer DoM
probe over layers {4, 9, 14, 19, 24} of the cached Qwen2.5-1.5B prefill
activations. Replicates DistillLens's K=5 interleaved finding (Table 4) for
correctness probing instead of distillation R-L: DistillLens reports +3.06 R-L
single mid-layer vs +3.56 R-L 5-interleaved (~14% of the multi-layer benefit
is depth-spread aggregation). If the same super-additivity holds for DoM, the
ensemble OOF AUROC should clear the single-L19 0.7731 reference.

Per fold (5-fold stratified OOF): compute a class-mean DoM direction per layer
on the train split, project, z-score by train stats, then (a) fit a logistic
regression over the 5 per-layer scores (interleaved probe), and (b) average the
z-scored scores (mean-z ensemble). The L19-only projection is scored in the
same folds as the baseline so the comparison is leakage-free and apples-to-
apples.

Requires per-layer prefill activations; the documented main cache only carries
L19, so a multi-layer companion cache must be present. If the {4,9,14,24}
layers cannot be assembled, prints MISSING_REGEN_INPUT and returns 2.
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
OUT_JSON = ROOT / "pathway11_h100/multilayer_prefill_dom_ensemble_tada_lay/results.json"

# Candidate companion caches holding the non-L19 layers (first match wins).
MULTI_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_multilayer.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_all_layers.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_layers.npz",
    ROOT / "pathway11_h100/data/multilayer_prefill/m15b_layers.npz",
    ROOT / "pathway11_h100/multilayer_prefill_dom_ensemble_tada_lay/m15b_layers.npz",
]

LAYERS = [4, 9, 14, 19, 24]
L19 = 19
SINGLE_L19_REFERENCE = 0.7731  # F-2, 1024-tok, OOF 5-fold
N_FOLDS = 5
SEED = 9999


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


def _extract_layer(data, keys, layer: int):
    """Pull a (500, 1536) matrix for a given layer from a loaded npz."""
    patterns = [
        f"L{layer}", f"l{layer}", f"layer_{layer}", f"prefill_L{layer}",
        f"prefill_{layer}", f"h{layer}", f"hidden_{layer}",
    ]
    for p in patterns:
        if p in keys:
            a = np.asarray(data[p], dtype=np.float64)
            if a.ndim == 2 and a.shape == (500, 1536):
                return a
    # Stacked 3D arrays: (500, n_layers, 1536) or (n_layers, 500, 1536).
    for k in ("prefill", "prefill_layers", "layers", "hidden", "acts", "hidden_states"):
        if k in keys:
            a = np.asarray(data[k])
            if a.ndim == 3:
                if a.shape[0] == 500 and a.shape[2] == 1536 and a.shape[1] > layer:
                    return a[:, layer, :].astype(np.float64)
                if a.shape[1] == 500 and a.shape[2] == 1536 and a.shape[0] > layer:
                    return a[layer].astype(np.float64)
    return None


def load_layer_stack():
    """Return (layers_dict, y) with all LAYERS present, or (None, reason)."""
    if not CACHE.exists():
        return None, f"MISSING_REGEN_INPUT {CACHE}"
    blob = np.load(CACHE)
    y = blob["correct"].astype(bool)
    layers = {L19: blob["prefill"].astype(np.float64)}

    need = [L for L in LAYERS if L != L19]
    for cand in MULTI_CANDIDATES:
        if not cand.exists():
            continue
        data = np.load(cand)
        keys = set(data.files)
        got = {}
        for L in need:
            arr = _extract_layer(data, keys, L)
            if arr is None:
                got = None
                break
            got[L] = arr
        if got is not None:
            layers.update(got)
            if "correct" in keys:
                yc = data["correct"].astype(bool)
                if yc.shape == y.shape:
                    y = yc
            break

    missing = [L for L in LAYERS if L not in layers]
    if missing:
        return None, f"MISSING_REGEN_INPUT layers {missing} (no companion cache among {[str(c) for c in MULTI_CANDIDATES]})"
    for L in LAYERS:
        if layers[L].shape != (500, 1536):
            return None, f"BAD_SHAPE layer {L}: {layers[L].shape}"
    return (layers, y), None


def main() -> int:
    loaded, err = load_layer_stack()
    if loaded is None:
        print(err, file=sys.stderr)
        return 2
    layers, y = loaded

    from sklearn.linear_model import LogisticRegression

    folds = stratified_kfold(y, N_FOLDS, SEED)
    n = len(y)

    l19_oof = np.zeros(n, dtype=np.float64)
    ens_oof = np.zeros(n, dtype=np.float64)
    meanz_oof = np.zeros(n, dtype=np.float64)
    perlayer_oof = {L: np.zeros(n, dtype=np.float64) for L in LAYERS}

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        ytr = y[train_mask]

        feats_tr, feats_te = [], []
        for L in LAYERS:
            XL = layers[L]
            Xtr, Xte = XL[train_mask], XL[test_idx]
            d = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
            s_tr = Xtr @ d
            s_te = Xte @ d
            mu = s_tr.mean()
            sd = s_tr.std() + 1e-12
            feats_tr.append((s_tr - mu) / sd)
            feats_te.append((s_te - mu) / sd)
            perlayer_oof[L][test_idx] = s_te
            if L == L19:
                l19_oof[test_idx] = s_te

        Ftr = np.column_stack(feats_tr)
        Fte = np.column_stack(feats_te)

        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        clf.fit(Ftr, ytr)
        ens_oof[test_idx] = clf.decision_function(Fte)
        meanz_oof[test_idx] = Fte.mean(axis=1)

    auroc_l19 = auroc(l19_oof, y)
    auroc_ens = auroc(ens_oof, y)
    auroc_meanz = auroc(meanz_oof, y)
    perlayer_auroc = {f"L{L}": auroc(perlayer_oof[L], y) for L in LAYERS}

    out = {
        "experiment": "P11-FE692",
        "description": "Interleaved 5-layer DoM ensemble vs single-layer L19 DoM",
        "layers": LAYERS,
        "n": int(n),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "single_l19_reference_auroc": SINGLE_L19_REFERENCE,
        "auroc_l19_oof": auroc_l19,
        "auroc_ensemble_logreg_oof": auroc_ens,
        "auroc_ensemble_meanz_oof": auroc_meanz,
        "perlayer_oof_auroc": perlayer_auroc,
        "delta_ensemble_vs_l19_oof": float(auroc_ens - auroc_l19),
        "delta_ensemble_vs_reference": float(auroc_ens - SINGLE_L19_REFERENCE),
        "super_additive": bool(auroc_ens > auroc_l19 + 1e-6),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())