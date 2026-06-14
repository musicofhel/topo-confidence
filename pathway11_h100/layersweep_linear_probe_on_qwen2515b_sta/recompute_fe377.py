"""P11-FE377 — Layer-sweep linear probe for MATH-500 problem-type identity.

Tests whether L19's privileged status (F-2, a correctness direction) extends to
problem-*identity* information. We train a multinomial linear probe to predict
the MATH-500 problem type (algebra / geometry / number theory / counting &
probability / precalculus / intermediate algebra / prealgebra) from the
Qwen-2.5-1.5B Stage-2 residual stream at every available layer, and report the
per-layer out-of-fold classification accuracy + macro one-vs-rest AUROC.

Reading of the result, per Dhanraj & Eliasmith (2502.01657), who pick L17/32
(~0.53 depth) in LLaMA-3.1 8B for problem-input encoding (cf. our L19/28 ≈ 0.68):
  - If problem-type accuracy ALSO peaks at ~L19, then F-2's L19 is a generic
    "all problem info is legible here" layer.
  - If it peaks elsewhere, F-2's L19 is specifically a correctness-direction
    phenomenon. Either way F-2's framing sharpens.

Inputs (all optional-path-probed; MISSING_REGEN_INPUT if absent):
  - a per-layer Stage-2 activation stack (500, n_layers, 1536) OR per-layer NPZs
  - MATH-500 problem-type labels (JSON list of 500 strings, or an NPZ field)
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

# Candidate locations for the per-layer Stage-2 activation stack.
LAYER_STACK_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_all_layers.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_layers.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_layerwise.npz",
    ROOT / "pathway11_h100/stage2/m15b_all_layers.npz",
    ROOT / "pathway11_h100/data/stage2/m15b_all_layers.npz",
]
# Candidate glob for per-layer files (each a 2D (500, 1536) array).
LAYER_GLOB_DIRS = [
    ROOT / "pathway11_h100/prefill_inversion/cache",
    ROOT / "pathway11_h100/stage2",
    ROOT / "pathway11_h100/data/stage2",
]
LAYER_GLOB_PATTERNS = ["m15b_layer_*.npz", "layer_*.npz", "stage2_layer_*.npz"]

# Candidate locations for MATH-500 problem-type labels.
LABEL_JSON_CANDIDATES = [
    ROOT / "pathway11_h100/data/math500_types.json",
    ROOT / "pathway11_h100/data/math500_metadata.json",
    ROOT / "data/math500_types.json",
    ROOT / "data/math500_metadata.json",
    ROOT / "configs/math500_types.json",
]
LABEL_NPZ_FIELDS = ["problem_type", "subject", "type", "category"]

OUT_JSON = ROOT / "pathway11_h100/layer_sweep_problemtype/results.json"

SEED = 9999
N_FOLDS = 5
N_PCA = 100  # per-fold dim reduction to keep the 28-layer sweep CPU-cheap
C_REG = 0.1


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Binary AUROC; used per-class for one-vs-rest macro averaging."""
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def macro_ovr_auroc(proba: np.ndarray, y_int: np.ndarray, n_classes: int) -> float:
    vals = []
    for c in range(n_classes):
        a = auroc(proba[:, c], y_int == c)
        if not np.isnan(a):
            vals.append(a)
    return float(np.mean(vals)) if vals else float("nan")


def stratified_kfold(y_int: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """Stratified fold assignment over integer class labels."""
    rng = np.random.default_rng(seed)
    n = len(y_int)
    fold_of = np.empty(n, dtype=int)
    for c in np.unique(y_int):
        idx = np.flatnonzero(y_int == c)
        rng.shuffle(idx)
        for j, chunk in enumerate(np.array_split(idx, k)):
            fold_of[chunk] = j
    return [np.flatnonzero(fold_of == j) for j in range(k)]


def load_layer_stack() -> tuple[np.ndarray, list[int]] | None:
    """Return (X_all: (500, n_layers, 1536), layer_indices) or None if absent."""
    # (a) single stacked NPZ with a 3D array
    for path in LAYER_STACK_CANDIDATES:
        if path.exists():
            blob = np.load(path)
            for key in blob.files:
                arr = blob[key]
                if arr.ndim == 3 and arr.shape[0] == 500 and arr.shape[2] == 1536:
                    li_key = next((k for k in blob.files if "layer" in k.lower()
                                   and blob[k].ndim == 1), None)
                    layers = (list(blob[li_key].astype(int))
                              if li_key is not None else list(range(arr.shape[1])))
                    return arr.astype(np.float64), layers
                # transposed (500, 1536, n_layers)
                if arr.ndim == 3 and arr.shape[0] == 500 and arr.shape[1] == 1536:
                    return np.transpose(arr, (0, 2, 1)).astype(np.float64), \
                        list(range(arr.shape[2]))
    # (b) per-layer files
    for d in LAYER_GLOB_DIRS:
        if not d.exists():
            continue
        for pat in LAYER_GLOB_PATTERNS:
            files = sorted(d.glob(pat))
            if len(files) < 2:
                continue
            layer_arrs, layers = [], []
            for f in files:
                stem = f.stem
                digits = "".join(ch for ch in stem if ch.isdigit())
                blob = np.load(f)
                arr = next((blob[k] for k in blob.files
                            if blob[k].ndim == 2 and blob[k].shape == (500, 1536)), None)
                if arr is None:
                    layer_arrs = []
                    break
                layer_arrs.append(arr.astype(np.float64))
                layers.append(int(digits) if digits else len(layers))
            if len(layer_arrs) >= 2:
                order = np.argsort(layers)
                X_all = np.stack([layer_arrs[i] for i in order], axis=1)
                return X_all, [layers[i] for i in order]
    return None


def load_labels() -> np.ndarray | None:
    """Return (500,) array of problem-type label strings, or None if absent."""
    for path in LABEL_JSON_CANDIDATES:
        if path.exists():
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                for key in ("type", "subject", "problem_type", "types", "labels"):
                    if key in data:
                        data = data[key]
                        break
            if isinstance(data, list) and len(data) == 500:
                # list of strings, or list of dicts with a type field
                if isinstance(data[0], dict):
                    for key in ("type", "subject", "problem_type", "category"):
                        if key in data[0]:
                            return np.array([str(r[key]) for r in data])
                else:
                    return np.array([str(x) for x in data])
    # fallback: a field on the main cache
    if CACHE.exists():
        blob = np.load(CACHE, allow_pickle=True)
        for fld in LABEL_NPZ_FIELDS:
            if fld in blob.files and blob[fld].shape == (500,):
                return np.array([str(x) for x in blob[fld]])
    return None


def main() -> int:
    try:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:  # pragma: no cover
        print("MISSING_REGEN_INPUT sklearn", e, file=sys.stderr)
        return 2

    stack = load_layer_stack()
    if stack is None:
        print("MISSING_REGEN_INPUT per-layer Stage-2 activation stack "
              "(searched stacked NPZ + per-layer globs)", file=sys.stderr)
        return 2
    X_all, layer_indices = stack

    labels_str = load_labels()
    if labels_str is None:
        print("MISSING_REGEN_INPUT MATH-500 problem-type labels", file=sys.stderr)
        return 2

    classes = sorted(set(labels_str.tolist()))
    cls_to_int = {c: i for i, c in enumerate(classes)}
    y_int = np.array([cls_to_int[c] for c in labels_str], dtype=int)
    n_classes = len(classes)

    # Guard fold count against rare classes.
    min_count = min(int(np.sum(y_int == c)) for c in range(n_classes))
    k = max(2, min(N_FOLDS, min_count))
    folds = stratified_kfold(y_int, k, SEED)

    n_layers = X_all.shape[1]
    # Majority-class baseline accuracy.
    counts = np.array([np.sum(y_int == c) for c in range(n_classes)])
    baseline_acc = float(counts.max() / len(y_int))

    per_layer = []
    for li in range(n_layers):
        X = X_all[:, li, :]
        oof_pred = np.zeros(len(y_int), dtype=int)
        oof_proba = np.zeros((len(y_int), n_classes), dtype=np.float64)
        for test_idx in folds:
            train_mask = np.ones(len(y_int), dtype=bool)
            train_mask[test_idx] = False
            Xtr, Xte = X[train_mask], X[test_idx]
            ytr = y_int[train_mask]

            scaler = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
            n_comp = min(N_PCA, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
            pca = PCA(n_components=n_comp, random_state=SEED).fit(Xtr_s)
            Xtr_p, Xte_p = pca.transform(Xtr_s), pca.transform(Xte_s)

            clf = LogisticRegression(
                C=C_REG, max_iter=2000, multi_class="multinomial",
                solver="lbfgs", random_state=SEED,
            ).fit(Xtr_p, ytr)
            oof_pred[test_idx] = clf.classes_[np.argmax(clf.predict_proba(Xte_p), axis=1)]
            # map clf.classes_ ordering into full class space
            proba = np.zeros((len(test_idx), n_classes), dtype=np.float64)
            proba[:, clf.classes_] = clf.predict_proba(Xte_p)
            oof_proba[test_idx] = proba

        acc = float(np.mean(oof_pred == y_int))
        m_auroc = macro_ovr_auroc(oof_proba, y_int, n_classes)
        per_layer.append({
            "layer": int(layer_indices[li]),
            "accuracy": acc,
            "macro_ovr_auroc": m_auroc,
        })

    accs = np.array([p["accuracy"] for p in per_layer])
    peak_i = int(np.argmax(accs))
    peak_layer = int(per_layer[peak_i]["layer"])

    # Locate L19's entry for the head-to-head with F-2.
    l19 = next((p for p in per_layer if p["layer"] == 19), None)

    out = {
        "experiment": "P11-FE377",
        "model": "Qwen-2.5-1.5B",
        "task": "MATH-500 problem-type classification (layer sweep)",
        "n_examples": int(len(y_int)),
        "classes": classes,
        "class_counts": {c: int(counts[i]) for i, c in enumerate(classes)},
        "n_layers": int(n_layers),
        "layer_indices": [int(x) for x in layer_indices],
        "n_folds": int(k),
        "n_pca": int(N_PCA),
        "C_reg": C_REG,
        "majority_baseline_accuracy": baseline_acc,
        "per_layer": per_layer,
        "peak_layer": peak_layer,
        "peak_accuracy": float(accs[peak_i]),
        "l19_accuracy": (float(l19["accuracy"]) if l19 else None),
        "l19_macro_ovr_auroc": (float(l19["macro_ovr_auroc"]) if l19 else None),
        "l19_is_peak": (peak_layer == 19),
        "interpretation": (
            "problem-type peaks at L19 -> F-2 L19 is generic 'all problem info "
            "legible' layer" if peak_layer == 19 else
            "problem-type peaks off L19 -> F-2 L19 is correctness-specific"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())