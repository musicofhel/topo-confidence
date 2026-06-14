"""P11-FE294 — CAST-style D-bucket condition-vector extraction vs F-7.

F-7 claims the D-bucket "signature" is collective, not per-problem. CAST
(Conditional Activation Steering) explicitly assumes per-prompt linear
separability and is the strongest supervised method for extracting such a
condition direction. This is a direct refutation test: run CAST's full
pipeline — per-layer grid search, PCA-PC1 + difference-of-means direction
candidates, oriented projection with an OOF threshold — to separate
D-bucket prompts (D⁺) from A-bucket prompts (D⁻).

Buckets are reconstructed from the K=8 self-consistency cache (stage-2
provenance): D-bucket = consistently-hard problems (0/8 generations correct),
A-bucket = consistently-easy problems (8/8 correct). All AUROCs are computed
out-of-fold so the test measures genuine per-prompt separability rather than
in-sample memorization of 1536-d activations.

Verdict:
  best OOF AUROC > 0.70           -> F7_REFUTED   (per-prompt, non-obvious layer)
  best OOF AUROC < 0.58 (chance)  -> F7_CORROBORATED (collective signature holds)
  otherwise                        -> INCONCLUSIVE
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
# Optional per-layer caches; CAST's layer grid search uses them if present,
# otherwise the search degenerates to the single cached L19 layer.
LAYER_GLOB = "pathway11_h100/prefill_inversion/cache/m15b_prefill_L*.npz"
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
OUT_JSON = ROOT / "pathway11_h100/cast_dbucket/results.json"

N_FOLDS = 5
N_PCS = 10
SEED = 9999

REFUTE_THRESH = 0.70
CHANCE_THRESH = 0.58


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


def load_buckets() -> np.ndarray | None:
    """Per-problem fraction-correct over K=8 generations; nan if file missing."""
    if not K8_DIR.exists():
        return None
    frac = np.full(500, np.nan, dtype=np.float64)
    found = 0
    for i in range(500):
        f = K8_DIR / f"problem_{i:03d}.npz"
        if not f.exists():
            continue
        c = np.load(f)["correct"].astype(bool)
        if c.size == 0:
            continue
        frac[i] = float(c.mean())
        found += 1
    if found == 0:
        return None
    return frac


def discover_layers() -> list[tuple[int, Path]]:
    """Return (layer_id, npz_path) pairs. Falls back to the single L19 cache."""
    layers: list[tuple[int, Path]] = []
    for p in sorted(ROOT.glob(LAYER_GLOB)):
        stem = p.stem  # m15b_prefill_L19
        tag = stem.rsplit("_L", 1)[-1]
        try:
            layers.append((int(tag), p))
        except ValueError:
            continue
    if not layers and CACHE.exists():
        layers.append((19, CACHE))
    return layers


def load_layer_matrix(path: Path) -> np.ndarray:
    blob = np.load(path)
    key = "prefill" if "prefill" in blob.files else blob.files[0]
    X = blob[key].astype(np.float64)
    if X.ndim != 2 or X.shape[0] != 500:
        raise ValueError(f"unexpected activation shape {X.shape} in {path}")
    return X


def cast_oof(X: np.ndarray, y: np.ndarray) -> dict:
    """Run CAST's direction grid (PCA components + difference-of-means) out-of-fold.

    Each config produces one OOF score vector (per-fold standardize + direction
    fit + train-oriented sign), scored once against y. Returns per-config and
    best OOF AUROC.
    """
    folds = stratified_kfold(y, N_FOLDS, SEED)
    n = len(y)
    configs = ["dom"] + [f"pca_pc{j + 1}" for j in range(N_PCS)]
    oof = {c: np.full(n, np.nan, dtype=np.float64) for c in configs}

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]

        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0) + 1e-8
        Ztr = (Xtr - mu) / sd
        Zte = (Xte - mu) / sd

        # difference-of-means direction (supervised, standardized space)
        dirs = {"dom": Ztr[ytr].mean(axis=0) - Ztr[~ytr].mean(axis=0)}
        # PCA components on the combined standardized training stimulus set
        Zc = Ztr - Ztr.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
            for j in range(min(N_PCS, Vt.shape[0])):
                dirs[f"pca_pc{j + 1}"] = Vt[j]
        except np.linalg.LinAlgError:
            pass

        for cfg, vec in dirs.items():
            norm = np.linalg.norm(vec)
            if norm < 1e-12:
                continue
            vec = vec / norm
            proj_tr = Ztr @ vec
            # orient so D⁺ (y=1) projects higher on the training fold
            if proj_tr[ytr].mean() < proj_tr[~ytr].mean():
                vec = -vec
            oof[cfg][test_idx] = Zte @ vec

    per_config = {}
    for cfg, scores in oof.items():
        if np.isnan(scores).any():
            continue
        per_config[cfg] = auroc(scores, y)
    if not per_config:
        return {"per_config": {}, "best_config": None, "best_auroc": float("nan")}
    best_cfg = max(per_config, key=lambda k: per_config[k])
    return {
        "per_config": per_config,
        "best_config": best_cfg,
        "best_auroc": per_config[best_cfg],
    }


def logreg_ceiling(X: np.ndarray, y: np.ndarray) -> float:
    """L2 logistic-regression OOF AUROC — strongest supervised reference."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return float("nan")
    folds = stratified_kfold(y, N_FOLDS, SEED)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(len(y), dtype=bool); train_mask[test_idx] = False
        sc = StandardScaler().fit(X[train_mask])
        clf = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X[train_mask]), y[train_mask])
        oof[test_idx] = clf.decision_function(sc.transform(X[test_idx]))
    return auroc(oof, y)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2

    frac = load_buckets()
    if frac is None:
        print("MISSING_REGEN_INPUT", K8_DIR, file=sys.stderr); return 2

    d_plus = frac == 0.0   # D-bucket: consistently hard (0/8 correct)
    d_minus = frac == 1.0  # A-bucket: consistently easy (8/8 correct)
    n_plus, n_minus = int(d_plus.sum()), int(d_minus.sum())
    if n_plus < N_FOLDS or n_minus < N_FOLDS:
        print(f"INSUFFICIENT_BUCKETS D+={n_plus} D-={n_minus}", file=sys.stderr)
        return 3

    sel = d_plus | d_minus
    y = d_plus[sel].astype(bool)  # True = D-bucket

    layers = discover_layers()
    if not layers:
        print("MISSING_REGEN_INPUT", "no layer caches", file=sys.stderr); return 2

    per_layer = {}
    overall_best = {"layer": None, "config": None, "auroc": float("-inf")}
    for layer_id, path in layers:
        try:
            X_full = load_layer_matrix(path)
        except (ValueError, KeyError) as e:
            print(f"SKIP_LAYER {layer_id}: {e}", file=sys.stderr)
            continue
        X = X_full[sel]
        res = cast_oof(X, y)
        per_layer[str(layer_id)] = res
        if res["best_config"] is not None and res["best_auroc"] > overall_best["auroc"]:
            overall_best = {
                "layer": layer_id,
                "config": res["best_config"],
                "auroc": res["best_auroc"],
            }

    if overall_best["layer"] is None:
        print("NO_VALID_CONFIG", file=sys.stderr); return 4

    # supervised ceiling on the best layer's activations
    best_path = dict(layers)[overall_best["layer"]]
    ceiling = logreg_ceiling(load_layer_matrix(best_path)[sel], y)

    best_auroc = overall_best["auroc"]
    if best_auroc > REFUTE_THRESH:
        verdict = "F7_REFUTED"
    elif best_auroc < CHANCE_THRESH:
        verdict = "F7_CORROBORATED"
    else:
        verdict = "INCONCLUSIVE"

    out = {
        "experiment": "P11-FE294",
        "description": "CAST-style D-bucket condition vector vs F-7 collective-signature framing",
        "bucket_definition": {
            "d_plus": "D-bucket = 0/8 correct (consistently hard)",
            "d_minus": "A-bucket = 8/8 correct (consistently easy)",
            "n_d_plus": n_plus,
            "n_d_minus": n_minus,
        },
        "n_layers_searched": len(per_layer),
        "layers_searched": sorted(int(k) for k in per_layer),
        "best_layer": overall_best["layer"],
        "best_config": overall_best["config"],
        "best_oof_auroc": float(best_auroc),
        "logreg_l2_oof_auroc_ceiling": float(ceiling),
        "per_layer": per_layer,
        "refute_threshold": REFUTE_THRESH,
        "chance_threshold": CHANCE_THRESH,
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"verdict={verdict} best_oof_auroc={best_auroc:.4f} "
          f"layer={overall_best['layer']} config={overall_best['config']} "
          f"ceiling={ceiling:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())