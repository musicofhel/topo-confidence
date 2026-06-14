"""P11-FE34 — Rieck et al. neural persistence (NP) on Qwen-2.5 weight matrices.

Direct test of F-10 generalization. F-10 found activation-cloud persistent
homology sits at the Gaussian null; Rieck et al. (2019) report a clear
trained-vs-null separation for *weight-graph* PH. Here we compute per-layer,
per-matrix neural persistence on the trained weights of Qwen-2.5-1.5B and
Qwen-2.5-7B (MLP W_up/W_down, attention W_q/W_k/W_v/W_o) and compare each
against Gaussian-resample and uniform-resample nulls of identical shape.

Neural persistence (Rieck et al.): treat a weight matrix W (m x n) as a
bipartite graph on m+n vertices with edge weights |W| normalized to [0,1] by
the max abs entry. Vertices are born at filtration value 1; edges are added in
descending weight order; on each component merge the younger component dies at
the edge's normalized weight. NP_p = ( sum_i (1 - death_i)^p )^(1/p), and the
normalized NP divides by (n_vertices - 1)^(1/p) (the all-ones-lifetime max).

The headline question: does L19 — the layer whose prefill DoM direction
predicts correctness at AUROC 0.7731 (F-2) — carry a distinctive NP relative to
(a) its own null and (b) the across-layer NP distribution? If trained weights
separate from null while activation-cloud PH did not, F-10 must be sharpened to
"activation-cloud PH at null" rather than "PH at null."

Weights are loaded from local NPZ snapshots (no network / no transformers).
Each NPZ has one array per matrix, keyed like "L19_mlp_up", "L3_attn_q", etc.
If they are absent the script prints MISSING_REGEN_INPUT and exits 2; regenerate
with a one-off GPU box via DATA_MANIFEST (export W matrices to float32 NPZ).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
WEIGHTS_15B = ROOT / "pathway11_h100/neural_persistence/weights_qwen15b.npz"
WEIGHTS_7B = ROOT / "pathway11_h100/neural_persistence/weights_qwen7b.npz"
OUT_JSON = ROOT / "pathway11_h100/neural_persistence/results.json"

SEED = 9999
P_NORM = 2.0
N_NULL = int(os.environ.get("NP_N_NULL", "3"))   # resamples per null family
FOCUS_LAYER = 19                                  # the F-2 prefill-DoM layer


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Standard pairwise AUROC helper (kept for pattern consistency)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def neural_persistence(W: np.ndarray, p: float = P_NORM) -> tuple[float, float]:
    """Return (NP_p, normalized NP_p) for a weight matrix W (m x n).

    Bipartite weight-graph 0-dim persistence via descending-weight union-find,
    stopping once the graph is connected (only m+n-1 merges are finite points).
    """
    W = np.asarray(W)
    if W.ndim != 2:
        W = W.reshape(W.shape[0], -1)
    m, n = W.shape
    n_vert = m + n
    A = np.abs(W).astype(np.float32).ravel()
    amax = float(A.max()) if A.size else 0.0
    if amax <= 0.0 or n_vert < 2:
        return 0.0, 0.0
    A /= amax  # normalized edge weights in [0,1]

    order = np.argsort(A)[::-1]  # descending; only scanned until connected
    parent = np.arange(n_vert, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    ncomp = n_vert
    lifetimes: list[float] = []
    for e in order:
        if ncomp == 1:
            break
        u = e // n
        v = m + (e % n)
        ru = find(int(u))
        rv = find(int(v))
        if ru != rv:
            parent[ru] = rv
            ncomp -= 1
            lifetimes.append(1.0 - float(A[e]))

    lt = np.asarray(lifetimes, dtype=np.float64)
    np_val = float(np.power(np.sum(np.power(lt, p)), 1.0 / p)) if lt.size else 0.0
    norm = np_val / float((n_vert - 1) ** (1.0 / p))
    return np_val, norm


def null_np(W: np.ndarray, rng: np.random.Generator) -> dict:
    """Gaussian- and uniform-resample nulls matched to W's stats / range."""
    mu, sd = float(W.mean()), float(W.std())
    lo, hi = float(W.min()), float(W.max())
    gauss, unif = [], []
    for _ in range(N_NULL):
        g = rng.normal(mu, sd if sd > 0 else 1e-8, size=W.shape).astype(np.float32)
        u = rng.uniform(lo, hi, size=W.shape).astype(np.float32)
        gauss.append(neural_persistence(g)[1])
        unif.append(neural_persistence(u)[1])
    g = np.asarray(gauss)
    u = np.asarray(unif)
    return {
        "gaussian_mean": float(g.mean()), "gaussian_std": float(g.std()),
        "uniform_mean": float(u.mean()), "uniform_std": float(u.std()),
    }


def layer_of(key: str) -> int:
    m = re.search(r"L(\d+)", key)
    return int(m.group(1)) if m else -1


def process_model(npz_path: Path, rng: np.random.Generator) -> dict:
    blob = np.load(npz_path)
    per_matrix: dict[str, dict] = {}
    for key in blob.files:
        W = blob[key]
        trained_norm = neural_persistence(W)[1]
        nulls = null_np(W, rng)
        g_z = ((trained_norm - nulls["gaussian_mean"]) /
               (nulls["gaussian_std"] + 1e-12))
        u_z = ((trained_norm - nulls["uniform_mean"]) /
               (nulls["uniform_std"] + 1e-12))
        per_matrix[key] = {
            "shape": list(W.shape),
            "layer": layer_of(key),
            "trained_np_norm": trained_norm,
            **nulls,
            "z_vs_gaussian": float(g_z),
            "z_vs_uniform": float(u_z),
        }

    # Across-layer distinctiveness of the focus layer, per matrix family.
    families: dict[str, list[tuple[int, float]]] = {}
    for key, rec in per_matrix.items():
        fam = re.sub(r"^L\d+_", "", key)
        families.setdefault(fam, []).append((rec["layer"], rec["trained_np_norm"]))

    focus: dict[str, dict] = {}
    for fam, entries in families.items():
        vals = np.asarray([v for _, v in entries], dtype=np.float64)
        layers = np.asarray([l for l, _ in entries], dtype=np.int64)
        hit = layers == FOCUS_LAYER
        if not hit.any() or vals.size < 2:
            continue
        mu, sd = float(vals.mean()), float(vals.std())
        fval = float(vals[hit][0])
        focus[fam] = {
            "focus_layer_np_norm": fval,
            "across_layer_mean": mu,
            "across_layer_std": sd,
            "focus_z_across_layers": float((fval - mu) / (sd + 1e-12)),
            "n_layers": int(vals.size),
        }

    gz = np.asarray([r["z_vs_gaussian"] for r in per_matrix.values()])
    uz = np.asarray([r["z_vs_uniform"] for r in per_matrix.values()])
    return {
        "n_matrices": len(per_matrix),
        "mean_abs_z_vs_gaussian": float(np.mean(np.abs(gz))),
        "mean_abs_z_vs_uniform": float(np.mean(np.abs(uz))),
        "frac_separated_gaussian_2sigma": float(np.mean(np.abs(gz) > 2.0)),
        "per_matrix": per_matrix,
        "focus_layer_analysis": focus,
    }


def main() -> int:
    available = [(name, p) for name, p in
                 (("qwen2.5_1.5b", WEIGHTS_15B), ("qwen2.5_7b", WEIGHTS_7B))
                 if p.exists()]
    if not available:
        print("MISSING_REGEN_INPUT", WEIGHTS_15B, WEIGHTS_7B, file=sys.stderr)
        return 2

    rng = np.random.default_rng(SEED)
    out = {
        "experiment": "P11-FE34",
        "method": "Rieck neural persistence on weight-graph PH",
        "p_norm": P_NORM,
        "n_null_per_family": N_NULL,
        "focus_layer": FOCUS_LAYER,
        "models": {},
    }
    for name, path in available:
        try:
            out["models"][name] = process_model(path, rng)
        except Exception as exc:  # noqa: BLE001 — record, don't crash the daemon
            out["models"][name] = {"error": repr(exc)}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())