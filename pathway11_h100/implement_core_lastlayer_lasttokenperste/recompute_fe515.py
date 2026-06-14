"""P11-FE515 — CoRE last-layer cycle detection vs F-2 (L19 prefill privilege) and F-9 (static DoM sufficiency).

Implements the CoRE (Chain-of-Representation) readout on cached Qwen-2.5-1.5B
MATH-500 1024-tok trajectories:

  1. Last-layer, last-token-per-step extraction is assumed already cached as a
     per-problem trajectory of step representations H_t (shape (T_steps, D)).
  2. Composite step signal  z_t = ||delta_t|| * (1 - cos(h_{t-1}, h_t)),
     where delta_t = h_t - h_{t-1}.
  3. Sliding-window Pearson cycle detection over z (window W=32, max period
     P_max=8, threshold rho*=0.7, min cycle run M=8). A step is in CYCLE state
     if the causal window ending at it shows lagged autocorrelation >= rho* for
     some period in 1..P_max.

Per-trial cycle features:
  - max_rho     : max lagged Pearson over all windows/periods
  - frac_cycle  : fraction of steps in CYCLE state
  - n_cycles    : # of distinct cycle runs (contiguous CYCLE runs of length >= M)

Tests:
  (a) AUROC of each feature + a 3-feature OOF logistic composite vs correctness.
  (b) Residual AUROC after OOF-partialling-out the L19 prefill DoM score.

R1 (CoRE last-layer beats/adds to L19 prefill) holds if a raw cycle AUROC is
materially > 0.5 and survives residualization. R2 (trajectory dynamics add to
static DoM, refuting F-9) holds if any residual AUROC stays > 0.5.

The per-step trajectory cache is NOT in the documented static-vector schema; if
it is absent this prints MISSING_REGEN_INPUT so the extraction can be regenerated.
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
TRAJ_DIR = ROOT / "pathway11_h100/data/core_trajectories"
OUT_JSON = ROOT / "pathway11_h100/core_cycle_detection/results.json"

N_PROBLEMS = 500
N_FOLDS = 5
SEED = 9999

# CoRE cycle-detection hyperparameters.
W = 32          # sliding-window length
P_MAX = 8       # max candidate period
RHO_STAR = 0.7  # lagged-autocorrelation threshold for CYCLE state
M = 8           # min contiguous CYCLE run to count as a distinct cycle

# Trajectory-array keys to probe, in preference order.
TRAJ_KEYS = ("step_hidden", "hidden", "last_layer", "H", "states")


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


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    da = float(np.sqrt((a * a).sum()))
    db = float(np.sqrt((b * b).sum()))
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float((a * b).sum() / (da * db))


def composite_z(H: np.ndarray) -> np.ndarray:
    """z_t = ||delta_t|| * (1 - cos(h_{t-1}, h_t)) over consecutive step reps."""
    if H.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    H = H.astype(np.float64)
    delta = np.diff(H, axis=0)
    norms = np.linalg.norm(delta, axis=1)
    h_prev, h_curr = H[:-1], H[1:]
    num = np.sum(h_prev * h_curr, axis=1)
    den = np.linalg.norm(h_prev, axis=1) * np.linalg.norm(h_curr, axis=1)
    cos = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return norms * (1.0 - cos)


def cycle_features(z: np.ndarray) -> tuple[float, float, float]:
    """(max_rho, frac_cycle, n_cycles) from sliding-window lagged autocorrelation."""
    L = len(z)
    if L < W + 1:
        return 0.0, 0.0, 0.0
    in_cycle = np.zeros(L, dtype=bool)
    global_max_rho = 0.0
    for start in range(0, L - W + 1):
        win = z[start:start + W]
        max_r = -1.0
        for P in range(1, P_MAX + 1):
            if W - P < 3:
                continue
            r = pearson(win[:-P], win[P:])
            if r > max_r:
                max_r = r
        if max_r > global_max_rho:
            global_max_rho = max_r
        # Causal assignment: the step at the window end carries the verdict.
        if max_r >= RHO_STAR:
            in_cycle[start + W - 1] = True

    frac_cycle = float(in_cycle.mean())

    # Count contiguous CYCLE runs of length >= M.
    n_cycles = 0
    run = 0
    for flag in in_cycle:
        if flag:
            run += 1
        else:
            if run >= M:
                n_cycles += 1
            run = 0
    if run >= M:
        n_cycles += 1

    return float(global_max_rho), frac_cycle, float(n_cycles)


def load_trajectory(path: Path) -> np.ndarray | None:
    try:
        blob = np.load(path)
    except Exception:
        return None
    for key in TRAJ_KEYS:
        if key in blob.files:
            arr = np.asarray(blob[key])
            if arr.ndim == 2 and arr.shape[0] >= 1:
                return arr
    # Fall back to the sole 2-D array if present.
    for key in blob.files:
        arr = np.asarray(blob[key])
        if arr.ndim == 2 and arr.shape[0] >= 1:
            return arr
    return None


def oof_residualize(score: np.ndarray, dom: np.ndarray, y: np.ndarray,
                    folds: list[np.ndarray]) -> np.ndarray:
    """Per-fold linear partialling-out of dom from score (OOF)."""
    res = np.zeros_like(score, dtype=np.float64)
    n = len(score)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        A = np.column_stack([dom[train_mask], np.ones(train_mask.sum())])
        c, _, _, _ = np.linalg.lstsq(A, score[train_mask], rcond=None)
        res[test_idx] = score[test_idx] - (dom[test_idx] * c[0] + c[1])
    return res


def oof_logistic_composite(feats: np.ndarray, y: np.ndarray,
                           folds: list[np.ndarray]) -> np.ndarray:
    """3-feature OOF logistic-regression score (standardized per train fold)."""
    from sklearn.linear_model import LogisticRegression

    n = len(y)
    oof = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = feats[train_mask], feats[test_idx]
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0); sd[sd < 1e-12] = 1.0
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit((Xtr - mu) / sd, y[train_mask])
        oof[test_idx] = clf.predict_proba((Xte - mu) / sd)[:, 1]
    return oof


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr); return 2
    if not TRAJ_DIR.exists():
        print("MISSING_REGEN_INPUT", TRAJ_DIR, file=sys.stderr); return 2

    cache = np.load(CACHE)
    correct_all = cache["correct"].astype(bool)
    dom_all = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert correct_all.shape == (N_PROBLEMS,) and dom_all.shape == (N_PROBLEMS,)

    present_idx: list[int] = []
    max_rho_l: list[float] = []
    frac_cycle_l: list[float] = []
    n_cycles_l: list[float] = []
    for i in range(N_PROBLEMS):
        path = TRAJ_DIR / f"problem_{i:03d}.npz"
        if not path.exists():
            continue
        H = load_trajectory(path)
        if H is None:
            continue
        mr, fc, nc = cycle_features(composite_z(H))
        present_idx.append(i)
        max_rho_l.append(mr)
        frac_cycle_l.append(fc)
        n_cycles_l.append(nc)

    if len(present_idx) == 0:
        print("MISSING_REGEN_INPUT", TRAJ_DIR, "no readable trajectory files",
              file=sys.stderr)
        return 2

    idx = np.asarray(present_idx, dtype=int)
    y = correct_all[idx]
    dom = dom_all[idx]
    max_rho = np.asarray(max_rho_l, dtype=np.float64)
    frac_cycle = np.asarray(frac_cycle_l, dtype=np.float64)
    n_cycles = np.asarray(n_cycles_l, dtype=np.float64)
    feats = np.column_stack([max_rho, frac_cycle, n_cycles])

    out: dict = {
        "experiment": "P11-FE515",
        "n_trials": int(len(idx)),
        "n_missing_trajectories": int(N_PROBLEMS - len(idx)),
        "params": {"W": W, "P_max": P_MAX, "rho_star": RHO_STAR, "M": M},
        "label_prevalence": float(y.mean()),
    }

    if y.sum() == 0 or (~y).sum() == 0 or len(idx) < N_FOLDS * 2:
        out["status"] = "DEGENERATE_LABELS_OR_TOO_FEW_TRIALS"
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(out, indent=2))
        return 0

    folds = stratified_kfold(y, N_FOLDS, SEED)

    # (a) Raw single-feature AUROCs (sign-agnostic: report both orientations).
    feat_map = {"max_rho": max_rho, "frac_cycle": frac_cycle, "n_cycles": n_cycles}
    out["auroc_raw"] = {}
    for name, vec in feat_map.items():
        a = auroc(vec, y)
        out["auroc_raw"][name] = float(a)
        out["auroc_raw"][name + "_neg"] = float(auroc(-vec, y))

    # 3-feature OOF logistic composite.
    oof_comp = oof_logistic_composite(feats, y, folds)
    out["auroc_composite_oof"] = float(auroc(oof_comp, y))

    # Baseline: DoM alone on this subset, for reference.
    out["auroc_dom_baseline"] = float(auroc(dom, y))

    # (b) Residual AUROC after OOF-partialling-out the L19 prefill DoM.
    out["auroc_residual_after_dom"] = {}
    for name, vec in feat_map.items():
        res = oof_residualize(vec, dom, y, folds)
        out["auroc_residual_after_dom"][name] = float(auroc(res, y))
        out["auroc_residual_after_dom"][name + "_neg"] = float(auroc(-res, y))
    res_comp = oof_residualize(oof_comp, dom, y, folds)
    out["auroc_residual_after_dom"]["composite"] = float(auroc(res_comp, y))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())