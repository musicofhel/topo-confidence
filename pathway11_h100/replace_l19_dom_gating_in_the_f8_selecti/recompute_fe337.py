"""P11-FE337 — CoE-C-gated selective prediction vs supervised L19 DoM (F-8).

F-8's headline (71.6% answered accuracy at 50% coverage, K=2.5 avg) currently
rests on a *supervised* L19 prefill DoM direction. This experiment swaps the
gate for a closed-form, label-free Chain-of-Embedding "CoE-C" score computed on
the *same* L19 prefill hidden states, then re-runs the identical selective-
prediction protocol (coverage 0.5, adaptive K averaging 2.5 over the K=8 self-
consistency cache). If CoE-C-gated acc@coverage-0.5 >= 71.6%, F-8 is
re-attributable to label-free trajectory geometry rather than a supervised
probe.

CoE-C in the source paper is a magnitude+angle statistic over the per-layer
hidden-state trajectory. The committed cache only exposes the L19 prefill layer
(no multi-layer trajectory NPZ exists in the schema), so we compute a faithful
single-layer closed-form analogue: the population L2 magnitude combined with the
angle of each state to the population mean direction. Both pieces are label-free
(only population statistics over the 500 prefill states are used). The score's
global orientation is fixed by AUROC sign (the paper's CoE-C carries a defined
orientation; here we resolve it empirically and report both for transparency).
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
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
F8_RESULTS = ROOT / "pathway11_h100/prefill_gated_compute/results.json"
OUT_JSON = ROOT / "pathway11_h100/coe_c_selective/results.json"

COVERAGE = 0.5
K_LO = 2
K_HI = 3          # top-half answered -> K_LO, bottom-half answered -> K_HI; avg = 2.5
F8_TARGET = 0.716


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-12)


def load_k8_correct(n: int) -> np.ndarray | None:
    """Stack per-problem (8,) correctness booleans into (n, 8)."""
    rows = []
    for i in range(n):
        f = K8_DIR / f"problem_{i:03d}.npz"
        if not f.exists():
            return None
        c = np.load(f)["correct"].astype(bool)
        if c.shape[0] < 8:
            return None
        rows.append(c[:8])
    return np.vstack(rows)


def majority_correct(c: np.ndarray, k: int) -> bool:
    """Majority-vote correctness over the first k generations.

    Only per-sample correctness is cached (not raw answers), so we use the
    standard no-collusion approximation: a sample of k is correct iff a strict
    majority of the k draws are correct; ties fall back to the greedy (first)
    draw.
    """
    s = int(c[:k].sum())
    thr = k / 2.0
    if s > thr:
        return True
    if s < thr:
        return False
    return bool(c[0])


def selective_acc(score: np.ndarray, k8: np.ndarray, coverage: float,
                  k_lo: int, k_hi: int) -> dict:
    """Answer the top-`coverage` fraction by confidence; adaptive K within."""
    n = len(score)
    order = np.argsort(-score, kind="stable")          # descending confidence
    n_ans = int(round(coverage * n))
    answered = order[:n_ans]
    half = n_ans // 2
    ks = np.empty(n_ans, dtype=int)
    ks[:half] = k_lo          # most confident answered -> least compute
    ks[half:] = k_hi          # least confident answered -> most compute
    n_correct = sum(majority_correct(k8[idx], int(k)) for idx, k in zip(answered, ks))
    return {
        "coverage": float(n_ans / n),
        "n_answered": int(n_ans),
        "n_correct": int(n_correct),
        "acc_answered": float(n_correct / n_ans) if n_ans else float("nan"),
        "k_avg": float(ks.mean()) if n_ans else float("nan"),
    }


def orient(score: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float, bool]:
    a = auroc(score, labels)
    if a < 0.5:
        return -score, 1.0 - a, True
    return score, a, False


def main() -> int:
    for p in (CACHE, DOM_NPZ):
        if not p.exists():
            print("MISSING_REGEN_INPUT", p, file=sys.stderr)
            return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert dom_score.shape == (500,)

    k8 = load_k8_correct(len(y))
    if k8 is None:
        print("MISSING_REGEN_INPUT", K8_DIR, file=sys.stderr)
        return 2

    # ---- closed-form, label-free CoE-C analogue on L19 prefill states ----
    mean_vec = X.mean(axis=0)
    norms = np.linalg.norm(X, axis=1)
    mean_norm = float(np.linalg.norm(mean_vec))
    cos_to_mean = (X @ mean_vec) / (norms * mean_norm + 1e-12)
    coe_mag = zscore(norms)                 # magnitude component
    coe_ang = zscore(cos_to_mean)           # angle component
    coe_c_raw = coe_mag + coe_ang           # combined CoE-C

    # orient each score so higher = more confident (label-free gate, sign only)
    coe_c, coe_c_auroc, coe_c_flipped = orient(coe_c_raw, y)
    dom, dom_auroc, _ = orient(dom_score, y)

    # ---- selective-prediction protocol (identical for both gates) ----
    coe_sel = selective_acc(coe_c, k8, COVERAGE, K_LO, K_HI)
    dom_sel = selective_acc(dom, k8, COVERAGE, K_LO, K_HI)
    rand = selective_acc(np.zeros(len(y)), k8, COVERAGE, K_LO, K_HI)  # no gating order

    f8_ref = None
    if F8_RESULTS.exists():
        try:
            f8_ref = json.loads(F8_RESULTS.read_text())
        except Exception:
            f8_ref = None

    coe_acc = coe_sel["acc_answered"]
    out = {
        "experiment": "P11-FE337",
        "description": "CoE-C-gated vs DoM-gated selective prediction (F-8 re-attribution)",
        "coverage_target": COVERAGE,
        "k_schedule": {"k_lo": K_LO, "k_hi": K_HI, "k_avg_expected": 2.5},
        "f8_target_acc": F8_TARGET,
        "auroc": {
            "dom_oriented": dom_auroc,
            "coe_c_oriented": coe_c_auroc,
            "coe_mag_raw": auroc(coe_mag, y),
            "coe_ang_raw": auroc(coe_ang, y),
            "coe_c_raw_unoriented": auroc(coe_c_raw, y),
        },
        "coe_c_flipped_for_orientation": bool(coe_c_flipped),
        "selective": {
            "coe_c_gate": coe_sel,
            "dom_gate": dom_sel,
            "ungated_baseline": rand,
        },
        "coe_c_acc_at_coverage_0p5": coe_acc,
        "dom_acc_at_coverage_0p5": dom_sel["acc_answered"],
        "matches_f8": bool(coe_acc >= F8_TARGET),
        "delta_vs_dom": float(coe_acc - dom_sel["acc_answered"]),
        "verdict": (
            "RE-ATTRIBUTABLE: label-free CoE-C matches F-8 target"
            if coe_acc >= F8_TARGET else
            "NOT re-attributable: CoE-C gate falls below F-8 target"
        ),
        "f8_reference_loaded": f8_ref is not None,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "coe_c_acc@0.5": coe_acc,
        "dom_acc@0.5": dom_sel["acc_answered"],
        "matches_f8": out["matches_f8"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())