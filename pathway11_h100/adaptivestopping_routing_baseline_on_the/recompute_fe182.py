"""P11-FE182 — SOFT-SC adaptive-stopping routing baseline on cached K=8 MATH-500.

Simulates SOFT-SC's adaptive procedure on the existing K=8 self-consistency
generations: walk the K samples in order, stop early as soon as a sample's
sequence confidence (min-token-log-prob) crosses a dev-tuned threshold τ, and
take that sample's answer. Sweep τ to trace the accuracy vs average-K Pareto.

The point of the experiment: SOFT-SC's adaptive variant needs no hidden-state
extraction and no verification pass — only the sequence likelihood the model
already emits. If this Pareto matches H-19's planned C_exact verification
routing at matched average K, the verification-routing story collapses into
"use the logits you already have."

Data dependency: this requires a per-sample sequence-confidence array in the
K=8 NPZ files (min token log-prob, or token-level log-probs we can min over).
The headline schema only guarantees `correct: (8,) bool`; if no log-prob field
is present we cannot simulate the likelihood gate and emit MISSING_REGEN_INPUT.
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
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
OUT_JSON = ROOT / "pathway11_h100/softsc_adaptive_routing/results.json"

# Candidate H-19 C_exact verification-routing result locations (compared if present).
H19_CANDIDATES = [
    ROOT / "pathway11_h100/verification_routing/results.json",
    ROOT / "pathway11_h100/results/h19_cexact_routing.json",
    ROOT / "pathway11_h100/cexact_routing/results.json",
]

# Confidence field candidates inside each problem_NNN.npz. Higher == more
# confident. Token-level (8, T) arrays are reduced by min-over-tokens.
CONF_KEYS = [
    "min_logprob", "min_token_logprob", "seq_min_logprob",
    "seq_logprob", "sequence_logprob", "mean_logprob",
    "logprob", "logprobs", "token_logprobs", "confidence", "conf",
]

N_PROBLEMS = 500
K = 8
SEED = 9999
N_TAU = 81
TARGET_AVG_K = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def auroc(scores, labels):
    labels = labels.astype(bool)
    pos = scores[labels]; neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def _reduce_conf(arr):
    """Coerce a per-sample confidence array to shape (K,), higher == confident."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2:           # (K, T) token log-probs -> min over tokens
        a = a.min(axis=1)
    a = a.reshape(-1)
    return a


def _detect_conf_key(npz):
    for key in CONF_KEYS:
        if key in npz.files:
            return key
    return None


def load_k8():
    """Returns (correct: (N,K) bool, conf: (N,K) float64) or (None, reason)."""
    if not K8_DIR.exists():
        return None, f"missing dir {K8_DIR}"
    files = sorted(K8_DIR.glob("problem_*.npz"))
    if not files:
        return None, f"no problem_*.npz in {K8_DIR}"

    conf_key = None
    correct_rows, conf_rows = [], []
    for f in files:
        with np.load(f) as npz:
            if "correct" not in npz.files:
                return None, f"{f.name} has no 'correct' field"
            c = np.asarray(npz["correct"]).astype(bool).reshape(-1)
            if conf_key is None:
                conf_key = _detect_conf_key(npz)
                if conf_key is None:
                    return None, ("no sequence-confidence field "
                                  f"(tried {CONF_KEYS}) in {f.name}")
            if conf_key not in npz.files:
                return None, f"{f.name} missing confidence field '{conf_key}'"
            v = _reduce_conf(npz[conf_key])
        if c.shape[0] != K or v.shape[0] != K:
            return None, f"{f.name}: expected K={K}, got correct={c.shape} conf={v.shape}"
        correct_rows.append(c)
        conf_rows.append(v)

    correct = np.vstack(correct_rows)
    conf = np.vstack(conf_rows).astype(np.float64)
    return (correct, conf, conf_key), None


def apply_tau(conf, correct, tau):
    """SOFT-SC stop-on-first-confident-sample. Fallback = argmax confidence.

    Returns (accuracy, avg_used_k)."""
    n = conf.shape[0]
    idx = np.arange(n)
    crossed_mask = conf >= tau                       # (N, K)
    crossed = crossed_mask.any(axis=1)               # (N,)
    first = crossed_mask.argmax(axis=1)              # first True (0 if none)
    fallback = conf.argmax(axis=1)                   # most-confident sample
    chosen = np.where(crossed, first, fallback)
    used_k = np.where(crossed, first + 1, K)
    pred_correct = correct[idx, chosen]
    return float(pred_correct.mean()), float(used_k.mean())


def tau_grid(conf):
    lo = float(conf.min()); hi = float(conf.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([lo - 1.0])
    qs = np.linspace(0.0, 1.0, N_TAU)
    grid = np.quantile(conf.reshape(-1), qs)
    # prepend a τ below the floor so "never stop early" (avg_k -> K) is reachable
    return np.unique(np.concatenate([[lo - abs(lo) - 1.0], grid]))


def load_h19():
    for p in H19_CANDIDATES:
        if p.exists():
            try:
                data = json.loads(p.read_text())
            except Exception as exc:  # noqa: BLE001
                return {"status": "unreadable", "path": str(p), "error": str(exc)}
            return {"status": "found", "path": str(p), "data": data}
    return {"status": "pending",
            "note": "H-19 C_exact verification-routing result not yet on disk; "
                    "compare SOFT-SC pareto against it when it lands.",
            "searched": [str(p) for p in H19_CANDIDATES]}


def main() -> int:
    loaded, reason = load_k8()
    if loaded is None:
        print("MISSING_REGEN_INPUT", reason, file=sys.stderr)
        return 2

    correct, conf, conf_key = loaded
    n = correct.shape[0]

    # --- baselines ---------------------------------------------------------
    k1_acc = float(correct[:, 0].mean())                      # single sample
    oracle_passk = float(correct.any(axis=1).mean())          # pass@K ceiling
    fullk_acc, _ = apply_tau(conf, correct, conf.min() - 1.0 + 1e30)  # never -> argmax
    # argmax-confidence at full budget (avg_k == K):
    bestconf_acc = float(correct[np.arange(n), conf.argmax(axis=1)].mean())

    # --- full-data Pareto over τ ------------------------------------------
    grid = tau_grid(conf)
    pareto = []
    for tau in grid:
        acc, avg_k = apply_tau(conf, correct, float(tau))
        pareto.append({"tau": float(tau), "avg_k": avg_k, "accuracy": acc})
    pareto.sort(key=lambda r: r["avg_k"])

    # --- dev/test-tuned operating points ----------------------------------
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    dev_idx, test_idx = perm[: n // 2], perm[n // 2:]
    conf_dev, corr_dev = conf[dev_idx], correct[dev_idx]
    conf_te, corr_te = conf[test_idx], correct[test_idx]

    operating = []
    for target in TARGET_AVG_K:
        # tune τ on dev to land avg_k closest to target
        best = None
        for tau in grid:
            _, avg_k_dev = apply_tau(conf_dev, corr_dev, float(tau))
            gap = abs(avg_k_dev - target)
            if best is None or gap < best[0]:
                best = (gap, float(tau), avg_k_dev)
        tau_star = best[1]
        te_acc, te_avg_k = apply_tau(conf_te, corr_te, tau_star)
        operating.append({
            "target_avg_k": target,
            "tau_star": tau_star,
            "dev_avg_k": best[2],
            "test_avg_k": te_avg_k,
            "test_accuracy": te_acc,
        })

    # confidence-as-correctness sanity AUROC on first sample
    conf_auroc_first = auroc(conf[:, 0], correct[:, 0])

    out = {
        "experiment": "P11-FE182",
        "title": "SOFT-SC adaptive-stopping routing baseline on K=8 MATH-500",
        "n_problems": int(n),
        "k": int(K),
        "conf_field_used": conf_key,
        "baselines": {
            "k1_accuracy": k1_acc,
            "pass_at_k_oracle": oracle_passk,
            "fullk_bestconf_accuracy": bestconf_acc,
            "fullk_accuracy_check": fullk_acc,
            "fullk_avg_k": float(K),
        },
        "conf_auroc_first_sample": conf_auroc_first,
        "pareto": pareto,
        "operating_points": operating,
        "h19_cexact_comparison": load_h19(),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())