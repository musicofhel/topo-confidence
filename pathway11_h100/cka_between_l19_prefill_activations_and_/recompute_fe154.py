"""P11-FE154 — CKA: L19 prefill input-geometry vs target-geometry alignment.

Tests whether the L19 prefill stream is better explained as an input-feature
detector than as a correctness direction. Computes linear CKA (Centered Kernel
Alignment) between the (500, 1536) L19 prefill activations and three targets:

  (a) correctness one-hot          — target geometry (what F-2 claims it tracks)
  (b) MATH-500 problem-token mean-pool — input geometry (the problem itself)
  (c) topic / subject one-hot       — coarse input geometry

Motivation (paper 2401.13558): ReLU-family transformers approximately preserve
input geometry through their layers. If so, the prefill DoM is an input-feature
detector and its correctness AUROC is inherited from an input<->correctness
correlation in the data. The discriminating prediction is

    CKA(L19, problem-tokens) >> CKA(L19, correctness)

which would refute F-2's framing of the prefill DoM as a *correctness* direction.

Targets (b) and (c) require auxiliary cached inputs (problem mean-pool NPZ,
subject metadata). They are optional: if absent the corresponding CKA is recorded
as null with a status note, and the verdict is marked incomplete. The main
prefill cache is required.
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
OUT_JSON = ROOT / "pathway11_h100/cka_input_geometry/results.json"

# Candidate auxiliary inputs (optional). First existing/parseable wins.
PROBLEM_POOL_KEYS = ["problem_meanpool", "input_meanpool", "problem_tokens_mean",
                     "embed_mean", "problem_pool", "input_mean"]
PROBLEM_POOL_NPZS = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_problem_meanpool.npz",
    ROOT / "pathway11_h100/data/problem_token_meanpool.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_input_meanpool.npz",
]
META_PATHS = [
    ROOT / "data/math500.jsonl",
    ROOT / "data/math500_meta.json",
    ROOT / "data/math500.json",
    ROOT / "pathway11_h100/data/math500_meta.json",
    ROOT / "configs/math500_subjects.json",
]
SUBJECT_FIELDS = ["subject", "type", "topic", "category"]

N = 500
SEED = 9999


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F).

    Rows are samples; columns are features. Both matrices are column-centered.
    Invariant to isotropic scaling and orthogonal transforms; in [0, 1].
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    cross = X.T @ Y                       # (p, q)
    hsic_xy = float((cross ** 2).sum())   # ||Y^T X||_F^2
    gx = X.T @ X
    gy = Y.T @ Y
    hsic_xx = float((gx ** 2).sum())
    hsic_yy = float((gy ** 2).sum())
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-30:
        return float("nan")
    return hsic_xy / denom


def cka_permutation_p(X: np.ndarray, Y: np.ndarray, observed: float,
                      n_perm: int = 200, seed: int = SEED) -> float:
    """Row-permutation null for CKA: fraction of permuted CKAs >= observed."""
    if not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        if linear_cka(X, Y[perm]) >= observed:
            ge += 1
    return float((ge + 1) / (n_perm + 1))


def load_problem_pool(blob) -> tuple[np.ndarray | None, str | None]:
    # First check the main cache for an embedded problem-pool key.
    for key in PROBLEM_POOL_KEYS:
        if key in getattr(blob, "files", []):
            arr = np.asarray(blob[key])
            if arr.ndim == 2 and arr.shape[0] == N:
                return arr.astype(np.float64), f"{CACHE.name}:{key}"
    # Then scan candidate auxiliary NPZs.
    for path in PROBLEM_POOL_NPZS:
        if not path.exists():
            continue
        try:
            aux = np.load(path)
        except Exception:
            continue
        for key in PROBLEM_POOL_KEYS + list(aux.files):
            if key in aux.files:
                arr = np.asarray(aux[key])
                if arr.ndim == 2 and arr.shape[0] == N:
                    return arr.astype(np.float64), f"{path.name}:{key}"
    return None, None


def load_topic_onehot() -> tuple[np.ndarray | None, str | None, list[str] | None]:
    for path in META_PATHS:
        if not path.exists():
            continue
        try:
            text = path.read_text()
            if path.suffix == ".jsonl":
                records = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                obj = json.loads(text)
                records = obj if isinstance(obj, list) else obj.get("problems", obj.get("data", []))
        except Exception:
            continue
        if not isinstance(records, list) or len(records) < N:
            continue
        field = None
        for f in SUBJECT_FIELDS:
            if isinstance(records[0], dict) and f in records[0]:
                field = f
                break
        if field is None:
            continue
        subjects = [str(r.get(field, "")) for r in records[:N]]
        cats = sorted(set(subjects))
        idx = {c: i for i, c in enumerate(cats)}
        onehot = np.zeros((N, len(cats)), dtype=np.float64)
        for i, s in enumerate(subjects):
            onehot[i, idx[s]] = 1.0
        return onehot, f"{path.name}:{field}", cats
    return None, None, None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    correct = blob["correct"].astype(bool)
    assert X.shape == (N, 1536) and correct.shape == (N,)

    # (a) correctness one-hot — always available.
    y_oh = np.column_stack([(~correct).astype(np.float64), correct.astype(np.float64)])
    cka_correct = linear_cka(X, y_oh)
    p_correct = cka_permutation_p(X, y_oh, cka_correct)

    # DoM correctness AUROC for context (what the input<->correctness story explains).
    dom_auroc = None
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        dom_auroc = auroc(dom_score, correct)

    # (b) problem-token mean-pool — optional input geometry.
    pool, pool_src = load_problem_pool(blob)
    if pool is not None:
        cka_problem = linear_cka(X, pool)
        p_problem = cka_permutation_p(X, pool, cka_problem)
        problem_status = "OK"
    else:
        cka_problem = None
        p_problem = None
        problem_status = "MISSING_REGEN_INPUT"
        print("MISSING_REGEN_INPUT problem_meanpool (CKA(b) skipped)", file=sys.stderr)

    # (c) topic / subject one-hot — optional coarse input geometry.
    topic_oh, topic_src, topic_cats = load_topic_onehot()
    if topic_oh is not None:
        cka_topic = linear_cka(X, topic_oh)
        p_topic = cka_permutation_p(X, topic_oh, cka_topic)
        topic_status = "OK"
    else:
        cka_topic = None
        p_topic = None
        topic_status = "MISSING_REGEN_INPUT"
        print("MISSING_REGEN_INPUT topic_meta (CKA(c) skipped)", file=sys.stderr)

    # Verdict: input geometry >> correctness geometry would refute F-2 framing.
    ratio_problem = (cka_problem / cka_correct) if (cka_problem and cka_correct) else None
    ratio_topic = (cka_topic / cka_correct) if (cka_topic and cka_correct) else None
    complete = problem_status == "OK"
    refutes_f2_framing = bool(ratio_problem is not None and ratio_problem >= 2.0)

    if not complete:
        interpretation = ("INCOMPLETE — problem-token mean-pool unavailable; "
                          "cannot evaluate the discriminating CKA(L19, problem) >> "
                          "CKA(L19, correctness) comparison.")
    elif refutes_f2_framing:
        interpretation = ("Input geometry dominates: CKA(L19, problem) >> "
                          "CKA(L19, correctness). Consistent with 2401.13558 — prefill "
                          "DoM behaves as an input-feature detector; weakens F-2's "
                          "correctness-direction framing.")
    else:
        interpretation = ("Correctness geometry not subsumed by input geometry: "
                          "CKA(L19, problem) does not dominate CKA(L19, correctness). "
                          "F-2's correctness-direction framing survives this test.")

    out = {
        "experiment": "P11-FE154",
        "n": int(N),
        "prefill_dim": int(X.shape[1]),
        "cka_method": "linear_CKA (column-centered, HSIC ratio)",
        "n_perm": 200,
        "cka_correct_onehot": cka_correct,
        "cka_correct_perm_p": p_correct,
        "cka_problem_meanpool": cka_problem,
        "cka_problem_perm_p": p_problem,
        "cka_topic_onehot": cka_topic,
        "cka_topic_perm_p": p_topic,
        "ratio_problem_over_correct": ratio_problem,
        "ratio_topic_over_correct": ratio_topic,
        "dom_correctness_auroc": dom_auroc,
        "problem_pool_source": pool_src,
        "problem_pool_status": problem_status,
        "topic_source": topic_src,
        "topic_categories": topic_cats,
        "topic_status": topic_status,
        "complete": complete,
        "refutes_f2_correctness_framing": refutes_f2_framing,
        "interpretation": interpretation,
        "trigger_paper": "2401.13558",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("cka_correct_onehot", "cka_problem_meanpool",
                       "cka_topic_onehot", "ratio_problem_over_correct",
                       "complete", "refutes_f2_correctness_framing")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())