"""P11-FE442 — Thought-count differential per A/B/C/D bucket (Refutation 4 for F-7).

Applies a SEAL-style keyword classifier (reflection + transition thoughts) to the
cached K=1 and K=8 generations, counts R+T thoughts per problem, buckets problems
by their K=8 pass rate (A = mostly correct … D = mostly wrong), and compares the
R+T distributions across buckets with Mann-Whitney U (focus: D vs A).

If the D-bucket carries significantly MORE reflection/transition thoughts than the
A-bucket, F-7's geometric signature is plausibly a downstream shadow of an aberrant
thought-pattern composition rather than a geometric primitive.

Behavioral generation text is required (the numeric NPZ caches do not carry it); if
no generation text can be located the script reports MISSING_REGEN_INPUT and exits 2.
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

try:
    from scipy.stats import mannwhitneyu
except Exception:  # scipy missing — fail loudly later
    mannwhitneyu = None

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
OUT_JSON = ROOT / "pathway11_h100/seal_thought_buckets/results.json"

N_PROBLEMS = 500

# Candidate locations for K=1 generation text (numeric caches carry no text).
K1_TEXT_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_gated_compute/generations.json",
    ROOT / "pathway11_h100/prefill_gated_compute/k1_generations.json",
    ROOT / "pathway11_h100/data/k1_generations.json",
    ROOT / "pathway11_h100/data/generations.json",
]
# NPZ fields that might hold generation text inside a problem_NNN.npz.
TEXT_FIELD_CANDIDATES = ("text", "texts", "generations", "completions", "responses", "outputs")

# SEAL-style reflection / transition lexicons (substring match, lowercased).
REFLECTION_KEYWORDS = (
    "wait", "actually", "hmm", "recheck", "re-check", "double check", "double-check",
    "let me verify", "let me check", "verify", "reconsider", "on second thought",
    "made a mistake", "mistake", "that's wrong", "is wrong", "incorrect",
    "re-examine", "reexamine", "rethink", "but wait", "however", "hold on",
    "let me recompute", "recompute", "scratch that", "correction",
)
TRANSITION_KEYWORDS = (
    "alternatively", "another approach", "another way", "another method",
    "different approach", "instead", "let's try", "let me try", "on the other hand",
    "moving on", "let's consider", "consider instead", "alternative", "or we could",
    "next,", "now,", "then,", "first,", "second,", "third,", "finally,",
)

SEGMENT_SPLIT = re.compile(r"[.!?\n;]+")


def auroc(scores, labels):
    pos = scores[labels]; neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def classify_text(text: str) -> tuple[int, int]:
    """Return (reflection_count, transition_count) over the segments of one generation."""
    if not text:
        return 0, 0
    refl = 0
    trans = 0
    for seg in SEGMENT_SPLIT.split(text.lower()):
        seg = seg.strip()
        if not seg:
            continue
        if any(kw in seg for kw in REFLECTION_KEYWORDS):
            refl += 1
        if any(kw in seg for kw in TRANSITION_KEYWORDS):
            trans += 1
    return refl, trans


def _coerce_text_list(obj) -> list[str]:
    """Normalize a loaded text payload into a flat list of strings."""
    out = []
    if obj is None:
        return out
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        for key in ("text", "generations", "completions", "response", "output", "completion"):
            if key in obj:
                return _coerce_text_list(obj[key])
        return out
    if isinstance(obj, (list, tuple, np.ndarray)):
        for item in obj:
            out.extend(_coerce_text_list(item))
        return out
    # numpy scalar / bytes
    try:
        s = obj.item() if hasattr(obj, "item") else obj
    except Exception:
        s = obj
    if isinstance(s, bytes):
        s = s.decode("utf-8", "ignore")
    if isinstance(s, str):
        out.append(s)
    return out


def load_k1_texts() -> list[list[str]] | None:
    """Return per-problem K=1 text lists, or None if no K=1 text source is found."""
    for path in K1_TEXT_CANDIDATES:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            data = data.get("generations", data.get("results", list(data.values())))
        if not isinstance(data, (list, tuple)) or len(data) < N_PROBLEMS:
            continue
        return [_coerce_text_list(data[i]) for i in range(N_PROBLEMS)]
    return None


def load_problem(idx: int):
    """Return (correct_array, k8_text_list) for problem idx, or (None, None) if absent."""
    p = K8_DIR / f"problem_{idx:03d}.npz"
    if not p.exists():
        return None, None
    blob = np.load(p, allow_pickle=True)
    correct = blob["correct"].astype(bool) if "correct" in blob.files else None
    texts: list[str] = []
    for field in TEXT_FIELD_CANDIDATES:
        if field in blob.files:
            texts.extend(_coerce_text_list(blob[field]))
    return correct, texts


def bucket_of(pass8: float) -> str:
    if pass8 >= 0.75:
        return "A"
    if pass8 >= 0.50:
        return "B"
    if pass8 >= 0.25:
        return "C"
    return "D"


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not K8_DIR.exists():
        print("MISSING_REGEN_INPUT", K8_DIR, file=sys.stderr); return 2
    if mannwhitneyu is None:
        print("MISSING_REGEN_INPUT scipy.stats.mannwhitneyu", file=sys.stderr); return 2

    cache = np.load(CACHE)
    k1_correct = cache["correct"].astype(bool)
    if k1_correct.shape[0] != N_PROBLEMS:
        print("MISSING_REGEN_INPUT unexpected cache shape", file=sys.stderr); return 2

    k1_texts = load_k1_texts()  # optional

    rt_counts = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    refl_counts = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    trans_counts = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    pass8 = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    buckets = np.empty(N_PROBLEMS, dtype="<U1")
    n_with_text = 0

    for i in range(N_PROBLEMS):
        correct, k8_texts = load_problem(i)
        if correct is not None and correct.size:
            pass8[i] = float(correct.mean())
        else:
            pass8[i] = float(k1_correct[i])  # fall back to K=1 label

        gen_texts: list[str] = list(k8_texts) if k8_texts else []
        if k1_texts is not None:
            gen_texts.extend(k1_texts[i])

        if gen_texts:
            r_total = t_total = 0
            for g in gen_texts:
                r, t = classify_text(g)
                r_total += r; t_total += t
            n_gen = max(len(gen_texts), 1)
            refl_counts[i] = r_total / n_gen
            trans_counts[i] = t_total / n_gen
            rt_counts[i] = (r_total + t_total) / n_gen
            n_with_text += 1

        buckets[i] = bucket_of(pass8[i])

    if n_with_text == 0:
        print("MISSING_REGEN_INPUT no generation text located for any problem "
              "(K=1 candidates + K=8 npz text fields all absent)", file=sys.stderr)
        return 2

    valid = ~np.isnan(rt_counts)
    bucket_stats = {}
    bucket_rt = {}
    for b in ("A", "B", "C", "D"):
        mask = valid & (buckets == b)
        vals = rt_counts[mask]
        bucket_rt[b] = vals
        bucket_stats[b] = {
            "n": int(mask.sum()),
            "mean_RT": float(np.mean(vals)) if vals.size else None,
            "median_RT": float(np.median(vals)) if vals.size else None,
            "mean_reflection": float(np.mean(refl_counts[mask])) if vals.size else None,
            "mean_transition": float(np.mean(trans_counts[mask])) if vals.size else None,
        }

    def mw(a, b, alternative):
        if a.size == 0 or b.size == 0:
            return None
        try:
            u, p = mannwhitneyu(a, b, alternative=alternative)
            return {"U": float(u), "p": float(p)}
        except Exception:
            return None

    d_vs_a_two = mw(bucket_rt["D"], bucket_rt["A"], "two-sided")
    d_gt_a = mw(bucket_rt["D"], bucket_rt["A"], "greater")

    pairwise = {}
    order = ["A", "B", "C", "D"]
    for i_b in range(len(order)):
        for j_b in range(i_b + 1, len(order)):
            bi, bj = order[i_b], order[j_b]
            pairwise[f"{bj}_vs_{bi}_two_sided"] = mw(bucket_rt[bj], bucket_rt[bi], "two-sided")

    # Sanity: does the geometric primitive (DoM proxy via K=1 correctness) survive?
    auroc_rt_vs_k1 = auroc(rt_counts[valid], k1_correct[valid])

    significant = bool(d_gt_a is not None and d_gt_a["p"] < 0.05)
    out = {
        "experiment": "P11-FE442",
        "refutation": "R4_thought_pattern_shadow",
        "n_problems": N_PROBLEMS,
        "n_with_generation_text": int(n_with_text),
        "k1_text_source_found": bool(k1_texts is not None),
        "bucket_scheme": "A: pass8>=0.75, B: >=0.50, C: >=0.25, D: <0.25",
        "bucket_stats": bucket_stats,
        "mannwhitney_D_vs_A_two_sided": d_vs_a_two,
        "mannwhitney_D_greater_than_A": d_gt_a,
        "pairwise_two_sided": pairwise,
        "auroc_RT_count_predicts_k1_correct": float(auroc_rt_vs_k1),
        "D_bucket_has_more_RT_thoughts": significant,
        "verdict": (
            "REFRAME_F7_AS_THOUGHT_SHADOW" if significant
            else "F7_GEOMETRY_NOT_EXPLAINED_BY_RT_COMPOSITION"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())