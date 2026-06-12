#!/usr/bin/env python3
"""v7_phase1c_t5.py — SPEC v7 Phase 1c (EXP-86): confirmatory T5, one shot.

Executes the Gate-G2 frozen recipe (SPEC_v7.md), exactly one evaluation per
arm per family. Families: SmolLM2-1.7B, Gemma-2-2b-it, OLMo-2-1B (MATH-500
cells, phase3_*.npz). Protocol identical to dev: frozen 5-fold OOF,
StandardScaler+LR(C=1.0), DeLong increments.

Adjudications (frozen at G2):
  - rescore: free+mean_token_entropy NO-HARM (combined >= free - 0.01) per
    family; other token arms descriptive. Alignment corr >= 0.90 enforced
    by v7_pod_to_sidecar.
  - P(True): fmtB adjudicates; fmtA recorded (sensitivity).
  - verbalized: descriptive; a family showing combined delta >= +0.01
    p < 0.05 is flagged as dev-contradiction, not promoted.
  - K=8 Gemma only: k{2,4,8} agreement + cluster entropy, signal view vs
    greedy correctness + majority system view; graded locally with the
    canonical causal_dom grader (check_correct).
  - H-I at T5: no frozen arm with rel_cost <= 1.05x adds >= +0.01 (p<0.05).

Output: results/v7_phase1c_t5.json
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "causal_dom"))

from common import _extract_boxed, check_correct                 # noqa: E402
from metrics import frozen_folds, auroc_sym                      # noqa: E402
from v7_phase1a import arm_row, impute_median, oof_scores        # noqa: E402
from v7_pod_to_sidecar import convert, convert_genscore          # noqa: E402

RESULTS = HERE / "results"
POD = HERE / "pod_results"
FAMILIES = ("smollm2", "gemma", "olmo2")
# Deviation D-4: SmolLM2/OLMo-2 tokenizers fail decode->encode round-trip
# (66%/0% exact), invalidating teacher-forced rescoring from cached texts.
# Their token arms come from the pod genscore task (features captured at
# generation time, faithfulness gated by text equality vs the pinned cache,
# non-matching rows NaN'd). Gemma round-trips (corr 0.9998) -> rescore.
SIDECAR_MODE = {"smollm2": "genscore", "gemma": "rescore",
                "olmo2": "genscore"}


def pod_json(name: str):
    f = POD / name
    if name.endswith(".gz"):
        with gzip.open(f, "rt") as fh:
            return json.load(fh)
    return json.loads(f.read_text())


class T5Cell:
    def __init__(self, key: str):
        d = np.load(RESULTS / f"phase3_{key}.npz", allow_pickle=True)
        self.key = key
        self.y = d["y"].astype(bool)
        self.texts = [str(t) for t in d["texts"]]
        self.base_X = np.column_stack([d["n_gen_tokens"].astype(float),
                                       d["mean_logprob"].astype(float)])
        self.folds = frozen_folds(self.y)
        self.free_oof = oof_scores(self.base_X, self.y, self.folds)
        self.free_auroc = auroc_sym(self.y, self.free_oof)

    def row(self, name, x, note=""):
        xi, cov = impute_median(np.asarray(x, dtype=np.float64))
        r = arm_row(name, xi, self.base_X, self.y, self.folds,
                    self.free_oof, self.free_auroc)
        r["finite_fraction"] = cov
        if note:
            r["note"] = note
        return r


def family_table(cell: T5Cell) -> dict:
    mode = SIDECAR_MODE[cell.key]
    if mode == "rescore":
        convert(f"{cell.key}_t5")        # pod JSON -> sidecar; corr gate
        side = np.load(RESULTS / f"v7_rescore_{cell.key}_t5.npz")
    else:                                # D-4: gen-time features
        convert_genscore(f"{cell.key}_t5")
        side = np.load(RESULTS / f"v7_genscore_{cell.key}_t5.npz")
    assert bool((side["y"] == cell.y).all())

    rows = []
    for a in ("min_lp", "p10_lp", "mean_entropy", "mean_top2_margin",
              "ans_span_lp"):
        rows.append(cell.row(f"token_{a}", side[a]))
    pt = pod_json(f"v7_ptrue_{cell.key}_t5.json")
    assert list(map(bool, pt["y"])) == list(map(bool, cell.y))
    rows.append(cell.row("ptrue_fmtB", pt["p_true_B"],
                         note="ADJUDICATING (frozen at G2)"))
    rows.append(cell.row("ptrue_fmtA", pt["p_true_A"], note="sensitivity"))
    vb = pod_json(f"v7_verbalized_{cell.key}_t5.json")
    assert list(map(bool, vb["y"])) == list(map(bool, cell.y))
    for arm, key, pf in (("verb_binary", "verb_binary",
                          vb["parse_failure_binary"]),
                         ("verb_score_0to100", "verb_score",
                          vb["parse_failure_score"])):
        r = cell.row(arm, vb[key], note="descriptive")
        r["parse_failure"] = pf
        rows.append(r)

    ent = next(r for r in rows if r["arm"] == "token_mean_entropy")
    no_harm = bool(ent["combined_auroc"] >= cell.free_auroc - 0.01)
    t = {"free_gate_oof_auroc": cell.free_auroc,
         "token_sidecar_mode": mode,
         "alignment_corr": float(side["alignment_corr"]),
         "entropy_no_harm": no_harm, "arms": rows}
    if mode == "genscore":
        t["text_match_frac"] = float(side["text_match_frac"])
    return t


def gemma_k8(cell: T5Cell) -> dict:
    d = pod_json("v7_k8t5_gemma.json.gz")
    rows = sorted(d["rows"], key=lambda r: r["idx"])
    n = len(rows)
    assert n == len(cell.y)
    greedy_ans = [_extract_boxed(t) or "" for t in cell.texts]

    feats = {K: {"agreement": np.zeros(n), "entropy": np.zeros(n),
                 "majority_correct": np.zeros(n, dtype=bool)}
             for K in (2, 4, 8)}
    for i, r in enumerate(rows):
        ans = [_extract_boxed(t) or "" for t in r["texts"]]
        corr = [check_correct(t, r["answer"]) for t in r["texts"]]
        for K in (2, 4, 8):
            sub = ans[:K]
            o = feats[K]
            o["agreement"][i] = float(np.mean(
                [a == greedy_ans[i] and a != "" for a in sub]))
            _, counts = np.unique(sub, return_counts=True)
            p = counts / K
            o["entropy"][i] = float(-(p * np.log(p)).sum())
            nonempty = [a for a in sub if a != ""]
            if nonempty:
                mode = Counter(nonempty).most_common(1)[0][0]
                o["majority_correct"][i] = any(
                    c for a, c in zip(sub, corr[:K]) if a == mode)

    out = {"meta": {k: d[k] for k in ("K", "temperature", "seed",
                                      "cost_tokens_generated")},
           "system_view": {"acc_greedy": float(cell.y.mean()),
                           **{f"acc_k{K}_majority":
                              float(feats[K]["majority_correct"].mean())
                              for K in (2, 4, 8)}},
           "signal_view": []}
    for K in (2, 4, 8):
        out["signal_view"].append(
            cell.row(f"k{K}_agreement", feats[K]["agreement"]))
        out["signal_view"].append(
            cell.row(f"k{K}_cluster_entropy", feats[K]["entropy"]))
    return out


def main():
    out = {"spec": "v7 Phase 1c (EXP-86) — confirmatory T5, one shot",
           "families": {}}
    cells = {}
    for key in FAMILIES:
        cells[key] = T5Cell(key)
        out["families"][key] = family_table(cells[key])

    out["k8_gemma"] = gemma_k8(cells["gemma"])

    # H-I at T5: any <=1.05x arm (token family) with delta>=+0.01, p<0.05?
    refuters = []
    for key, t in out["families"].items():
        for r in t["arms"]:
            if (r["arm"].startswith("token_")
                    and r["incremental_delta"] >= 0.01
                    and r["delong_p"] < 0.05):
                refuters.append({"family": key, **{k: r[k] for k in
                                 ("arm", "incremental_delta", "delong_p")}})
    out["H_I_t5"] = {"holds": not refuters, "refuted_by": refuters}
    out["entropy_no_harm_all_families"] = all(
        t["entropy_no_harm"] for t in out["families"].values())

    f = RESULTS / "v7_phase1c_t5.json"
    f.write_text(json.dumps(out, indent=1, default=float))
    print(f"-> {f}\n")
    for key, t in out["families"].items():
        print(f"{key}: free={t['free_gate_oof_auroc']:.4f} "
              f"corr={t['alignment_corr']:.4f} "
              f"entropy_no_harm={t['entropy_no_harm']}")
        for r in t["arms"]:
            print(f"  {r['arm']:22s} alone={r['standalone_auroc']:.4f} "
                  f"comb={r['combined_auroc']:.4f} "
                  f"d={r['incremental_delta']:+.4f} p={r['delong_p']:.3f}")
    print(f"\nH-I at T5 holds: {out['H_I_t5']['holds']} "
          f"{out['H_I_t5']['refuted_by'] or ''}")
    print(f"entropy no-harm all families: "
          f"{out['entropy_no_harm_all_families']}")
    print(f"\nGemma K=8 system view: {out['k8_gemma']['system_view']}")
    for r in out["k8_gemma"]["signal_view"]:
        print(f"  {r['arm']:22s} alone={r['standalone_auroc']:.4f} "
              f"comb={r['combined_auroc']:.4f} "
              f"d={r['incremental_delta']:+.4f} p={r['delong_p']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
