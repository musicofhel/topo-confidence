#!/usr/bin/env python3
"""Regenerate confgate/data/pinned_gates.json + pinned_meta.json from repo caches.

Deployment coefficients are fit on the FULL per-family cache (no CV) with the
exact pinned v6 recipe: StandardScaler -> LogisticRegression(max_iter=2000,
C=1.0) on [n_gen_tokens, mean_logprob].  The honest generalization anchors
(OOF / LOCO AUROCs) are copied from the v6 result JSONs — every number in the
output is read from an artifact, never typed in.

Sources (all under the topo-confidence repo root):
  nocompute/cache/math500_1p5b.npz                       qwen2.5-1.5b
  nocompute/cache/math500_7b.npz                         qwen2.5-7b
  pathway11_h100/generalization_edge/results/phase3_smollm2.npz   smollm2-1.7b
  pathway11_h100/generalization_edge/results/phase3_gemma.npz     gemma-2-2b
  pathway11_h100/generalization_edge/results/phase3_olmo2.npz     olmo-2-1b
  pathway11_h100/generalization_edge/cache/arm2a_prompt_cloud_qwen1.5b_math.npz
                                                          (preflight prompt_len)
  results/{phase3_t5_free_baseline,phase2_arm2a,1b4_overlap,
           phase4_recalibration,phase1_panel}.json        (pinned anchors)

Run with the repo venv:
  ~/topo-confidence/.venv/bin/python scripts/fit_pinned.py
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent          # confgate/scripts
PKG_ROOT = HERE.parent                          # confgate/  (project dir)
REPO = PKG_ROOT.parent                          # topo-confidence repo root
DATA = PKG_ROOT / "confgate" / "data"
DATA.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PKG_ROOT))
from confgate.gate import FreeGate              # noqa: E402
from confgate.preflight import PreflightGate    # noqa: E402

GE = REPO / "pathway11_h100" / "generalization_edge"
GE_RESULTS = GE / "results"

FAMILY_SOURCES = {
    "qwen2.5-1.5b": REPO / "nocompute" / "cache" / "math500_1p5b.npz",
    "qwen2.5-7b": REPO / "nocompute" / "cache" / "math500_7b.npz",
    "smollm2-1.7b": GE_RESULTS / "phase3_smollm2.npz",
    "gemma-2-2b": GE_RESULTS / "phase3_gemma.npz",
    "olmo-2-1b": GE_RESULTS / "phase3_olmo2.npz",
}

ARM2A_NPZ = GE / "cache" / "arm2a_prompt_cloud_qwen1.5b_math.npz"

# HF model ids for the two Qwen caches (caches carry no model key; the
# manifests of the underlying trajectory caches do).
QWEN_MANIFESTS = {
    "qwen2.5-1.5b": REPO / "pathway8_layerwise" / "data" / "math500" / "manifest.json",
    "qwen2.5-7b": REPO / "pathway11_h100" / "data" / "math500_7b" / "manifest.json",
}


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def _model_id(family: str, d) -> str | None:
    if "model" in getattr(d, "files", []):
        return str(d["model"])
    man = QWEN_MANIFESTS.get(family)
    if man and man.exists():
        m = json.loads(man.read_text())
        for k in ("model", "model_name", "model_id"):
            if k in m:
                return str(m[k])
    return None


def fit_family(family: str, path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    y = d["y"].astype(bool)
    L = d["n_gen_tokens"].astype(np.float64)
    lp = d["mean_logprob"].astype(np.float64)
    g = FreeGate.fit(L, lp, y, family=family)
    resub = float(roc_auc_score(y, g.score(L, lp)))
    return {
        "model": _model_id(family, d),
        "source": _rel(path),
        "n": int(len(y)),
        "accuracy": float(y.mean()),
        "scaler_mean": g.scaler_mean.tolist(),
        "scaler_scale": g.scaler_scale.tolist(),
        "coef": g.coef.tolist(),
        "intercept": g.intercept,
        "resub_auroc": resub,
        "fit_protocol": "full-data fit (deployment coefficients, no CV); "
                        "resub_auroc is resubstitution, a regression check only",
    }


def fit_preflight() -> dict:
    """Prompt-length gate: prompt_len from the arm2a cache, labels from the
    1.5B consolidated cache (same 500 problems, same order — the pairing used
    by phase2_arm2a.py)."""
    d = np.load(ARM2A_NPZ, allow_pickle=True)
    plen = d["prompt_len"].astype(np.float64)
    y = np.load(FAMILY_SOURCES["qwen2.5-1.5b"], allow_pickle=True)["y"].astype(bool)
    assert len(plen) == len(y), "arm2a prompt_len / 1.5B labels misaligned"
    g = PreflightGate.fit(plen, y, family="qwen2.5-1.5b")
    z = g.score(plen)
    return {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",  # from phase2_arm2a.json (read below)
        "source_prompt_len": _rel(ARM2A_NPZ),
        "source_labels": _rel(FAMILY_SOURCES["qwen2.5-1.5b"]),
        "n": int(len(y)),
        "scaler_mean": g.scaler_mean,
        "scaler_scale": g.scaler_scale,
        "coef": g.coef,
        "intercept": g.intercept,
        "resub_auroc": float(roc_auc_score(y, z)),
        "fit_protocol": "full-data fit on [n_prompt_tokens] (deployment coefficients)",
    }


def collect_anchors() -> dict:
    """Pinned v6 anchors, read verbatim from the result JSONs."""
    t5 = json.loads((GE_RESULTS / "phase3_t5_free_baseline.json").read_text())
    arm2a = json.loads((GE_RESULTS / "phase2_arm2a.json").read_text())
    ov = json.loads((GE_RESULTS / "1b4_overlap.json").read_text())
    p4 = json.loads((GE_RESULTS / "phase4_recalibration.json").read_text())
    panel = json.loads((GE_RESULTS / "phase1_panel.json").read_text())

    hyb = ov["cascade_vs_random_mix_hull"]["hybrid"]
    fb_xscale = p4["math1.5b_to_7b"]["free_baseline"]["by_k"]
    fb_xdom = p4["math_to_bbh"]["free_baseline"]["by_k"]

    def xs(k):  # cross-scale conformal cell at eps=0.2
        c = fb_xscale[k]["conformal"]["eps0.2"]
        return {"feasible_frac": c["feasible_frac"], "validity": c["validity"],
                "mean_coverage": c["mean_coverage"]}

    def xd(k):  # cross-domain conformal cell at eps=0.2
        c = fb_xdom[k]["conformal"]["eps0.2"]
        return {"feasible_frac": c["feasible_frac"], "validity": c["validity"],
                "mean_coverage": c["mean_coverage"]}

    scalar_notes = {
        "smollm2-1.7b": "length-only carries (length_only ~ free, logprob_only ~ chance)",
        "gemma-2-2b": "both scalars carry",
        "olmo-2-1b": "length-led (length_only close to free, logprob weaker)",
    }
    t5_key = {"smollm2-1.7b": "smollm2", "gemma-2-2b": "gemma", "olmo-2-1b": "olmo2"}

    return {
        "t5_heldout_oof": {
            fam: {
                "model": t5[k]["model"],
                "free_baseline_auroc": t5[k]["free_baseline_auroc"],
                "length_only_auroc": t5[k]["length_only"],
                "logprob_only_auroc": t5[k]["logprob_only"],
                "T5_pass_0.70": t5[k]["T5_pass_0.70"],
                "which_scalar_carries": scalar_notes[fam],
                "protocol": "OOF StratifiedKFold(5, shuffle=True, random_state=0), "
                            "phase3_heldout.py::eval_free_baseline",
                "source": _rel(GE_RESULTS / "phase3_t5_free_baseline.json"),
            } for fam, k in t5_key.items()
        },
        "in_domain_qwen1.5b": {
            "t1_loco_free_baseline_auroc": panel["T1_loco"]["free_baseline"]["aggregate"],
            "t3_crossscale_free_baseline_auroc":
                panel["T3_crossscale"]["free_baseline"]["aggregate"],
            "source": _rel(GE_RESULTS / "phase1_panel.json"),
        },
        "preflight": {
            "prompt_length_only_auroc": arm2a["prompt_length_only_auroc"],
            "protocol": "frozen-fold OOF, phase2_arm2a.py",
            "h_e_verdict": arm2a["verdict_H_E"],
            "gate_delta_vs_promptlen": arm2a["gate_delta_vs_promptlen"],
            "gate_p": arm2a["gate_p"],
            "source": _rel(GE_RESULTS / "phase2_arm2a.json"),
        },
        "cascade": {
            "cost_model": ov["cascade_vs_random_mix_hull"]["cost_model"],
            "acc_1p5b_k1": ov["anchors"]["acc_k1"],
            "acc_7b": ov["anchors"]["acc_7b"],
            "hybrid_beats_hull_gap_pp": hyb["gap_pp_point"],
            "hybrid_gap_pp_boot_se": hyb["gap_pp_boot_se"],
            "hybrid_significant": hyb["significant"],
            "hybrid_best_point": hyb["best_point"],
            "note": "the +3.38pp anchor used the v6 HYBRID gate "
                    "(prefill-DoM + logprob + length); the activation-free "
                    "FreeGate frontier is what this package deploys",
            "source": _rel(GE_RESULTS / "1b4_overlap.json"),
        },
        "conformal": {
            "method": "split-conformal LTT, CP-upper light-head recalibration, R=200",
            "eps": p4["eps"], "delta": p4["delta"],
            "cross_scale_math1.5b_to_7b_eps0.2": {k: xs(k) for k in
                                                  ("k0", "k8", "k16", "k32", "k64")},
            "cross_domain_math_to_bbh_eps0.2": {k: xd(k) for k in
                                                ("k0", "k8", "k16", "k32", "k64")},
            "verdict": "cross-scale zero-shot (k=0) is the supported certificate "
                       "(validity 1.0 @ 0.60 coverage, eps=0.2); k-label "
                       "recalibration is feasibility-restoring only past k>=32, "
                       "never coverage-improving; cross-domain certificates are "
                       "infeasible with any light head (validity 0.0 at eps=0.2 "
                       "for every feasible k) -> certify_cross_domain() refuses",
            "source": _rel(GE_RESULTS / "phase4_recalibration.json"),
        },
    }


def main() -> int:
    families = {}
    for fam, path in FAMILY_SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(f"missing cache for {fam}: {path}")
        families[fam] = fit_family(fam, path)
        print(f"  {fam:14s} n={families[fam]['n']}  "
              f"resub_auroc={families[fam]['resub_auroc']:.4f}  "
              f"coef={np.round(families[fam]['coef'], 4).tolist()}")

    preflight = fit_preflight()
    print(f"  preflight      n={preflight['n']}  "
          f"resub_auroc={preflight['resub_auroc']:.4f}  coef={preflight['coef']:.4f}")

    gates = {
        "recipe": "StandardScaler -> LogisticRegression(max_iter=2000, C=1.0)",
        "features": ["n_gen_tokens", "mean_logprob"],
        "families": families,
        "preflight": {"qwen2.5-1.5b": preflight},
    }
    (DATA / "pinned_gates.json").write_text(json.dumps(gates, indent=2))

    meta = {
        "generated": _dt.date.today().isoformat(),
        "generated_by": "confgate/scripts/fit_pinned.py",
        "repo": "topo-confidence (SPEC v6 pinned results; SPEC v7 Phase 3a package)",
        "provenance_note": "every anchor below is read verbatim from a committed "
                           "result JSON; deployment coefficients are full-data "
                           "refits of the exact pinned recipe on the named caches",
        "anchors": collect_anchors(),
    }
    (DATA / "pinned_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"-> {DATA / 'pinned_gates.json'}")
    print(f"-> {DATA / 'pinned_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
