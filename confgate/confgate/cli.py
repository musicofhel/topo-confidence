"""confgate CLI.

  confgate demo --dataset math500     reproduce the pinned anchors from the
                                      repo caches (requires the topo-confidence
                                      repo; set CONFGATE_REPO if not default)
  confgate score <file.jsonl>         score JSONL rows -> gate score + route
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from .certify import certify, apply_certificate
from .gate import FreeGate, pinned_meta
from .preflight import PreflightGate
from .route import Cascade

DEFAULT_REPO = Path(os.environ.get("CONFGATE_REPO",
                                   Path.home() / "topo-confidence"))
ANCHOR_TOL = 1e-6


def _load_cache(repo: Path, name: str):
    p = repo / "nocompute" / "cache" / name
    if not p.exists():
        sys.exit(f"confgate demo needs the repo cache {p} "
                 "(set CONFGATE_REPO to the topo-confidence checkout)")
    d = np.load(p, allow_pickle=True)
    return (d["y"].astype(bool), d["n_gen_tokens"].astype(np.float64),
            d["mean_logprob"].astype(np.float64))


def cmd_demo(args) -> int:
    from sklearn.metrics import roc_auc_score

    if args.dataset != "math500":
        sys.exit("only --dataset math500 is wired up")
    repo = Path(args.repo)
    meta = pinned_meta()

    y15, L15, lp15 = _load_cache(repo, "math500_1p5b.npz")
    y7, L7, lp7 = _load_cache(repo, "math500_7b.npz")

    # 1) pinned 1.5B gate ----------------------------------------------------
    g15 = FreeGate.from_pinned("qwen2.5-1.5b")
    s15 = g15.score(L15, lp15)
    auroc = float(roc_auc_score(y15, s15))
    pin = g15.meta["resub_auroc"]
    ok = abs(auroc - pin) <= ANCHOR_TOL
    print(f"[gate]    qwen2.5-1.5b MATH-500: resub AUROC {auroc:.6f} "
          f"(pinned {pin:.6f}) {'OK' if ok else 'MISMATCH'}")
    if not ok:
        sys.exit("anchor mismatch: pinned 1.5B resubstitution AUROC")

    g7 = FreeGate.from_pinned("qwen2.5-7b")
    s7_own = g7.score(L7, lp7)
    auroc7 = float(roc_auc_score(y7, s7_own))
    pin7 = g7.meta["resub_auroc"]
    ok7 = abs(auroc7 - pin7) <= ANCHOR_TOL
    print(f"[gate]    qwen2.5-7b   MATH-500: resub AUROC {auroc7:.6f} "
          f"(pinned {pin7:.6f}) {'OK' if ok7 else 'MISMATCH'}")
    if not ok7:
        sys.exit("anchor mismatch: pinned 7B resubstitution AUROC")

    # 2) cascade frontier ----------------------------------------------------
    rows = Cascade.frontier(s15, y15, y7)
    best = Cascade.best_gap(rows)
    canchor = meta["anchors"]["cascade"]
    print(f"[cascade] free-gate frontier (keep 1.5B if score >= tau, else 7B; "
          f"{canchor['cost_model']}; resubstitution scores — "
          "OOF anchors live in pinned_meta):")
    for r in rows[:: max(1, len(rows) // 10)]:
        print(f"    keep={r['pct_kept']:.2f} cost_rel={r['cost_rel']:.3f} "
              f"acc={r['accuracy']:.3f} hull={r['hull_acc']:.3f} "
              f"gap={r['gap_pp']:+.2f}pp")
    print(f"    best gap vs random-mix hull: {best['gap_pp']:+.2f}pp at "
          f"keep={best['pct_kept']:.2f} (tau={best['tau']:.3f})")
    print(f"    pinned v6 HYBRID-gate anchor: "
          f"{canchor['hybrid_beats_hull_gap_pp']:+.2f}pp "
          f"(boot SE {canchor['hybrid_gap_pp_boot_se']:.2f}, "
          f"{'SIG' if canchor['hybrid_significant'] else 'n.s.'}) — "
          "hybrid used activations; this package ships the free gate")

    # 3) conformal certificate (cross-scale zero-shot, the supported path) ---
    eps, delta = 0.2, 0.1
    cert = certify(s15, y15, eps=eps, delta=delta)
    print(f"[certify] calibrate on 1.5B (n={cert['n_cal']}, eps={eps}, "
          f"delta={delta}): tau={cert['tau']:.4f} "
          f"cal_coverage={cert['cal_coverage']:.3f}")
    s7_frozen = g15.score(L7, lp7)   # frozen 1.5B gate scored on the 7B cell
    dep = apply_certificate(cert, s7_frozen, y=y7)
    pinc = meta["anchors"]["conformal"]["cross_scale_math1.5b_to_7b_eps0.2"]["k0"]
    print(f"          deploy zero-shot on 7B: coverage={dep['coverage']:.3f} "
          f"risk={dep['test_risk']:.3f} (<= eps? {dep['risk_le_eps']})")
    print(f"          pinned k=0 cell: validity {pinc['validity']}, "
          f"coverage {pinc['mean_coverage']}")
    if not dep["risk_le_eps"]:
        sys.exit("cross-scale zero-shot certificate violated on the demo cell")

    print("[demo] all pinned anchors reproduced")
    return 0


def cmd_score(args) -> int:
    gate = FreeGate.from_pinned(args.family)
    pf = None
    casc = Cascade(gate=gate, tau=args.tau)
    n_bad = 0
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                g, lp = float(row["gen_tokens"]), float(row["mean_logprob"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                n_bad += 1
                continue
            s = float(gate.score([g], [lp])[0])
            out = {"gate_score": round(s, 6),
                   "route": casc.decide([s])[0]}
            if "prompt_tokens" in row and row["prompt_tokens"] is not None:
                if pf is None:
                    pf = PreflightGate.from_pinned()
                out["preflight_score"] = round(
                    float(pf.score([float(row["prompt_tokens"])])[0]), 6)
            print(json.dumps(out))
    if n_bad:
        print(f"# skipped {n_bad} unparseable line(s)", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="confgate",
        description="Free confidence gate for LLM correctness "
                    "(pinned topo-confidence SPEC-v6 result)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="reproduce pinned anchors from repo caches")
    d.add_argument("--dataset", default="math500")
    d.add_argument("--repo", default=str(DEFAULT_REPO),
                   help="topo-confidence checkout (or set CONFGATE_REPO)")
    d.set_defaults(fn=cmd_demo)

    s = sub.add_parser("score", help="score a JSONL of generations")
    s.add_argument("jsonl", help='lines: {"prompt_tokens":int(opt),'
                                 '"gen_tokens":int,"mean_logprob":float}')
    s.add_argument("--family", default="qwen2.5-1.5b")
    s.add_argument("--tau", type=float, default=0.5,
                   help="route threshold: keep if gate_score >= tau")
    s.set_defaults(fn=cmd_score)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
