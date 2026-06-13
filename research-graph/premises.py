"""Premise nodes — first-class research assumptions that FutureExperiments rely on.

A :Premise is a short, falsifiable statement that many FEs implicitly assume
("the DoM direction is a causal lever"). When an experiment refutes a premise,
every open FE with a RELIES_ON edge to it is MOOTED in one deterministic
cascade — no LLM involved. The semantic long tail (FEs without RELIES_ON
edges) is handled separately by moot_sweep.py.

Usage:
    python premises.py seed                 # MERGE the controlled vocabulary (idempotent)
    python premises.py list                 # show premises + reliant-FE counts
    python premises.py link P11-FE44 dom-steering
    python premises.py refute dom-steering --by FE269 \\
        --reason "Causal noising verdict DIAGNOSTIC — DoM is a readout, not a lever."
    python premises.py confirm free-baseline-strongest-readout --by P11-FE-EDGEGEN \\
        --reason "T5 PASS on three held-out families (0.81/0.84/0.84)."

Refuting cascades immediately: open FEs (READY/TRIGGERED/BLOCKED) that RELY_ON
the premise become MOOTED with a MOOTED_BY edge to the premise. Linking an FE
to an already-REFUTED premise also moots it on the spot (same rule
promote_brief.py applies at admission).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")

OPEN_STATUSES = ["READY", "TRIGGERED", "BLOCKED"]

# Controlled vocabulary. Add entries here (or via `seed` after editing) rather
# than ad-hoc MERGEs, so premise ids stay stable for triage briefs' relies-on:.
SEED_PREMISES = [
    {
        "id": "dom-causal-lever",
        "statement": "The L19 prefill DoM direction is a causal lever — "
                     "intervening on it (patching, ablating, noising) changes "
                     "answer correctness.",
        "status": "REFUTED",
        "refuted_by": "FE269",
        "reason": "FE269 causal test 2026-06-09: verdict DIAGNOSTIC — L19 "
                  "prefill DoM is a correlational readout, not a causal lever.",
    },
    {
        "id": "dom-steering",
        "statement": "Steering generation along DoM/correctness directions "
                     "(fixed vector or scheduled) can improve math accuracy.",
        "status": "REFUTED",
        "refuted_by": "FE269",
        "reason": "P10 v2 direction-rotation refuted fixed-vector steering "
                  "(cos<0.2 to final DoM at every position); FE269 showed the "
                  "direction is diagnostic, not causal — no lever to steer.",
    },
    {
        "id": "verification-routing-helps",
        "statement": "Routing low-confidence problems to self-verification / "
                     "re-generation (C_exact) improves accuracy at 1.5B scale.",
        "status": "REFUTED",
        "refuted_by": "FE19",
        "reason": "FE19 2026-06-10: H-19 refuted — self-correction hurts at "
                  "1.5B; verification routing loses to the selective-prediction "
                  "baseline.",
    },
    {
        "id": "geometry-portable-signal",
        "statement": "Activation-geometry features (token-cloud spectra, "
                     "layer-depth profiles, prompt clouds, PH summaries) add "
                     "portable correctness signal over free baselines "
                     "(length, logprob).",
        "status": "REFUTED",
        "refuted_by": "P11-FE-EDGEGEN",
        "reason": "SPEC v6 (EXP-81/82) 2026-06-11: H-C refuted (depth-grid "
                  "n.s., transfers ≤0.63), H-E refuted (prompt-cloud 0.694 < "
                  "prompt-length 0.706, p=0.42), token-cloud spectrum negative "
                  "incremental. Geometry fully closed.",
    },
    {
        "id": "conformal-cross-domain",
        "statement": "Light-head recalibration (k ≤ 64 target labels) can "
                     "restore a valid conformal selective-prediction "
                     "certificate on a harder out-of-domain task.",
        "status": "REFUTED",
        "refuted_by": "P11-FE-EDGEGEN",
        "reason": "v6 Phase 4: no k≤64 head buys a valid certificate "
                  "MATH→BBH even at ε=0.3 — target-task accuracy is the "
                  "ceiling, not calibration.",
    },
    {
        "id": "escalation-beats-introspection",
        "statement": "At matched token-FLOP cost, no confidence probe "
                     "(token-level features, K-consistency, P(True), "
                     "verbalized, spectral-α, PRM, cost-aware router) beats "
                     "spending the same tokens on escalation to a stronger "
                     "model.",
        "status": "CONFIRMED",
        "refuted_by": None,
        "reason": "SPEC v7 (EXP-84) 2026-06-12, P11-FE-SHIPGATE: H-I held on "
                  "dev + confirmatory T5; K=8 majority 0.554 vs cascade 0.732 "
                  "(H-J); router −0.40pp p=0.68 (H-M); PRM-7B superb verifier "
                  "(0.94) but loses at matched cost. Binding constraint is "
                  "rescue density (47% of 1.5B failures unrescuable by 7B), "
                  "not ranking quality.",
    },
    {
        "id": "free-baseline-strongest-readout",
        "statement": "The free length+logprob readout is the most generalizing "
                     "correctness signal — any proposed readout must beat it "
                     "out-of-family to matter.",
        "status": "CONFIRMED",
        "refuted_by": None,
        "reason": "v6 T1 LOCO 0.845 / T3 0.865; T5 held-out PASS on SmolLM2 "
                  "0.810, Gemma-2 0.844, OLMo-2 0.838 (EXP-82/83).",
    },
]

VALID_PREMISE_STATUS = {"LIVE", "REFUTED", "CONFIRMED"}


def _driver():
    return GraphDatabase.driver(
        BOLT, auth=(USER, PASSWORD), notifications_min_severity="OFF",
    )


def cascade_moot(session, premise_id: str, closed_by: str, reason: str,
                 dry_run: bool = False) -> list[str]:
    """MOOT every open FE that RELIES_ON the (refuted) premise."""
    today = date.today().isoformat()
    if dry_run:
        rows = session.run(
            """
            MATCH (fe:FutureExperiment)-[:RELIES_ON]->(p:Premise {id: $pid})
            WHERE fe.status IN $open
            RETURN fe.id AS id ORDER BY fe.id
            """,
            pid=premise_id, open=OPEN_STATUSES,
        )
        return [r["id"] for r in rows]
    rows = session.run(
        """
        MATCH (fe:FutureExperiment)-[:RELIES_ON]->(p:Premise {id: $pid})
        WHERE fe.status IN $open
        SET fe.status = 'MOOTED',
            fe.outcome = $reason,
            fe.closed_by = $by,
            fe.blocked_by = null,
            fe.completed_date = $today
        MERGE (fe)-[r:MOOTED_BY]->(p)
        SET r.reason = $reason, r.date = $today
        RETURN fe.id AS id ORDER BY fe.id
        """,
        pid=premise_id, open=OPEN_STATUSES, by=closed_by,
        reason=f"Premise '{premise_id}' refuted by {closed_by}: {reason}",
        today=today,
    )
    return [r["id"] for r in rows]


def cmd_seed(args) -> None:
    today = date.today().isoformat()
    with _driver() as drv, drv.session() as s:
        for p in SEED_PREMISES:
            s.run(
                """
                MERGE (pr:Premise {id: $id})
                SET pr.statement = $statement,
                    pr.status = coalesce(pr.status, $status),
                    pr.refuted_by = coalesce(pr.refuted_by, $refuted_by),
                    pr.reason = coalesce(pr.reason, $reason),
                    pr.created_date = coalesce(pr.created_date, $today),
                    pr.status_date = coalesce(pr.status_date, $today)
                """,
                id=p["id"], statement=p["statement"], status=p["status"],
                refuted_by=p["refuted_by"], reason=p["reason"], today=today,
            )
            print(f"  ok Premise {p['id']} [{p['status']}]")
    print(f"\nSeeded {len(SEED_PREMISES)} premises (idempotent — existing "
          "status/provenance preserved).")


def cmd_list(_args) -> None:
    with _driver() as drv, drv.session() as s:
        rows = list(s.run(
            """
            MATCH (pr:Premise)
            OPTIONAL MATCH (fe:FutureExperiment)-[:RELIES_ON]->(pr)
            WITH pr, count(fe) AS reliant,
                 sum(CASE WHEN fe.status IN $open THEN 1 ELSE 0 END) AS open_reliant
            RETURN pr.id AS id, pr.status AS status, pr.statement AS statement,
                   pr.refuted_by AS refuted_by, pr.status_date AS status_date,
                   reliant, open_reliant
            ORDER BY pr.id
            """,
            open=OPEN_STATUSES,
        ))
    print(f"\nPremises ({len(rows)}):")
    for r in rows:
        prov = f" by {r['refuted_by']}" if r["refuted_by"] else ""
        print(f"\n  {r['id']} [{r['status']}{prov}, {r['status_date']}] — "
              f"{r['reliant']} reliant FE(s), {r['open_reliant']} still open")
        print(f"    {r['statement']}")


def cmd_link(args) -> None:
    today = date.today().isoformat()
    with _driver() as drv, drv.session() as s:
        rec = s.run(
            """
            MATCH (fe:FutureExperiment {id: $fid}), (pr:Premise {id: $pid})
            MERGE (fe)-[:RELIES_ON]->(pr)
            RETURN fe.status AS fe_status, pr.status AS pr_status,
                   pr.refuted_by AS refuted_by, pr.reason AS reason
            """,
            fid=args.fe_id, pid=args.premise_id,
        ).single()
        if not rec:
            print(f"FE {args.fe_id!r} or Premise {args.premise_id!r} not found.")
            sys.exit(1)
        print(f"  ok {args.fe_id} -[RELIES_ON]-> {args.premise_id}")
        if rec["pr_status"] == "REFUTED" and rec["fe_status"] in OPEN_STATUSES:
            mooted = cascade_moot(s, args.premise_id,
                                  rec["refuted_by"] or args.premise_id,
                                  rec["reason"] or "premise refuted")
            print(f"  premise already REFUTED -> mooted: {', '.join(mooted)}")
            _regen_hint()


def _set_status(args, new_status: str) -> None:
    today = date.today().isoformat()
    dry = getattr(args, "dry_run", False)
    with _driver() as drv, drv.session() as s:
        if dry:
            rec = s.run("MATCH (pr:Premise {id: $pid}) RETURN pr.id AS id",
                        pid=args.premise_id).single()
        else:
            rec = s.run(
                """
                MATCH (pr:Premise {id: $pid})
                SET pr.status = $status, pr.refuted_by = $by,
                    pr.reason = $reason, pr.status_date = $today
                RETURN pr.id AS id
                """,
                pid=args.premise_id, status=new_status, by=args.by,
                reason=args.reason, today=today,
            ).single()
        if not rec:
            print(f"No Premise {args.premise_id!r}. Run `premises.py list`.")
            sys.exit(1)
        verb = "would become" if dry else "->"
        print(f"Premise {args.premise_id} {verb} {new_status} (by {args.by})")

        if new_status == "REFUTED":
            mooted = cascade_moot(s, args.premise_id, args.by, args.reason,
                                  dry_run=args.dry_run)
            verb = "would moot" if args.dry_run else "mooted"
            print(f"  cascade {verb} {len(mooted)} open FE(s)"
                  + (f": {', '.join(mooted)}" if mooted else ""))
            if not args.dry_run and mooted:
                _regen_hint()


def _regen_hint() -> None:
    print("Run `python generate_next_experiments.py` to refresh "
          "NEXT_EXPERIMENTS.md.")


def cmd_refute(args) -> None:
    _set_status(args, "REFUTED")


def cmd_confirm(args) -> None:
    args.dry_run = False
    _set_status(args, "CONFIRMED")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seed", help="MERGE the controlled premise vocabulary")
    s.set_defaults(func=cmd_seed)

    s = sub.add_parser("list", help="Premises + reliant-FE counts")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("link", help="FE -[RELIES_ON]-> Premise (moots on the "
                                    "spot if the premise is already REFUTED)")
    s.add_argument("fe_id")
    s.add_argument("premise_id")
    s.set_defaults(func=cmd_link)

    s = sub.add_parser("refute", help="Refute a premise and cascade-MOOT "
                                      "reliant open FEs")
    s.add_argument("premise_id")
    s.add_argument("--by", required=True, help="Refuting experiment id")
    s.add_argument("--reason", required=True)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_refute)

    s = sub.add_parser("confirm", help="Confirm a premise (no cascade)")
    s.add_argument("premise_id")
    s.add_argument("--by", required=True)
    s.add_argument("--reason", required=True)
    s.set_defaults(func=cmd_confirm)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
