"""Verdict-driven moot sweep — close adjacent FutureExperiments after a result lands.

When an experiment produces a clarifying verdict, many open FEs become dead
(premise refuted) or redundant (question already answered) without ever being
run. Premise RELIES_ON edges handle the known fault lines deterministically
(premises.py); this script handles the semantic long tail:

  1. embed the verdict text (all-MiniLM-L6-v2, same space as the graph),
  2. cosine-shortlist the top-K open FEs,
  3. adjudicate each batch with `claude -p` under a CONSERVATIVE rubric
     (default KEEP — a false moot silently kills live work, the opposite
     polarity of the inclusive admission gates),
  4. write MOOTED/ANSWERED statuses + MOOTED_BY/ANSWERED_BY edges,
     a review report under briefs/, and regenerate NEXT_EXPERIMENTS.md.

Usage:
    # Preview (no writes):
    python moot_sweep.py --by FE269 --verdict "L19 prefill DoM is a
        correlational readout, NOT a causal lever; steering is dead." --dry-run

    # Apply, optionally tagging mooted FEs with the premise they relied on:
    python moot_sweep.py --by FE269 --verdict-file verdict.txt \\
        --premise dom-causal-lever

    # Consume verdicts queued by promote_result.py:
    python moot_sweep.py --from-trigger

Adjudications are cached in cache/llm-cache.json keyed on
(closing id, fe id, verdict hash, model), so re-runs are free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from llm_cache import cache_get, cache_set

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")

REPO_ROOT = ROOT.parent
TRIGGER_FILE = REPO_ROOT / ".autopilot" / "moot-trigger"
BRIEFS_DIR = ROOT / "briefs"

OPEN_STATUSES = ["READY", "TRIGGERED", "BLOCKED"]
VALID_VERDICTS = {"MOOT", "ANSWERED", "KEEP"}

ADJUDICATION_PROMPT = """\
You are auditing a research project's experiment backlog after a new result \
landed. For each candidate experiment below, decide whether the verdict closes it.

THE VERDICT (from experiment {by}):
{verdict}

Return verdict for each candidate:
- "MOOT"     — the experiment's PREMISE is refuted by the verdict. Running it \
would test an assumption we now know is false (e.g. it assumes a signal is a \
causal lever, a feature family adds value, or a routing strategy helps, and \
the verdict refuted exactly that).
- "ANSWERED" — the experiment's QUESTION is already settled by the verdict \
(the number or comparison it wants now exists, even if measured differently).
- "KEEP"     — still live. The verdict does not directly settle it, it tests \
something orthogonal, or you are uncertain.

BE CONSERVATIVE. A wrong MOOT silently kills live work; a wrong KEEP just \
leaves one stale entry. When in doubt: KEEP. Do NOT moot an experiment merely \
because it is thematically related to the verdict — the verdict must refute \
its premise or answer its question.

CANDIDATES:
{candidates}

Respond with ONLY a JSON array, one object per candidate, no markdown fences:
[{{"id": "...", "verdict": "MOOT|ANSWERED|KEEP", "reason": "<one sentence>"}}]
"""


def _driver():
    return GraphDatabase.driver(
        BOLT, auth=(USER, PASSWORD), notifications_min_severity="OFF",
    )


def _embed(text: str) -> list[float]:
    from backfill_embeddings import embed_texts
    return embed_texts([text])[0]


def shortlist(verdict: str, top_k: int, min_sim: float) -> list[dict]:
    """Cosine-rank open FEs against the verdict. Embeddings are normalized,
    so dot product == cosine."""
    vec = _embed(verdict)
    with _driver() as drv, drv.session() as s:
        missing = s.run(
            """
            MATCH (fe:FutureExperiment)
            WHERE fe.status IN $open AND fe.embedding IS NULL
            RETURN count(fe) AS n
            """, open=OPEN_STATUSES,
        ).single()["n"]
        if missing:
            print(f"  WARNING: {missing} open FE(s) lack embeddings and are "
                  "invisible to the sweep. Run: python backfill_embeddings.py "
                  "--node-type FutureExperiment")
        rows = list(s.run(
            """
            MATCH (fe:FutureExperiment)
            WHERE fe.status IN $open AND fe.embedding IS NOT NULL
            WITH fe, reduce(dot = 0.0, i IN range(0, size(fe.embedding)-1) |
                            dot + fe.embedding[i] * $vec[i]) AS sim
            WHERE sim >= $min_sim
            RETURN fe.id AS id, fe.description AS description,
                   fe.rationale AS rationale, fe.status AS status, sim
            ORDER BY sim DESC LIMIT $k
            """,
            open=OPEN_STATUSES, vec=vec, min_sim=min_sim, k=top_k,
        ))
    return [dict(r) for r in rows]


def _verdict_hash(by: str, verdict: str, model: str) -> str:
    return hashlib.sha256(f"{by}|{model}|{verdict}".encode()).hexdigest()[:16]


def _parse_json_array(raw: str) -> list[dict]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON array in adjudicator output: {raw[:300]}")
    return json.loads(text[start:end + 1])


def adjudicate_batch(batch: list[dict], by: str, verdict: str,
                     model: str, timeout: int = 300) -> list[dict]:
    """One `claude -p` call over a batch of candidates. Per-FE results cached."""
    vhash = _verdict_hash(by, verdict, model)
    out: list[dict] = []
    uncached: list[dict] = []
    for fe in batch:
        hit = cache_get("moot", f"{vhash}:{fe['id']}")
        if hit is not None:
            out.append(json.loads(hit))
        else:
            uncached.append(fe)
    if not uncached:
        return out

    candidates = "\n".join(
        f"- id: {fe['id']}\n  what: {(fe['description'] or '')[:600]}\n"
        f"  why: {(fe['rationale'] or '')[:300]}"
        for fe in uncached
    )
    prompt = ADJUDICATION_PROMPT.format(by=by, verdict=verdict,
                                        candidates=candidates)
    result = subprocess.run(
        ["claude", "-p", "--model", model],
        input=prompt, capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed: {result.stderr[:500]}")
    decisions = _parse_json_array(result.stdout)

    by_id = {d.get("id"): d for d in decisions if isinstance(d, dict)}
    for fe in uncached:
        d = by_id.get(fe["id"])
        if not d or d.get("verdict") not in VALID_VERDICTS:
            d = {"id": fe["id"], "verdict": "KEEP",
                 "reason": "adjudicator returned no/invalid verdict — kept"}
        cache_set("moot", f"{vhash}:{fe['id']}", json.dumps(d))
        out.append(d)
    return out


def apply_decisions(decisions: list[dict], by: str, premise: str | None,
                    dry_run: bool) -> dict[str, int]:
    today = date.today().isoformat()
    counts = {"MOOT": 0, "ANSWERED": 0, "KEEP": 0}
    with _driver() as drv, drv.session() as s:
        for d in decisions:
            counts[d["verdict"]] += 1
            if d["verdict"] == "KEEP" or dry_run:
                continue
            status = "MOOTED" if d["verdict"] == "MOOT" else "ANSWERED"
            edge = "MOOTED_BY" if status == "MOOTED" else "ANSWERED_BY"
            s.run(
                """
                MATCH (fe:FutureExperiment {id: $id})
                WHERE fe.status IN $open
                SET fe.status = $status, fe.outcome = $reason,
                    fe.closed_by = $by, fe.blocked_by = null,
                    fe.completed_date = $today
                """,
                id=d["id"], open=OPEN_STATUSES, status=status,
                reason=f"{status} by {by}: {d['reason']}", by=by, today=today,
            )
            s.run(
                f"""
                MATCH (fe:FutureExperiment {{id: $id}})
                MATCH (src) WHERE (src:FutureExperiment OR src:Experiment
                                   OR src:Finding) AND src.id = $by
                MERGE (fe)-[r:{edge}]->(src)
                SET r.reason = $reason, r.date = $today
                """,
                id=d["id"], by=by, reason=d["reason"], today=today,
            )
            if status == "MOOTED" and premise:
                s.run(
                    """
                    MATCH (fe:FutureExperiment {id: $id}),
                          (pr:Premise {id: $pid})
                    MERGE (fe)-[:RELIES_ON]->(pr)
                    """,
                    id=d["id"], pid=premise,
                )
    return counts


def write_report(decisions: list[dict], shortlisted: list[dict], by: str,
                 verdict: str, counts: dict[str, int], dry_run: bool) -> Path:
    sim_by_id = {fe["id"]: fe["sim"] for fe in shortlisted}
    lines = [
        f"# Moot sweep — {by} ({date.today().isoformat()})"
        + (" [DRY RUN]" if dry_run else ""),
        "",
        f"**Verdict:** {verdict}",
        "",
        f"**Shortlisted:** {len(shortlisted)} | **MOOT:** {counts['MOOT']} | "
        f"**ANSWERED:** {counts['ANSWERED']} | **KEEP:** {counts['KEEP']}",
        "",
        "| FE | sim | verdict | reason |",
        "|---|---|---|---|",
    ]
    order = {"MOOT": 0, "ANSWERED": 1, "KEEP": 2}
    for d in sorted(decisions, key=lambda x: (order[x["verdict"]], x["id"])):
        reason = d["reason"].replace("|", "/")
        lines.append(f"| {d['id']} | {sim_by_id.get(d['id'], 0):.3f} | "
                     f"{d['verdict']} | {reason} |")
    lines += ["", "Review: wrongly-closed FEs can be revived with",
              "`python update_status.py <id> READY`.", ""]
    BRIEFS_DIR.mkdir(exist_ok=True)
    path = BRIEFS_DIR / f"moot-sweep-{date.today().isoformat()}-{by}.md"
    path.write_text("\n".join(lines))
    return path


def sweep(by: str, verdict: str, premise: str | None, top_k: int,
          min_sim: float, batch_size: int, model: str, dry_run: bool) -> None:
    print(f"\nMoot sweep — closing experiment: {by}"
          + (" [DRY RUN]" if dry_run else ""))
    print(f"  verdict: {verdict[:160]}{'…' if len(verdict) > 160 else ''}")

    shortlisted = shortlist(verdict, top_k=top_k, min_sim=min_sim)
    print(f"  shortlisted {len(shortlisted)} open FE(s) "
          f"(top-{top_k}, cos ≥ {min_sim})")
    if not shortlisted:
        print("  nothing to adjudicate.")
        return

    decisions: list[dict] = []
    for i in range(0, len(shortlisted), batch_size):
        batch = shortlisted[i:i + batch_size]
        decisions.extend(adjudicate_batch(batch, by, verdict, model))
        done = min(i + batch_size, len(shortlisted))
        closed = sum(1 for d in decisions if d["verdict"] != "KEEP")
        print(f"  adjudicated {done}/{len(shortlisted)} ({closed} to close)")

    counts = apply_decisions(decisions, by, premise, dry_run)
    report = write_report(decisions, shortlisted, by, verdict, counts, dry_run)
    verb = "would close" if dry_run else "closed"
    print(f"\n  {verb}: {counts['MOOT']} MOOTED, {counts['ANSWERED']} ANSWERED "
          f"({counts['KEEP']} kept)")
    print(f"  report: {report}")

    if not dry_run and (counts["MOOT"] or counts["ANSWERED"]):
        subprocess.run([sys.executable, str(ROOT / "generate_next_experiments.py")],
                       check=False)


def consume_trigger() -> list[tuple[str, str]]:
    """Read and clear .autopilot/moot-trigger. Lines: <by-id>\\t<verdict text>."""
    if not TRIGGER_FILE.exists():
        return []
    entries = []
    for line in TRIGGER_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            by, verdict = line.split("\t", 1)
            entries.append((by.strip(), verdict.strip()))
    TRIGGER_FILE.write_text("")
    return entries


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--by", help="Id of the closing experiment (e.g. FE269)")
    p.add_argument("--verdict", help="The clarifying outcome, one paragraph")
    p.add_argument("--verdict-file", type=Path,
                   help="Read the verdict text from a file instead")
    p.add_argument("--from-trigger", action="store_true",
                   help="Consume queued verdicts from .autopilot/moot-trigger")
    p.add_argument("--premise", default=None,
                   help="Premise id to RELIES_ON-link mooted FEs to (provenance)")
    p.add_argument("--top-k", type=int, default=150)
    p.add_argument("--min-sim", type=float, default=0.30,
                   help="Cosine floor for the shortlist (default 0.30)")
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--model", default="haiku",
                   help="Model for `claude -p` adjudication (default haiku)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.from_trigger:
        entries = consume_trigger()
        if not entries:
            print("No queued verdicts in .autopilot/moot-trigger.")
            return
        for by, verdict in entries:
            sweep(by, verdict, None, args.top_k, args.min_sim,
                  args.batch_size, args.model, args.dry_run)
        return

    if not args.by:
        p.error("--by is required (or use --from-trigger)")
    verdict = args.verdict
    if args.verdict_file:
        verdict = args.verdict_file.read_text().strip()
    if not verdict:
        p.error("--verdict or --verdict-file is required")

    sweep(args.by, verdict, args.premise, args.top_k, args.min_sim,
          args.batch_size, args.model, args.dry_run)


if __name__ == "__main__":
    main()
