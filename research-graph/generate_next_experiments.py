"""Generate ~/topo-confidence/NEXT_EXPERIMENTS.md from the research graph.

Snapshot, not a live view — regenerate explicitly after seeding or status updates:

    python generate_next_experiments.py
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")

REPO_ROOT = ROOT.parent
OUTPUT = REPO_ROOT / "NEXT_EXPERIMENTS.md"

# Status-ordering for sorting within tier: TRIGGERED first, then READY, then BLOCKED
STATUS_ORDER = {"TRIGGERED": 0, "READY": 1, "BLOCKED": 2, "COMPLETED": 3,
                "ABANDONED": 4, "ANSWERED": 5, "MOOTED": 6}

# Closed-by-adjacency statuses: rendered as a compact one-line-per-FE section,
# never in the live tiers (a mooted FE revives only if its premise does).
CLOSED_STATUSES = ("MOOTED", "ANSWERED")


def _driver():
    return GraphDatabase.driver(
        BOLT, auth=(USER, PASSWORD), notifications_min_severity="OFF",
    )


def fetch_all() -> list[dict[str, Any]]:
    cypher = """
    MATCH (fe:FutureExperiment)
    OPTIONAL MATCH (fe)-[t:TRIGGERED_BY]->(p:Paper)
    OPTIONAL MATCH (fe)-[:DEPENDS_ON_FINDING]->(d:Finding)
    OPTIONAL MATCH (fe)-[:WOULD_UPDATE]->(u:Finding)
    WITH fe,
         collect(DISTINCT {
             arxiv_id: p.arxiv_id, title: p.title,
             their_method: t.their_method, their_result: t.their_result,
             our_method: t.our_method, same: t.same, differs: t.differs
         }) AS triggered_by,
         collect(DISTINCT d.id) AS depends_on,
         collect(DISTINCT u.id) AS would_update
    RETURN fe.id AS id, fe.pathway_id AS pathway_id,
           fe.description AS description, fe.rationale AS rationale,
           fe.trigger AS trigger, fe.status AS status,
           fe.blocked_by AS blocked_by, fe.priority AS priority,
           fe.estimated_cost AS estimated_cost, fe.roi_score AS roi_score,
           fe.created_date AS created_date, fe.completed_date AS completed_date,
           fe.outcome AS outcome, fe.closed_by AS closed_by,
           [t IN triggered_by WHERE t.arxiv_id IS NOT NULL] AS triggered_by,
           [x IN depends_on WHERE x IS NOT NULL] AS depends_on,
           [x IN would_update WHERE x IS NOT NULL] AS would_update
    ORDER BY coalesce(fe.roi_score, 0) DESC, fe.id
    """
    with _driver() as drv, drv.session() as s:
        return [dict(r) for r in s.run(cypher)]


def fetch_watchlist() -> list[dict[str, Any]]:
    cypher = """
    MATCH (fe:FutureExperiment)-[:TRIGGERED_BY]->(p:Paper)
    WHERE NOT fe.status IN ['COMPLETED', 'MOOTED', 'ANSWERED']
    WITH p, collect(DISTINCT {fe_id: fe.id, status: fe.status}) AS triggers
    RETURN p.arxiv_id AS arxiv_id, p.title AS title, p.year AS year,
           triggers
    ORDER BY p.year DESC, p.arxiv_id
    """
    with _driver() as drv, drv.session() as s:
        return [dict(r) for r in s.run(cypher)]


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _natkey(s: str) -> list:
    """Natural-sort key so FE27 sorts before FE100, deterministically."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _arxiv_link(arxiv_id: str, title: str | None = None) -> str:
    label = title or arxiv_id
    return f"[{label}](https://arxiv.org/abs/{arxiv_id})"


def _tier(roi: int | None) -> str:
    if roi is None:
        return "LOW"
    if roi >= 9:
        return "CRITICAL"
    if roi >= 7:
        return "HIGH"
    if roi >= 5:
        return "MEDIUM"
    return "LOW"


def render_source_paper(trig: dict[str, Any]) -> list[str]:
    """Emit the per-paper 'what they did vs what we did' bullet block.

    Bullets are emitted only when the corresponding edge property is set, so
    edges seeded before the comparison fields existed degrade gracefully.
    """
    arxiv_id = trig.get("arxiv_id")
    title = trig.get("title") or arxiv_id
    out = [f"**Source paper:** {_arxiv_link(arxiv_id, title)}"]
    if trig.get("their_method"):
        out.append(f"- *What they did:* {trig['their_method']}")
    if trig.get("their_result"):
        out.append(f"- *Their result:* {trig['their_result']}")
    if trig.get("our_method"):
        out.append(f"- *What we'll do:* {trig['our_method']}")
    if trig.get("same"):
        out.append(f"- *Same:* {trig['same']}")
    if trig.get("differs"):
        out.append(f"- *Differs:* {trig['differs']}")
    return out


def render_experiment(fe: dict[str, Any]) -> str:
    lines: list[str] = []
    priority = fe.get("priority") or "MEDIUM"
    title = f"### {fe['id']} ({fe['pathway_id']}) — [ROI: {fe['roi_score']}, {fe['status']}, {priority}]"
    lines.append(title)
    lines.append("")
    lines.append(f"**What:** {fe['description']}")
    lines.append("")
    if fe.get("rationale"):
        lines.append(f"**Why:** {fe['rationale']}")
        lines.append("")

    triggered = fe.get("triggered_by") or []
    annotated = [t for t in triggered if any(t.get(k) for k in ("their_method", "their_result", "our_method", "same", "differs"))]
    if annotated:
        for trig in annotated:
            lines.extend(render_source_paper(trig))
            lines.append("")
    elif not triggered:
        lines.append("**Source:** Internal re-validation — no external trigger paper.")
        lines.append("")

    cost = fe.get("estimated_cost") or "—"
    lines.append(f"**Cost:** {cost}")

    if fe.get("blocked_by"):
        lines.append(f"**Blocked by:** {fe['blocked_by']}")

    if fe.get("triggered_by"):
        formatted = ", ".join(
            _arxiv_link(t["arxiv_id"], t.get("title")) for t in fe["triggered_by"]
        )
        lines.append(f"**Triggered by:** {formatted}")

    if fe.get("depends_on"):
        lines.append(f"**Depends on:** {', '.join(fe['depends_on'])}")

    if fe.get("would_update"):
        lines.append(f"**Would update:** {', '.join(fe['would_update'])}")

    if fe.get("trigger"):
        lines.append(f"**Trigger condition:** {fe['trigger']}")

    if fe["status"] == "COMPLETED" and fe.get("outcome"):
        lines.append(f"**Outcome:** {fe['outcome']}")

    lines.append("")
    return "\n".join(lines)


def _short_title(description: str, max_chars: int = 90) -> str:
    """Use the first sentence (or up to max_chars) as a section title."""
    first = description.split(". ")[0].strip().rstrip(".")
    if len(first) > max_chars:
        return first[: max_chars - 1] + "..."
    return first


def render_watchlist(papers: list[dict[str, Any]]) -> str:
    lines = [
        "| arxiv_id | title | triggers experiment(s) | status |",
        "|---|---|---|---|",
    ]
    for p in papers:
        triggers = p.get("triggers") or []
        ids = ", ".join(sorted((t["fe_id"] for t in triggers), key=_natkey))
        statuses = ", ".join(sorted({t["status"] for t in triggers}))
        title = p.get("title") or "(untitled)"
        link = _arxiv_link(p["arxiv_id"], p["arxiv_id"])
        lines.append(f"| {link} | {title} | {ids} | {statuses} |")
    return "\n".join(lines)


def generate(experiments: list[dict[str, Any]], papers: list[dict[str, Any]]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    by_tier: dict[str, list[dict[str, Any]]] = {
        "CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": [], "COMPLETED": [],
        "CLOSED": [],
    }
    for fe in experiments:
        if fe["status"] == "COMPLETED":
            by_tier["COMPLETED"].append(fe)
        elif fe["status"] in CLOSED_STATUSES:
            by_tier["CLOSED"].append(fe)
        else:
            by_tier[_tier(fe["roi_score"])].append(fe)

    # Within each tier, sort: highest ROI first, then TRIGGERED before READY before BLOCKED, then id
    for tier, items in by_tier.items():
        items.sort(key=lambda f: (-(f["roi_score"] or 0), STATUS_ORDER.get(f["status"], 99), f["id"]))

    out: list[str] = []
    out.append("# Next Experiments — Priority Queue")
    out.append("")
    out.append("_Auto-generated from the research graph. Do not edit directly._")
    out.append("_Regenerate: `cd research-graph && python generate_next_experiments.py`_")
    out.append(f"_Generated: {now}_")
    out.append("")
    out.append("---")
    out.append("")

    sections = [
        ("CRITICAL", "CRITICAL (ROI 9-10) — Do these first"),
        ("HIGH", "HIGH (ROI 7-8)"),
        ("MEDIUM", "MEDIUM (ROI 5-6)"),
        ("LOW", "LOW (ROI 1-4)"),
    ]

    for key, heading in sections:
        items = by_tier[key]
        out.append(f"## {heading}")
        out.append("")
        if not items:
            out.append("_(none)_")
            out.append("")
        else:
            for fe in items:
                out.append(render_experiment(fe))
        out.append("---")
        out.append("")

    out.append("## Completed")
    out.append("")
    if not by_tier["COMPLETED"]:
        out.append("_(none yet)_")
        out.append("")
    else:
        for fe in by_tier["COMPLETED"]:
            out.append(render_experiment(fe))
    out.append("---")
    out.append("")

    out.append("## Closed by adjacency (MOOTED / ANSWERED)")
    out.append("")
    out.append("Closed without being run: an adjacent experiment refuted the premise")
    out.append("(MOOTED) or already answered the question (ANSWERED). Provenance is on")
    out.append("the MOOTED_BY/ANSWERED_BY edge; resurrect with `update_status.py <id> READY`.")
    out.append("")
    if not by_tier["CLOSED"]:
        out.append("_(none)_")
        out.append("")
    else:
        closed = sorted(by_tier["CLOSED"],
                        key=lambda f: (f.get("completed_date") or "", f["id"]),
                        reverse=True)
        for fe in closed:
            reason = (fe.get("outcome") or "").replace("\n", " ")
            if len(reason) > 140:
                reason = reason[:139] + "…"
            by = fe.get("closed_by") or "?"
            out.append(f"- **{fe['id']}** [{fe['status']} by {by}, "
                       f"{fe.get('completed_date') or '—'}] — {reason}")
        out.append("")
    out.append("---")
    out.append("")

    out.append("## Watchlist — Papers to monitor for new triggers")
    out.append("")
    out.append("These papers are referenced as triggers for future experiments. When a")
    out.append("follow-up appears (or the original methodology gets a public implementation),")
    out.append("check whether any experiment's status should change.")
    out.append("")
    out.append(render_watchlist(papers))
    out.append("")

    return "\n".join(out)


def main() -> None:
    experiments = fetch_all()
    papers = fetch_watchlist()
    md = generate(experiments, papers)
    OUTPUT.write_text(md)
    print(f"Wrote {OUTPUT}")
    print(f"  experiments: {len(experiments)}")
    print(f"  watchlist papers: {len(papers)}")


if __name__ == "__main__":
    main()
