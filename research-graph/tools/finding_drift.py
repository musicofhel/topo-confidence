#!/usr/bin/env python3
"""
tool-02-graph-finding-drift-detector.

Compares the Neo4j research-graph's ``(:Finding {id})`` set against the
canonical F-N IDs in ``FINDINGS.md``. Reports:

    1. **Extras** — Findings in the graph with no F-N entry in FINDINGS.md.
       (Drift accumulated through promote/demote cycles without ID-collapse.)
    2. **Missing** — F-N entries in FINDINGS.md with no graph node.
       (Either FINDINGS.md is ahead, or the graph wasn't reseeded.)
    3. **Claim mismatch** — IDs that exist in both but whose claim text
       diverges (heuristic word-overlap below threshold).

Optional ``--cypher`` mode emits a content-aware migration that proposes ID
remappings based on claim-text similarity (operator-reviewed, not auto-applied).

Usage:
    python research-graph/tools/finding_drift.py
    python research-graph/tools/finding_drift.py --verbose
    python research-graph/tools/finding_drift.py --json drift.json
    python research-graph/tools/finding_drift.py --cypher migration.cypher

Exit: 0 if graph and FINDINGS.md align, 1 on any drift detected.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FINDINGS_MD = REPO_ROOT / "FINDINGS.md"

# --- FINDINGS.md parser ---------------------------------------------------

_HEADING_RE = re.compile(r"^### (F-\d+):\s*(.+?)\s*$", re.MULTILINE)
_CLAIM_RE = re.compile(
    r"^\*\*Claim\.\*\*\s*(.+?)(?=^\*\*[A-Z]|^### |\Z)",
    re.MULTILINE | re.DOTALL,
)


@dataclass
class FindingMd:
    id: str
    title: str
    claim: str


def parse_findings_md(path: Path) -> list[FindingMd]:
    """Parse F-N entries out of FINDINGS.md."""
    text = path.read_text(encoding="utf-8")
    out: list[FindingMd] = []
    headings = list(_HEADING_RE.finditer(text))
    for i, m in enumerate(headings):
        body_start = m.end()
        body_end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[body_start:body_end]
        cm = _CLAIM_RE.search(body)
        claim = ""
        if cm:
            claim = " ".join(cm.group(1).split())
        out.append(FindingMd(id=m.group(1), title=m.group(2).strip(), claim=claim))
    return out


# --- Neo4j fetch ----------------------------------------------------------

def _neo4j_creds() -> tuple[str, str, str]:
    bolt = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pw = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")
    return bolt, user, pw


@dataclass
class FindingGraph:
    id: str
    claim: str
    status: str | None
    strength: str | None
    n_evidence: int = 0


def fetch_graph_findings() -> list[FindingGraph]:
    try:
        from neo4j import GraphDatabase  # type: ignore
        from dotenv import load_dotenv  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit(f"missing dep: {e.name} (run pip install -r requirements.txt)")

    load_dotenv(REPO_ROOT / "research-graph" / ".env")
    bolt, user, pw = _neo4j_creds()
    drv = GraphDatabase.driver(bolt, auth=(user, pw), notifications_min_severity="OFF")
    out: list[FindingGraph] = []
    try:
        with drv.session() as s:
            rs = s.run(
                """
                MATCH (f:Finding)
                RETURN f.id AS id, f.claim AS claim, f.status AS status,
                       f.strength AS strength,
                       size(coalesce(f.evidence, [])) AS n_evidence
                ORDER BY f.id
                """
            )
            for r in rs:
                out.append(
                    FindingGraph(
                        id=r["id"],
                        claim=" ".join((r["claim"] or "").split()),
                        status=r["status"],
                        strength=r["strength"],
                        n_evidence=int(r["n_evidence"] or 0),
                    )
                )
    finally:
        drv.close()
    return out


# --- Similarity heuristic -------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]+")
# Common words that don't carry topical signal — drop from overlap calc.
_STOPWORDS = frozenset({
    "the", "and", "for", "but", "with", "from", "this", "that", "have", "has",
    "had", "are", "was", "were", "into", "than", "then", "they", "them",
    "their", "there", "which", "while", "what", "when", "where", "who", "whom",
    "whose", "would", "could", "should", "shall", "will", "may", "might",
    "must", "can", "not", "all", "any", "some", "such", "only", "also",
    "more", "most", "less", "least", "very", "much", "many", "each", "every",
    "one", "two", "three", "first", "second", "third", "is", "be", "been",
    "being", "of", "to", "in", "on", "at", "by", "as", "or", "if", "it",
    "its", "an", "a", "do", "does", "did", "done", "doing", "we", "our",
    "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "his", "him", "she", "her", "hers", "i", "me", "my", "mine",
    "no", "nor", "than", "so", "too",
})


def _toks(s: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((s or "").lower()) if t not in _STOPWORDS and len(t) > 2}


def jaccard(a: str, b: str) -> float:
    ta, tb = _toks(a), _toks(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# --- Drift report ---------------------------------------------------------

@dataclass
class DriftReport:
    graph_ids: list[str] = field(default_factory=list)
    md_ids: list[str] = field(default_factory=list)
    extras: list[dict] = field(default_factory=list)        # in graph, not in MD
    missing: list[dict] = field(default_factory=list)       # in MD, not in graph
    mismatched_claims: list[dict] = field(default_factory=list)  # both, but claim differs
    suggested_remapping: list[dict] = field(default_factory=list)  # extras → MD by content
    aligned: list[str] = field(default_factory=list)        # IDs whose claims overlap


JACCARD_OK = 0.30   # ≥ this = same claim
JACCARD_HINT = 0.05  # ≥ this = candidate remap (titles+claims combined)


def _md_corpus(m: FindingMd) -> str:
    """Title plus claim — title carries the topical keywords most reliably."""
    return f"{m.title} {m.claim}"


def build_report(graph: list[FindingGraph], md: list[FindingMd]) -> DriftReport:
    g_by_id = {f.id: f for f in graph}
    m_by_id = {f.id: f for f in md}
    rep = DriftReport(
        graph_ids=sorted(g_by_id, key=_sort_key),
        md_ids=sorted(m_by_id, key=_sort_key),
    )

    md_id_set = set(m_by_id)
    g_id_set = set(g_by_id)

    # IDs in both → claim alignment check
    for fid in sorted(g_id_set & md_id_set, key=_sort_key):
        g, m = g_by_id[fid], m_by_id[fid]
        sim = jaccard(g.claim, _md_corpus(m))
        if sim >= JACCARD_OK:
            rep.aligned.append(fid)
        else:
            rep.mismatched_claims.append({
                "id": fid,
                "graph_claim": g.claim[:240],
                "md_claim": m.claim[:240],
                "md_title": m.title,
                "jaccard": round(sim, 3),
            })

    # IDs only in graph → suggest remap to MD
    for fid in sorted(g_id_set - md_id_set, key=_sort_key):
        g = g_by_id[fid]
        candidates = []
        for mfid, mf in m_by_id.items():
            sim = jaccard(g.claim, _md_corpus(mf))
            if sim >= JACCARD_HINT:
                candidates.append({"to_id": mfid, "to_title": mf.title, "jaccard": round(sim, 3)})
        candidates.sort(key=lambda c: -c["jaccard"])
        rep.extras.append({
            "id": fid,
            "graph_claim": g.claim[:240],
            "n_evidence": g.n_evidence,
            "best_match": candidates[0] if candidates else None,
            "candidates": candidates[:3],
        })
        if candidates:
            rep.suggested_remapping.append({
                "from_id": fid,
                "to_id": candidates[0]["to_id"],
                "jaccard": candidates[0]["jaccard"],
                "graph_claim": g.claim[:240],
                "md_claim": m_by_id[candidates[0]["to_id"]].claim[:240],
            })

    # IDs only in MD → simply missing
    for fid in sorted(md_id_set - g_id_set, key=_sort_key):
        m = m_by_id[fid]
        rep.missing.append({
            "id": fid,
            "title": m.title,
            "md_claim": m.claim[:240],
        })

    return rep


def _sort_key(s: str) -> tuple[int, int]:
    m = re.match(r"F-(\d+)", s)
    return (0, int(m.group(1))) if m else (1, 0)


# --- Cypher migration emitter --------------------------------------------

def emit_cypher(rep: DriftReport) -> str:
    lines = [
        "// tool-02-graph-finding-drift-detector — proposed migration.",
        "// REVIEW EVERY STATEMENT before executing. Heuristic content-similarity",
        "// only — operator must confirm each remap. Run inside a single",
        "// transaction (`:source` style) so a partial apply rolls back.",
        "//",
        f"// Graph IDs: {rep.graph_ids}",
        f"// MD IDs:    {rep.md_ids}",
        "",
    ]
    if not (rep.extras or rep.missing or rep.mismatched_claims):
        lines.append("// No drift detected — graph and FINDINGS.md align. No migration needed.")
        return "\n".join(lines) + "\n"

    if rep.suggested_remapping:
        lines.append("// --- Proposed ID collapses (extras → FINDINGS.md ground truth) ---")
        # Sort proposals by jaccard descending so highest-confidence remaps are
        # easiest to review first.
        for prop in sorted(rep.suggested_remapping, key=lambda p: -p["jaccard"]):
            lines.append("")
            lines.append(f"// {prop['from_id']} → {prop['to_id']}  (jaccard={prop['jaccard']})")
            lines.append(f"//   graph: {prop['graph_claim']}")
            lines.append(f"//   md:    {prop['md_claim']}")
            lines.append("// MATCH (f:Finding {id: $from_id}) SET f.id = $to_id;")
            lines.append(
                f"// :param from_id => '{prop['from_id']}'; :param to_id => '{prop['to_id']}';"
            )
        lines.append("")

    unmatched = [e for e in rep.extras if not e.get("best_match")]
    if unmatched:
        lines.append("// --- Extras with no FINDINGS.md candidate above threshold ---")
        lines.append("// These need human triage: archive, retitle, or seed a new F-N.")
        for e in unmatched:
            lines.append(f"//   {e['id']}: {e['graph_claim']}")
        lines.append("")

    if rep.missing:
        lines.append("// --- F-N entries in FINDINGS.md missing from the graph ---")
        lines.append("// These need to be seeded via add_finding.py:")
        for m in rep.missing:
            lines.append(f"//   {m['id']}: {m['title']}")
        lines.append("")

    if rep.mismatched_claims:
        lines.append("// --- Same ID, divergent claim text (jaccard < {:.2f}) ---".format(JACCARD_OK))
        lines.append("// Update the graph claim to match FINDINGS.md (or vice versa):")
        for mm in rep.mismatched_claims:
            lines.append("")
            lines.append(f"//   {mm['id']}  jaccard={mm['jaccard']}")
            lines.append(f"//     graph: {mm['graph_claim']}")
            lines.append(f"//     md:    {mm['md_claim']}")
            lines.append(
                f"// MATCH (f:Finding {{id: '{mm['id']}'}}) "
                f"SET f.claim = $md_claim;  // // paste from FINDINGS.md"
            )

    return "\n".join(lines) + "\n"


# --- CLI -----------------------------------------------------------------

def render_terminal(rep: DriftReport, *, verbose: bool) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("tool-02-graph-finding-drift-detector")
    lines.append("=" * 72)
    lines.append(f"Graph (:Finding) IDs : {len(rep.graph_ids):>3}  {rep.graph_ids}")
    lines.append(f"FINDINGS.md F-N IDs  : {len(rep.md_ids):>3}  {rep.md_ids}")
    lines.append("")

    drift_total = len(rep.extras) + len(rep.missing) + len(rep.mismatched_claims)
    if drift_total == 0:
        lines.append("OK — graph and FINDINGS.md align. No drift.")
        lines.append("")
        if verbose:
            lines.append("Aligned IDs (claim-overlap above threshold):")
            for fid in rep.aligned:
                lines.append(f"  {fid}")
        return "\n".join(lines) + "\n"

    if rep.extras:
        lines.append(f"DRIFT: {len(rep.extras)} graph Finding(s) with no FINDINGS.md entry")
        lines.append("-" * 72)
        for e in rep.extras:
            best = e.get("best_match")
            best_s = (
                f"  best match: {best['to_id']} (jaccard={best['jaccard']}) — {best['to_title']}"
                if best else "  best match: <none above threshold — needs human triage>"
            )
            lines.append(f"  {e['id']}  evidence={e['n_evidence']}")
            lines.append(f"    claim: {e['graph_claim'][:120]}")
            lines.append(best_s)
        lines.append("")

    if rep.missing:
        lines.append(f"DRIFT: {len(rep.missing)} FINDINGS.md F-N(s) absent from the graph")
        lines.append("-" * 72)
        for m in rep.missing:
            lines.append(f"  {m['id']}: {m['title']}")
        lines.append("")

    if rep.mismatched_claims:
        lines.append(
            f"DRIFT: {len(rep.mismatched_claims)} ID(s) match but claim text diverges "
            f"(jaccard < {JACCARD_OK:.2f})"
        )
        lines.append("-" * 72)
        for mm in rep.mismatched_claims:
            lines.append(f"  {mm['id']}  jaccard={mm['jaccard']}")
            lines.append(f"    graph: {mm['graph_claim'][:120]}")
            lines.append(f"    md:    {mm['md_claim'][:120]}")
        lines.append("")

    lines.append("Use --cypher <out.cypher> to emit a proposed migration (review before apply).")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verbose", action="store_true", help="also list aligned IDs in clean reports")
    p.add_argument("--json", metavar="PATH", help="write machine-readable JSON report")
    p.add_argument("--cypher", metavar="PATH", help="emit proposed migration Cypher")
    p.add_argument("--findings-md", default=str(FINDINGS_MD), help="path to FINDINGS.md")
    args = p.parse_args(argv)

    md = parse_findings_md(Path(args.findings_md))
    graph = fetch_graph_findings()
    rep = build_report(graph, md)

    print(render_terminal(rep, verbose=args.verbose), end="")

    if args.json:
        Path(args.json).write_text(json.dumps(asdict(rep), indent=2) + "\n", encoding="utf-8")
    if args.cypher:
        Path(args.cypher).write_text(emit_cypher(rep), encoding="utf-8")

    drift_total = len(rep.extras) + len(rep.missing) + len(rep.mismatched_claims)
    return 0 if drift_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
