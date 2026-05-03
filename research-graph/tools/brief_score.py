#!/usr/bin/env python3
"""
tool-11-brief-depth-scorer — score the 6 expanded sections of a /paper-triage brief.

Goal: stop ``promote_brief.py`` from promoting briefs whose expanded sections
are filler. The 220-paper deep-pass cycle is only worth the runtime if the
artifacts coming out of it carry actual depth.

Sections scored
---------------
The six expanded sections introduced in the max-depth re-triage plan:

    1. Methodologies extracted
    2. Approaches & framings
    3. Datasets & benchmarks
    4. Implementation details worth capturing
    5. Replicable intermediates
    6. Cross-paper signals

Plus the existing ``Refutations`` section, which the plan tightened from
≥2 to ≥3 bullets minimum.

Heuristic scoring (per section, 0–3)
------------------------------------
  0 — absent OR explicit "N/A" placeholder (legitimate for some papers)
  1 — present but superficial: <80 words, or <2 bullets
  2 — substantive: 80–250 words AND ≥2 bullets
  3 — rich: >250 words, ≥3 bullets, AND mentions cost / code link / impl detail

Promote-block policy
--------------------
A brief blocks promotion if any of:
  - Refutations has fewer than 3 bullets
  - Any of the 6 expanded sections is score 0 *without* an explicit-N/A marker
  - Fewer than 4 of the 6 expanded sections reach score ≥2

CLI
---
    python research-graph/tools/brief_score.py <brief.md>             # score one
    python research-graph/tools/brief_score.py --all-in briefs/       # batch
    python research-graph/tools/brief_score.py --check-promote <b.md> # exit non-zero on block

Designed to be importable from promote_brief.py:

    from research_graph.tools.brief_score import score_brief, ScoreReport
    rep = score_brief(brief_path)
    if not rep.passes_promote_gate:
        ...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --- Configuration --------------------------------------------------------

EXPANDED_SECTIONS: tuple[str, ...] = (
    "Methodologies extracted",
    "Approaches & framings",
    "Datasets & benchmarks",
    "Implementation details worth capturing",
    "Replicable intermediates",
    "Cross-paper signals",
)

REFUTATIONS_HEADER = "Refutations"
MIN_REFUTATIONS = 3
MIN_SUBSTANTIVE_EXPANDED = 4

# Phrases that, when they appear (case-insensitive) in a section body,
# legitimize a score-0 (explicit "this paper has none of this kind"):
NA_MARKERS: tuple[str, ...] = (
    "no genuine refutations identified",
    "no methodologies identified",
    "no approaches identified",
    "no datasets identified",
    "no implementation details",
    "no replicable intermediates",
    "no cross-paper signals identified",
    "n/a — ",
    "n/a -- ",
    "not applicable —",
    "not applicable --",
    "pure theoretical paper",
    "purely theoretical",
)

# Keywords whose presence elevates a borderline section to "rich":
RICH_KEYWORDS = re.compile(
    r"(min cpu|h100|hour|hours|day|days|cost:|replication cost|"
    r"hyperparameter|ablation|epoch|batch size|learning rate|"
    r"github\.com|huggingface\.co|arxiv\.org|"
    r"\bcode\s*(?:link|repo|release)\b)",
    re.IGNORECASE,
)
URL_PAT = re.compile(r"https?://[^\s)\]>]+")


# --- Datatypes -----------------------------------------------------------

@dataclass
class SectionScore:
    name: str
    score: int                # 0..3
    word_count: int
    bullet_count: int
    has_rich_kw: bool
    has_url: bool
    explicit_na: bool
    note: str = ""


@dataclass
class ScoreReport:
    brief_path: str
    sections: dict[str, SectionScore] = field(default_factory=dict)
    refutation_bullets: int = 0
    expanded_substantive: int = 0
    blocks: list[str] = field(default_factory=list)

    @property
    def passes_promote_gate(self) -> bool:
        return not self.blocks


# --- Section parsing -----------------------------------------------------

_HEADER_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _split_sections(text: str) -> dict[str, str]:
    """Return a dict of {header_text: body_text} for every '## ' section."""
    parts: dict[str, str] = {}
    matches = list(_HEADER_RE.finditer(text))
    for i, m in enumerate(matches):
        header = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts[header] = text[start:end].strip()
    return parts


_LETTER_NUM_PREFIX = re.compile(r"^(?:\*\*|__)?[A-Z]\d+\s*[.—–\-]\s+\S")  # "R1 — ...", "**H3.**", "R2."
_H3_HEADING = re.compile(r"^###\s+\S")                       # "### 1. <text>"


def _bullet_count(body: str) -> int:
    """
    Count top-level item markers. Recognizes:
      - ``- ``, ``* ``, ``+ ``    (markdown bullets)
      - ``1. ``, ``2. ``           (markdown ordered lists)
      - ``R1 — ``, ``H3 — ``       (Refutation/Hypothesis numbering style)
      - ``### 1. ...``, ``### Foo``  (h3 subheadings used as item separators)
    """
    count = 0
    for line in body.splitlines():
        s = line.lstrip()
        # Skip indented continuations of bullets.
        if (line.startswith(" " * 4) or line.startswith("\t")) and not s.startswith(("- ", "* ", "+ ")):
            continue
        if s.startswith(("- ", "* ", "+ ")):
            count += 1
            continue
        if re.match(r"^(?:\*\*|__)?\d+\.\s+\S", s):
            count += 1
            continue
        if _LETTER_NUM_PREFIX.match(s):
            count += 1
            continue
        if _H3_HEADING.match(s):
            count += 1
    return count


def _word_count(body: str) -> int:
    # Strip backticked code, URLs, and bullet markers; what's left is prose.
    t = re.sub(r"```.*?```", " ", body, flags=re.S)
    t = re.sub(r"`[^`]*`", " ", t)
    t = URL_PAT.sub(" ", t)
    return len(t.split())


def _score_section(name: str, body: str) -> SectionScore:
    if not body.strip():
        return SectionScore(name=name, score=0, word_count=0, bullet_count=0,
                            has_rich_kw=False, has_url=False, explicit_na=False,
                            note="empty section body")
    lower = body.lower()
    explicit_na = any(marker in lower for marker in NA_MARKERS)
    if explicit_na:
        return SectionScore(name=name, score=0, word_count=_word_count(body),
                            bullet_count=_bullet_count(body),
                            has_rich_kw=False, has_url=False,
                            explicit_na=True, note="explicit N/A marker — legitimate empty")
    wc = _word_count(body)
    bc = _bullet_count(body)
    has_kw = bool(RICH_KEYWORDS.search(body))
    has_url = bool(URL_PAT.search(body))

    if wc < 80 or bc < 2:
        score = 1
        note = "superficial (under 80 words or under 2 bullets)"
    elif wc >= 200 and bc >= 3 and (has_kw or has_url):
        score = 3
        note = "rich (deep, with cost or code link)"
    else:
        score = 2
        note = "substantive"
    return SectionScore(name=name, score=score, word_count=wc, bullet_count=bc,
                        has_rich_kw=has_kw, has_url=has_url,
                        explicit_na=False, note=note)


# --- Scorer --------------------------------------------------------------

def score_brief(brief_path: str | Path) -> ScoreReport:
    p = Path(brief_path)
    text = p.read_text(encoding="utf-8")
    sections = _split_sections(text)
    rep = ScoreReport(brief_path=str(p))

    # Score the 6 expanded sections.
    substantive = 0
    for name in EXPANDED_SECTIONS:
        body = sections.get(name, "")
        s = _score_section(name, body)
        rep.sections[name] = s
        if s.score >= 2:
            substantive += 1
        if name not in sections:
            rep.blocks.append(f"missing section: ## {name}")
        elif s.score == 0 and not s.explicit_na:
            rep.blocks.append(
                f"section '{name}' is score 0 without explicit N/A marker"
            )
    rep.expanded_substantive = substantive
    if substantive < MIN_SUBSTANTIVE_EXPANDED:
        rep.blocks.append(
            f"only {substantive}/6 expanded sections reach score ≥2 "
            f"(need ≥{MIN_SUBSTANTIVE_EXPANDED})"
        )

    # Refutations: count bullets, require ≥ MIN_REFUTATIONS.
    refut_body = sections.get(REFUTATIONS_HEADER, "")
    bn = _bullet_count(refut_body)
    rep.refutation_bullets = bn
    if REFUTATIONS_HEADER not in sections:
        rep.blocks.append(f"missing section: ## {REFUTATIONS_HEADER}")
    elif bn < MIN_REFUTATIONS:
        # Allow the explicit "no genuine refutations identified" escape per plan.
        if "no genuine refutations identified" not in refut_body.lower():
            rep.blocks.append(
                f"Refutations has {bn} bullets, need ≥{MIN_REFUTATIONS} "
                "(or explicit 'no genuine refutations identified' escape)"
            )

    return rep


# --- Pretty-print --------------------------------------------------------

def render_report(rep: ScoreReport) -> str:
    lines = [f"# brief_score: {rep.brief_path}"]
    for name in EXPANDED_SECTIONS:
        s = rep.sections.get(name)
        if s is None:
            lines.append(f"  {name:42s}  : MISSING")
            continue
        flags = []
        if s.has_rich_kw:
            flags.append("kw")
        if s.has_url:
            flags.append("url")
        if s.explicit_na:
            flags.append("na")
        flagstr = ",".join(flags) if flags else "—"
        lines.append(
            f"  {name:42s}  : {s.score}  "
            f"(w={s.word_count:3d} b={s.bullet_count:2d} {flagstr:6s}) "
            f"{s.note}"
        )
    lines.append(f"  {'(Refutations bullets)':42s}  : {rep.refutation_bullets} "
                 f"(need ≥{MIN_REFUTATIONS})")
    lines.append(f"  {'(Expanded sections ≥2)':42s}  : "
                 f"{rep.expanded_substantive}/6 (need ≥{MIN_SUBSTANTIVE_EXPANDED})")
    if rep.blocks:
        lines.append("  PROMOTE-BLOCKED:")
        for b in rep.blocks:
            lines.append(f"    - {b}")
    else:
        lines.append("  PASSES promote gate")
    return "\n".join(lines)


# --- CLI -----------------------------------------------------------------

def _cmd_score_one(brief: str, *, json_out: bool) -> int:
    rep = score_brief(brief)
    if json_out:
        out = {
            "brief_path": rep.brief_path,
            "passes_promote_gate": rep.passes_promote_gate,
            "expanded_substantive": rep.expanded_substantive,
            "refutation_bullets": rep.refutation_bullets,
            "blocks": rep.blocks,
            "sections": {n: asdict(s) for n, s in rep.sections.items()},
        }
        print(json.dumps(out, indent=2))
    else:
        print(render_report(rep))
    return 0 if rep.passes_promote_gate else 1


def _cmd_all_in(directory: str, *, json_out: bool) -> int:
    root = Path(directory)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    briefs = sorted(root.glob("triage-*.md"))
    if not briefs:
        print(f"no triage-*.md briefs in {root}", file=sys.stderr)
        return 2
    fail = 0
    summaries = []
    for b in briefs:
        rep = score_brief(b)
        summaries.append(rep)
        if not rep.passes_promote_gate:
            fail += 1
    if json_out:
        print(json.dumps([{
            "brief_path": r.brief_path,
            "passes_promote_gate": r.passes_promote_gate,
            "expanded_substantive": r.expanded_substantive,
            "refutation_bullets": r.refutation_bullets,
            "blocks": r.blocks,
        } for r in summaries], indent=2))
    else:
        for r in summaries:
            status = "PASS" if r.passes_promote_gate else "BLOCK"
            print(f"{status}  {r.brief_path}  "
                  f"({r.expanded_substantive}/6 ≥2, {r.refutation_bullets} refut)")
            for b in r.blocks:
                print(f"    - {b}")
        print(f"\n# Summary: {len(briefs)} scored, {fail} block promote")
    return 0 if fail == 0 else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("brief", nargs="?", help="Path to a single brief.md to score.")
    p.add_argument("--all-in", metavar="DIR",
                   help="Score every triage-*.md in DIR.")
    p.add_argument("--check-promote", action="store_true",
                   help="(legacy alias for default behavior — exit 1 on promote-block)")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of pretty output.")
    args = p.parse_args(argv)
    if args.all_in:
        return _cmd_all_in(args.all_in, json_out=args.json)
    if not args.brief:
        p.print_usage(sys.stderr)
        return 2
    return _cmd_score_one(args.brief, json_out=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
