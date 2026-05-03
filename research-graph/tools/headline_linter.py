#!/usr/bin/env python3
"""
tool-01-headline-linter — drift detection across canonical narrative docs.

Source of truth: ``validate_claims.py``'s ``CLAIMS`` list. Every claim there
carries ``(cid, expected, tol)``; the headline linter scans the canonical
narrative docs for backtick-tagged claim IDs ```<cid>```, finds the
nearest numeric value, and flags any that disagree with the registry.

Usage:
    cd ~/topo-confidence
    python research-graph/tools/headline_linter.py            # report only
    python research-graph/tools/headline_linter.py --verbose  # also list clean tags
    python research-graph/tools/headline_linter.py --json out.json  # machine-readable

Exit code: 0 if no drift, 1 if any drift detected.

Briefs / handoff / per-experiment result files are excluded — they are
historical snapshots and may legitimately diverge from the current canonical
values. The linter's beat is the live narrative-doc surface only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from validate_claims import CLAIMS, Claim  # noqa: E402

# --- Configuration --------------------------------------------------------

# The live, canonical narrative-doc surface. Add to this list when a new
# top-level doc is introduced. Briefs/, handoff/, archive/, scratch/ are
# excluded by design.
NARRATIVE_DOCS: list[str] = [
    "SYNTHESIS.md",
    "README.md",
    "QUICKSTART.md",
    "CLAUDE.md",
    "STATE.md",
    "PROJECT_RECORD.md",
    "FINDINGS.md",
    "HYPOTHESES.md",
    "NOVELTY_AUDIT.md",
    "APPLICATIONS.md",
    "PERSPECTIVES.md",
    "RESEARCH_GRAPH.md",
    "PAPER_INDEX.md",
    "DATA_MANIFEST.md",
    "EXPERIMENT_LOG.md",
    "NEXT_EXPERIMENTS.md",
    "docs/index.html",
]

# Window (in chars) around a tag in which we look for a numeric value. ~4
# lines of typical prose / one table row. Wider risks cross-row leakage.
# Bumped from 140 → 220 after observing some cids placed on a separate line
# from the value they reference (e.g. ``Gap **−0.074**. ... (\`pca-pc1resid-ph-gap\`).``
# spans 3 wrapped lines).
WINDOW_CHARS = 220

# Backtick-wrapped lowercase identifier; ` is the backtick.
TAG_RE = re.compile(r"`([a-z0-9][a-z0-9\-_]+[a-z0-9])`")

# Numeric values: signed (ASCII or unicode minus), integer or decimal, optional
# percent sign, with strict word boundaries on both sides so we don't pull
# digits out of identifiers / claim IDs / file paths.
NUMERIC_RE = re.compile(
    r"(?<![A-Za-z0-9_/.])"
    r"([−\-]?\d+(?:\.\d+)?%?)"
    r"(?![A-Za-z0-9_])"
)


@dataclass
class Drift:
    cid: str
    doc: str
    line: int
    expected: float
    tol: float
    nearest_value: float
    nearest_raw: str
    delta: float
    all_in_window: list[str]


@dataclass
class Warning_:
    cid: str
    doc: str
    line: int
    expected: float
    tol: float
    reason: str


# --- Helpers --------------------------------------------------------------

def _parse_value(raw: str) -> float | None:
    """Parse '0.7679' / '71.6%' / '−0.074' / '12' to a float (percent normalized)."""
    s = raw.strip().replace("−", "-")
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    try:
        v = float(s)
    except ValueError:
        return None
    return v / 100.0 if pct else v


def _expected_as_float(c: Claim) -> float | None:
    if isinstance(c.expected, bool):
        return None  # bool is technically int; skip the one bool claim
    if isinstance(c.expected, (int, float)):
        return float(c.expected)
    return None


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _decimal_places(raw: str) -> int:
    """Number of decimal places in a doc-literal numeric token, e.g. '0.690' -> 3."""
    s = raw.replace("−", "-").rstrip("%")
    if "." not in s:
        return 0
    return len(s.split(".", 1)[1])


def _value_matches(v: float, raw: str, exp: float, tol: float, *, fuzzy: bool = False) -> bool:
    """Decide whether a doc-side numeric token matches the registry expected.

    ``fuzzy`` is set when the token was prefixed with ``~`` in the doc
    (e.g. ``~30``), signalling deliberate approximation; in that case we
    relax to ±1 unit at the displayed precision, capped by 5% of |exp|.
    """
    # 1. Strict tolerance match.
    if abs(v - exp) <= tol:
        return True
    # 2. Display-precision match: if the doc's literal token has fewer
    # decimals than `expected`, accept any value that rounds to the same
    # display-precision representation. Example: doc "0.690" vs expected
    # "0.6902" with tol 0.0001 — strict fails, but rounding 0.6902 to 3
    # decimals yields 0.690, so the doc is correct at the precision shown.
    nd = _decimal_places(raw)
    half_unit = 0.5 * (10 ** (-nd))
    if abs(v - round(exp, nd)) <= half_unit + 1e-12:
        return True
    # 3. Percent-vs-fraction unit confusion (e.g. doc "71.6" vs expected
    # 0.716). Only trip when the gap is large enough to suggest a unit
    # mistake, not a small drift.
    if "%" not in raw and abs(v - exp) > 1 and abs(v / 100 - exp) <= max(tol, 0.001):
        return True
    # 4. Fuzzy match (doc said "~30"): one full display-precision unit, or
    # 5% of |exp|, whichever is larger.
    if fuzzy:
        relaxed = max(10 ** (-nd), 0.05 * abs(exp))
        if abs(v - exp) <= relaxed:
            return True
    return False


def _expected_repr(exp: float, tol: float) -> str:
    """Pretty-print expected value with a sensible number of digits."""
    if abs(exp) >= 100 or abs(exp - round(exp)) < 1e-9:
        return f"{exp:g}"
    return f"{exp:.4f}".rstrip("0").rstrip(".")


# --- Core scan ------------------------------------------------------------

def scan_doc(path: Path, idx: dict[str, Claim]) -> tuple[list[Drift], list[Warning_], int]:
    """
    Scan one doc. Return (drifts, warnings, tags_seen).

    A drift is a registered cid with a nearby numeric value that doesn't
    match the registry within tolerance.

    A warning is a registered cid with NO nearby numeric value at all
    (might be a name-only reference, e.g. inside a methods table).
    """
    text = path.read_text(encoding="utf-8")
    drifts: list[Drift] = []
    warnings: list[Warning_] = []
    tags_seen = 0

    for m in TAG_RE.finditer(text):
        cid = m.group(1)
        c = idx.get(cid)
        if c is None:
            # Common non-claim backticks: file paths, function names, etc.
            continue
        tags_seen += 1
        exp = _expected_as_float(c)
        if exp is None:
            continue

        start = max(0, m.start() - WINDOW_CHARS)
        end = min(len(text), m.end() + WINDOW_CHARS)
        window = text[start:end]
        # Mask the tag itself so we don't grab digits out of the cid.
        masked = window.replace(m.group(0), " " * len(m.group(0)))
        cands: list[tuple[float, str, bool]] = []  # (value, raw, fuzzy)
        for nm in NUMERIC_RE.finditer(masked):
            v = _parse_value(nm.group(1))
            if v is None:
                continue
            # Fuzzy when preceded by ~ (deliberate approximation). Look at the
            # original window text at the same offset, not the masked copy.
            tok_start = nm.start()
            fuzzy = tok_start > 0 and window[tok_start - 1] == "~"
            cands.append((v, nm.group(1), fuzzy))

        if not cands:
            warnings.append(Warning_(
                cid=cid, doc=str(path.relative_to(REPO_ROOT)),
                line=_line_of(text, m.start()),
                expected=exp, tol=c.tol,
                reason="cid mentioned but no numeric value within window",
            ))
            continue

        # Match policy:
        #   1. Within registry tolerance: clean match.
        #   2. Within display-precision rounding (e.g. doc shows "0.690"
        #      for registry "0.6902"): clean match.
        #   3. Percent-vs-fraction unit confusion (e.g. doc "71.6" vs
        #      registry "0.716"): clean match.
        #   4. Fuzzy approximation (doc shows "~30" for registry 30.5544):
        #      clean match — author deliberately downgraded precision.
        # Anything else is drift.
        if any(_value_matches(v, raw, exp, c.tol, fuzzy=fz) for v, raw, fz in cands):
            continue

        # Suppress drift when the doc explicitly opts out via an inline
        # comment marker. Useful for "name-only" references (cid mentioned
        # as evidence, no value to drift-check). Convention:
        #   <!-- noclaim:CID -->  (HTML/Markdown)
        # The marker must appear in the same window as the cid.
        if f"noclaim:{cid}" in window or "noclaim:*" in window:
            continue
        nearest = min(cands, key=lambda vr: abs(vr[0] - exp))
        drifts.append(Drift(
            cid=cid, doc=str(path.relative_to(REPO_ROOT)),
            line=_line_of(text, m.start()),
            expected=exp, tol=c.tol,
            nearest_value=nearest[0], nearest_raw=nearest[1],
            delta=nearest[0] - exp,
            all_in_window=[raw for _, raw, _ in cands],
        ))

    return drifts, warnings, tags_seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print clean docs and per-doc tag counts.")
    ap.add_argument("--json", metavar="PATH",
                    help="Write a machine-readable JSON report to PATH.")
    ap.add_argument("--strict-warnings", action="store_true",
                    help="Treat warnings (missing-value cases) as failures.")
    args = ap.parse_args()

    idx = {c.cid: c for c in CLAIMS}

    print(f"# headline_linter — drift detection across canonical narrative docs")
    print(f"# Registry: {len(idx)} claims from validate_claims.py")
    print(f"# Window: ±{WINDOW_CHARS} chars around each backtick-tagged cid")
    print()

    all_drifts: list[Drift] = []
    all_warnings: list[Warning_] = []
    docs_scanned = 0
    docs_missing = 0

    for rel in NARRATIVE_DOCS:
        p = REPO_ROOT / rel
        if not p.exists():
            docs_missing += 1
            print(f"## {rel} — NOT FOUND, skipped")
            continue
        drifts, warnings, tags = scan_doc(p, idx)
        docs_scanned += 1
        all_drifts.extend(drifts)
        all_warnings.extend(warnings)
        if not drifts and not warnings:
            if args.verbose:
                print(f"## {rel} — clean ({tags} registered cids found)")
            continue
        head = f"## {rel} — {len(drifts)} drift, {len(warnings)} warning ({tags} cids)"
        print(head)
        for d in drifts:
            print(
                f"  [drift]   `{d.cid}` at line {d.line}: "
                f"registry expects {_expected_repr(d.expected, d.tol)}±{d.tol:g}, "
                f"nearest in doc is {d.nearest_raw!r} "
                f"(Δ={d.delta:+.4f}); window had: "
                + ", ".join(repr(s) for s in d.all_in_window)
            )
        for w in warnings:
            print(
                f"  [warn]    `{w.cid}` at line {w.line}: {w.reason} "
                f"(expected {_expected_repr(w.expected, w.tol)}±{w.tol:g})"
            )
        print()

    print(f"# Summary: {docs_scanned} docs scanned"
          + (f", {docs_missing} missing" if docs_missing else "")
          + f", {len(all_drifts)} drift, {len(all_warnings)} warnings")

    if args.json:
        report = {
            "registry_size": len(idx),
            "docs_scanned": docs_scanned,
            "docs_missing": docs_missing,
            "drifts": [asdict(d) for d in all_drifts],
            "warnings": [asdict(w) for w in all_warnings],
        }
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"# JSON report written to {args.json}")

    fail = bool(all_drifts) or (args.strict_warnings and bool(all_warnings))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
