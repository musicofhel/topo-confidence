"""Centralized FE-ID matching with exact-boundary semantics.

Every function returns exact matches only — never substring.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_fe_num(fe_id: str) -> str:
    """Extract and normalize the FE suffix.

    'P11-FE101' -> 'fe101'
    'FE42'      -> 'fe42'
    'fe26841a'  -> 'fe26841a'
    """
    raw = fe_id.split("-")[-1].lower()
    if not raw.startswith("fe"):
        raw = f"fe{raw}"
    return raw


def _num_part(fe_num: str) -> str:
    """Strip the 'fe' prefix: 'fe101' -> '101'."""
    return fe_num[2:] if fe_num.startswith("fe") else fe_num


def find_recompute_script(fe_id: str) -> Path | None:
    """Find the recompute script for an FE using exact filename match only.

    No fallback substring glob — better to return None than the wrong script.
    """
    fe_num = parse_fe_num(fe_id)
    num = _num_part(fe_num)

    # Try recompute_{num}.py (e.g. recompute_101.py)
    for script in REPO_ROOT.glob(f"pathway11_h100/**/recompute_{num}.py"):
        stem_num = script.stem.replace("recompute_", "").lower()
        if stem_num == num or stem_num == fe_num:
            log.debug("Exact match: %s -> %s", fe_id, script)
            return script
        log.critical(
            "BREAKPOINT: script filename mismatch! fe_id=%s expected num=%s "
            "but script stem has %s (path: %s)",
            fe_id, num, stem_num, script,
        )

    # Try recompute_{fe_num}.py (e.g. recompute_fe101.py)
    for script in REPO_ROOT.glob(f"pathway11_h100/**/recompute_{fe_num}.py"):
        log.debug("Exact match (fe-prefixed): %s -> %s", fe_id, script)
        return script

    log.info(
        "No recompute script found for %s (searched recompute_{%s,%s}.py)",
        fe_id, num, fe_num,
    )
    return None


def match_result_dir(fe_id: str, name: str) -> bool:
    """Check if a directory/file name matches an FE ID with word-boundary semantics.

    fe42 matches 'fe42', 'fe42_foo', 'foo_fe42'
    fe42 does NOT match 'fe421', 'fe421_foo', 'fe4200'
    """
    fe_num = parse_fe_num(fe_id)
    num = _num_part(fe_num)
    lower = name.lower()

    if lower == fe_num or lower == num:
        return True
    # Word boundary: fe101 followed by non-alphanumeric or end
    if re.search(rf'(?:^|[_\-]){re.escape(fe_num)}(?:[_\-]|$)', lower):
        return True
    # Bare numeric with underscore boundaries only (avoid "15" matching inside "fe115")
    if re.search(rf'(?:^|_){re.escape(num)}(?:_|$)', lower):
        return True
    return False
