#!/usr/bin/env python3
"""
tool-05-artifact-metadata-stamp — provenance for hidden-state artifacts.

Goal: every NPZ / JSON carrying experimental results stamps an explicit
provenance block, so silent cross-``seq_len`` mixing (the v1 256-tok / 1024-tok
failure mode) can no longer happen without a loud refusal.

Convention
----------
Every artifact carries a ``meta`` block with:

    {
      "seq_len":   int,         # tokenizer-truncation length (e.g. 1024)
      "tokenizer": str,         # HF model ID for the tokenizer used
      "model_id":  str,         # HF model ID for the model whose hidden states
      "dataset":   str,         # logical dataset name (e.g. "MATH-500")
      "git_sha":   str,         # 7-char commit at write time
      "timestamp": str,         # ISO-8601 UTC at write time
      "script":    str,         # relative path of the producing script
      "schema":    "topo-confidence/v1",
    }

In **NPZ** files the block lives under the reserved key ``__meta__`` and is
JSON-serialized into a 0-d unicode array.

In **JSON** files the block lives under the top-level key ``meta``.

Loaders refuse to compare artifacts with disagreeing ``seq_len`` /
``tokenizer`` / ``model_id`` unless the caller explicitly opts in.

Library API
-----------
    from research_graph.tools.artifact_meta import (
        pack_meta, write_npz, write_json,
        read_meta, assert_compatible,
        CompatibilityError, MissingMetaError,
    )

CLI
---
    python research-graph/tools/artifact_meta.py inspect <path>
    python research-graph/tools/artifact_meta.py scan <dir>            # list missing-meta files
    python research-graph/tools/artifact_meta.py compat <path> <path>  # check pair
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# numpy is heavy and only needed for NPZ paths; load lazily.
_np = None


def _load_numpy():
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


SCHEMA = "topo-confidence/v1"
META_KEY_NPZ = "__meta__"
META_KEY_JSON = "meta"
REQUIRED_FIELDS = ("seq_len", "tokenizer", "model_id", "dataset", "schema")
PROVENANCE_FIELDS = ("git_sha", "timestamp", "script")


class MissingMetaError(RuntimeError):
    """Raised when an artifact has no recognizable meta block."""


class CompatibilityError(RuntimeError):
    """Raised when artifacts disagree on a load-bearing meta field."""


# --- Construction --------------------------------------------------------

def _git_sha(short: int = 7) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", f"--short={short}", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def pack_meta(
    *,
    seq_len: int,
    tokenizer: str,
    model_id: str,
    dataset: str,
    script: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a meta block. ``script`` defaults to argv[0] relativized to cwd."""
    if script is None:
        try:
            script = os.path.relpath(sys.argv[0])
        except ValueError:
            script = sys.argv[0] or "<unknown>"
    meta = {
        "seq_len": int(seq_len),
        "tokenizer": str(tokenizer),
        "model_id": str(model_id),
        "dataset": str(dataset),
        "git_sha": _git_sha(),
        "timestamp": _now_iso(),
        "script": script,
        "schema": SCHEMA,
    }
    if extra:
        # extras are merged, but cannot overwrite the canonical fields.
        for k, v in extra.items():
            if k in meta:
                raise ValueError(f"meta extra cannot overwrite reserved field {k!r}")
            meta[k] = v
    return meta


# --- Writers -------------------------------------------------------------

def write_npz(path: str | Path, meta: dict[str, Any], **arrays) -> None:
    """Write an NPZ with provenance. ``meta`` is JSON-serialized into __meta__."""
    np = _load_numpy()
    if META_KEY_NPZ in arrays:
        raise ValueError(f"{META_KEY_NPZ!r} is reserved; rename your array")
    _validate(meta)
    blob = np.array(json.dumps(meta), dtype=object)
    np.savez_compressed(str(path), **{META_KEY_NPZ: blob}, **arrays)


def write_json(path: str | Path, payload: dict[str, Any], meta: dict[str, Any]) -> None:
    """Write a JSON with provenance. ``meta`` lives under top-level ``meta``."""
    if META_KEY_JSON in payload:
        raise ValueError(f"{META_KEY_JSON!r} is reserved; rename your top-level key")
    _validate(meta)
    out = {META_KEY_JSON: meta, **payload}
    Path(path).write_text(json.dumps(out, indent=2, default=_json_default))


def _json_default(o):
    np = _load_numpy()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"unserializable: {type(o).__name__}")


# --- Readers -------------------------------------------------------------

def read_meta(path: str | Path) -> dict[str, Any]:
    """Read the meta block from an NPZ or JSON. Raises MissingMetaError if absent."""
    p = Path(path)
    if p.suffix == ".npz":
        np = _load_numpy()
        with np.load(p, allow_pickle=True) as z:
            if META_KEY_NPZ not in z.files:
                raise MissingMetaError(f"{p}: no {META_KEY_NPZ} key")
            blob = z[META_KEY_NPZ].item()
            try:
                meta = json.loads(blob)
            except (TypeError, json.JSONDecodeError) as e:
                raise MissingMetaError(f"{p}: {META_KEY_NPZ} is not valid JSON ({e})")
    elif p.suffix == ".json":
        try:
            payload = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise MissingMetaError(f"{p}: not valid JSON ({e})")
        if not isinstance(payload, dict) or META_KEY_JSON not in payload:
            raise MissingMetaError(f"{p}: no top-level {META_KEY_JSON!r} key")
        meta = payload[META_KEY_JSON]
    else:
        raise MissingMetaError(f"{p}: unsupported extension {p.suffix!r}")
    if not isinstance(meta, dict):
        raise MissingMetaError(f"{p}: meta is not a dict")
    return meta


def _validate(meta: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_FIELDS if k not in meta]
    if missing:
        raise ValueError(f"meta missing required fields: {missing}")
    if meta["schema"] != SCHEMA:
        raise ValueError(f"schema mismatch: got {meta['schema']!r}, want {SCHEMA!r}")


# --- Compatibility guard -------------------------------------------------

@dataclass(frozen=True)
class CompatField:
    name: str
    strict: bool  # disagreement always errors when True; warning otherwise


COMPAT_FIELDS: tuple[CompatField, ...] = (
    CompatField("seq_len", True),
    CompatField("tokenizer", True),
    CompatField("model_id", True),
    CompatField("dataset", False),
)


def assert_compatible(
    paths: Iterable[str | Path],
    *,
    require_same_seq_len: bool = True,
    require_same_tokenizer: bool = True,
    require_same_model_id: bool = True,
    require_same_dataset: bool = False,
    allow_mixed: bool = False,
) -> dict[str, list[Any]]:
    """
    Verify all artifacts agree on the load-bearing fields. Returns a summary
    dict mapping field-name → list of distinct values seen (for inspection).

    ``allow_mixed=True`` short-circuits the strict checks (escape hatch for
    explicitly cross-config comparisons; use sparingly).
    """
    paths = list(paths)
    if not paths:
        return {}
    metas = [read_meta(p) for p in paths]
    overrides = {
        "seq_len": require_same_seq_len,
        "tokenizer": require_same_tokenizer,
        "model_id": require_same_model_id,
        "dataset": require_same_dataset,
    }
    distinct: dict[str, list[Any]] = {}
    violations: list[str] = []
    for f in COMPAT_FIELDS:
        seen = []
        for p, m in zip(paths, metas):
            v = m.get(f.name, "<missing>")
            if v not in seen:
                seen.append(v)
        distinct[f.name] = seen
        strict = overrides.get(f.name, f.strict)
        if strict and len(seen) > 1 and not allow_mixed:
            violations.append(
                f"{f.name}: {len(seen)} distinct values across "
                f"{len(paths)} artifacts: {seen!r}"
            )
    if violations:
        msg = "\n  ".join(violations)
        raise CompatibilityError(
            f"artifacts incompatible (pass allow_mixed=True to override):\n  {msg}"
        )
    return distinct


# --- CLI -----------------------------------------------------------------

def _cmd_inspect(args) -> int:
    try:
        meta = read_meta(args.path)
    except MissingMetaError as e:
        print(f"NO META: {e}", file=sys.stderr)
        return 1
    print(json.dumps(meta, indent=2))
    return 0


def _cmd_scan(args) -> int:
    root = Path(args.directory)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    has = []
    missing = []
    for p in sorted(root.rglob("*")):
        if p.suffix not in (".npz", ".json"):
            continue
        # Skip the small auxiliary files the project peppers everywhere.
        if p.name.startswith("validation_report") or p.name == "__meta__.json":
            continue
        try:
            read_meta(p)
            has.append(p)
        except MissingMetaError:
            missing.append(p)
        except Exception as e:  # noqa: BLE001
            missing.append((p, repr(e)))
    print(f"# artifact_meta scan {root}")
    print(f"# {len(has)} stamped, {len(missing)} missing meta")
    if args.show_stamped:
        for p in has:
            print(f"  ok    {p.relative_to(root)}")
    print()
    print("# missing-meta artifacts:")
    for p in missing:
        if isinstance(p, tuple):
            path, err = p
            print(f"  {path.relative_to(root)}\t{err}")
        else:
            print(f"  {p.relative_to(root)}")
    return 0 if not missing else 1


def _cmd_compat(args) -> int:
    try:
        summary = assert_compatible(
            args.paths,
            require_same_seq_len=not args.allow_mixed_seq_len,
            require_same_tokenizer=not args.allow_mixed_tokenizer,
            require_same_model_id=not args.allow_mixed_model,
            require_same_dataset=args.require_same_dataset,
            allow_mixed=args.allow_mixed,
        )
    except CompatibilityError as e:
        print(f"INCOMPATIBLE\n{e}", file=sys.stderr)
        return 1
    except MissingMetaError as e:
        print(f"NO META: {e}", file=sys.stderr)
        return 2
    print("COMPATIBLE")
    for k, vs in summary.items():
        print(f"  {k}: {vs}")
    return 0


def _cmd_stamp_demo(args) -> int:
    """Demo: write a tiny stamped NPZ + JSON pair to show the convention."""
    import numpy as np  # noqa: F401  (ensures lazy import works)
    meta = pack_meta(
        seq_len=args.seq_len, tokenizer=args.tokenizer,
        model_id=args.model_id, dataset=args.dataset,
        script="research-graph/tools/artifact_meta.py demo",
    )
    write_npz(args.out_npz, meta, x=_load_numpy().zeros(3))
    write_json(args.out_json, {"value": 1}, meta)
    print(f"wrote {args.out_npz} and {args.out_json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="Print the meta block of one artifact.")
    pi.add_argument("path")
    pi.set_defaults(func=_cmd_inspect)

    ps = sub.add_parser("scan", help="Walk a directory; report which artifacts are unstamped.")
    ps.add_argument("directory")
    ps.add_argument("--show-stamped", action="store_true",
                    help="Also list artifacts that already carry meta.")
    ps.set_defaults(func=_cmd_scan)

    pc = sub.add_parser("compat", help="Verify multiple artifacts share seq_len / tokenizer / model_id.")
    pc.add_argument("paths", nargs="+")
    pc.add_argument("--allow-mixed", action="store_true",
                    help="Bypass all strictness — for deliberately cross-config comparisons.")
    pc.add_argument("--allow-mixed-seq-len", action="store_true")
    pc.add_argument("--allow-mixed-tokenizer", action="store_true")
    pc.add_argument("--allow-mixed-model", action="store_true")
    pc.add_argument("--require-same-dataset", action="store_true",
                    help="Also strict-check the dataset field.")
    pc.set_defaults(func=_cmd_compat)

    pd = sub.add_parser("demo", help="Write a small stamped NPZ + JSON pair for inspection.")
    pd.add_argument("--out-npz", default="/tmp/artifact_meta_demo.npz")
    pd.add_argument("--out-json", default="/tmp/artifact_meta_demo.json")
    pd.add_argument("--seq-len", type=int, default=1024)
    pd.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    pd.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    pd.add_argument("--dataset", default="MATH-500")
    pd.set_defaults(func=_cmd_stamp_demo)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
