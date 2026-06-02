"""JSON KV cache for LLM calls (HyDE, concept expansion, reranking).

Same format as link-forge: {"type:key": "value", ...}

Cache path is in cache/, NOT data/ — data/ is the Neo4j Docker volume mount.
"""
from __future__ import annotations

import json
from pathlib import Path

CACHE_PATH = Path(__file__).parent / "cache" / "llm-cache.json"

_cache: dict[str, str] | None = None


def _load() -> dict[str, str]:
    global _cache
    if _cache is None:
        if CACHE_PATH.exists():
            _cache = json.loads(CACHE_PATH.read_text())
        else:
            _cache = {}
    return _cache


def cache_get(type_: str, key: str) -> str | None:
    return _load().get(f"{type_}:{key}")


def cache_set(type_: str, key: str, value: str) -> None:
    data = _load()
    data[f"{type_}:{key}"] = value
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def cache_size() -> int:
    return len(_load())


def cache_clear() -> None:
    global _cache
    _cache = None
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
