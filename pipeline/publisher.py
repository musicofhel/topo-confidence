"""Optional Redis publisher for node-graph-substrate observability.

Opt-in via PUBLISHER_REDIS_URL env var. When empty, all operations are no-ops
with zero import cost (redis-py imported lazily).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

_client: Any = None


def init_redis_publisher() -> None:
    global _client
    url = os.environ.get("PUBLISHER_REDIS_URL", "")
    if not url:
        return
    try:
        import redis
        _client = redis.Redis.from_url(url, decode_responses=True)
        _client.ping()
        log.info("Publisher connected to %s", url)
    except Exception as e:
        log.warning("Publisher init failed: %s", e)
        _client = None


def publish(stream: str, data: dict[str, Any]) -> None:
    if _client is None:
        return
    try:
        _client.xadd(stream, {"payload": json.dumps(data)}, maxlen=10000, approximate=True)
    except Exception:
        pass


def set_research_field(arxiv_id: str, fields: dict[str, str]) -> None:
    if _client is None:
        return
    key = f"topoconf:research:{arxiv_id}"
    try:
        _client.hset(key, mapping=fields)
        _client.expire(key, 2592000)
    except Exception:
        pass


def close_redis_publisher() -> None:
    global _client
    if _client is None:
        return
    try:
        _client.close()
    except Exception:
        pass
    _client = None
