"""
Hot-query cache — avoids re-running retrieval + reranking + LLM for identical queries.

Cache is invalidated when new documents are indexed (call invalidate_all() from indexer).
Only non-conversational queries are cached (no conversation_history).

At scale: replace with Redis + distributed pub/sub for cache invalidation.
"""
import hashlib
import logging
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_cache: Dict[str, Tuple[Any, float]] = {}
_MAX_ENTRIES = 500


def _make_key(query: str, user_role: str, agent_type: str, use_graph: bool) -> str:
    raw = f"{query.lower().strip()}|{user_role}|{agent_type}|{use_graph}"
    return hashlib.md5(raw.encode()).hexdigest()


def cache_get(query: str, user_role: str, agent_type: str, use_graph: bool) -> Optional[Dict]:
    key = _make_key(query, user_role, agent_type, use_graph)
    entry = _cache.get(key)
    if entry is None:
        return None
    result, expires_at = entry
    if time.time() > expires_at:
        del _cache[key]
        return None
    logger.debug(f"Cache hit for query: '{query[:60]}...'")
    return result


def cache_set(
    query: str,
    user_role: str,
    agent_type: str,
    use_graph: bool,
    result: Dict,
    ttl_seconds: int = 300,
) -> None:
    if len(_cache) >= _MAX_ENTRIES:
        # Evict the entry closest to expiry
        oldest = min(_cache, key=lambda k: _cache[k][1])
        del _cache[oldest]
    key = _make_key(query, user_role, agent_type, use_graph)
    _cache[key] = (result, time.time() + ttl_seconds)
    logger.debug(f"Cached result for: '{query[:60]}...' (TTL={ttl_seconds}s)")


def invalidate_all() -> None:
    count = len(_cache)
    _cache.clear()
    if count:
        logger.info(f"Query cache invalidated — cleared {count} entries")


def cache_stats() -> Dict:
    now = time.time()
    live = sum(1 for _, (_, exp) in _cache.items() if exp > now)
    return {"total_entries": len(_cache), "live_entries": live}
