"""
Pre-retrieval filtering — narrows the search scope BEFORE vector search runs.

The two-stage filter:
  1. Active-only filter  — removes deprecated document versions
  2. Access-control filter — removes documents the user's role cannot see

Both run before FAISS / BM25 / graph retrieval.
Non-permitted and deprecated documents are never scored.

At scale: these filters become partition keys in a distributed vector store
(Pinecone namespaces, Qdrant payload filters, Weaviate where clauses).
"""
import logging
from typing import Optional, Set

from config.settings import settings

logger = logging.getLogger(__name__)


def _get_active_doc_ids() -> Set[str]:
    """Return doc_ids for currently active (non-deprecated) documents."""
    from ingestion.indexer import get_index
    index = get_index()
    active = {
        doc_id
        for doc_id, meta in index.documents.items()
        if meta.get("is_active", True)
    }
    logger.debug(f"Active docs: {len(active)}/{len(index.documents)}")
    return active


def get_allowed_doc_ids(user_role: str) -> Optional[Set[str]]:
    """
    Return the set of doc_ids the user may search.

    Combines two filters:
      - Active-only: deprecated versions are excluded from retrieval
      - Access control: role-based permission filter (when enabled)

    Returns None only when access control is OFF and all docs are active
    (which means 'search everything' — no filter needed).
    """
    from ingestion.indexer import get_index
    index = get_index()

    # Always filter out deprecated documents
    active_ids = _get_active_doc_ids()

    if not settings.enable_access_control:
        # No role filter — but still exclude deprecated docs
        if len(active_ids) == len(index.documents):
            return None  # All docs are active, no filter needed
        return active_ids

    from access.access_control import _LEVEL_RANK, _ROLE_MAX_LEVEL, _load_registry
    role = (user_role or "guest").lower()
    max_rank = _LEVEL_RANK.get(_ROLE_MAX_LEVEL.get(role, "public"), 0)
    registry = _load_registry()

    access_allowed = {
        doc_id
        for doc_id in index.documents
        if _LEVEL_RANK.get(registry.get(doc_id, "public"), 0) <= max_rank
    }

    # Intersection: must be both active AND permitted
    allowed = active_ids & access_allowed
    logger.debug(
        f"Pre-filter: role='{role}' → {len(allowed)} docs "
        f"(active={len(active_ids)}, access_allowed={len(access_allowed)})"
    )
    return allowed


def apply_shard_routing(
    allowed_doc_ids: Optional[Set[str]],
    detected_intent: str,
    min_type_docs: int = 2,
) -> Optional[Set[str]]:
    """
    Narrow search scope to docs matching the detected document type.
    Only activates when intent is specific AND enough typed docs exist.

    Falls back to the full allowed set when no typed docs are found.
    """
    if detected_intent == "general":
        return allowed_doc_ids

    from ingestion.indexer import get_index
    index = get_index()

    typed_ids = {
        doc_id
        for doc_id, meta in index.documents.items()
        if meta.get("doc_type") == detected_intent
    }

    if allowed_doc_ids is not None:
        typed_ids &= allowed_doc_ids

    if len(typed_ids) >= min_type_docs:
        logger.debug(
            f"Shard routing: intent='{detected_intent}' → "
            f"{len(typed_ids)} docs (was {len(allowed_doc_ids) if allowed_doc_ids else 'all'})"
        )
        return typed_ids

    return allowed_doc_ids
