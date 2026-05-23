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


def get_allowed_doc_ids(user_role: str, tenant_id: str = "default") -> Optional[Set[str]]:
    """
    Return the set of doc_ids the user may search.

    Three-stage filter (applied in order):
      1. Tenant isolation — only docs belonging to tenant_id
      2. Active-only — deprecated versions excluded
      3. Access control — role-based RBAC (when enabled)
    """
    from ingestion.indexer import get_index
    index = get_index()

    # Stage 1: tenant isolation
    tenant_ids = {
        doc_id
        for doc_id, meta in index.documents.items()
        if meta.get("tenant_id", "default") == tenant_id
    }
    logger.debug(f"Tenant '{tenant_id}': {len(tenant_ids)} docs")

    # Stage 2: active-only within tenant
    active_ids = _get_active_doc_ids()
    allowed = tenant_ids & active_ids

    if not settings.enable_access_control:
        return allowed if allowed != set(index.documents.keys()) else None

    # Stage 3: RBAC
    from access.access_control import _LEVEL_RANK, _ROLE_MAX_LEVEL, _load_registry
    role = (user_role or "guest").lower()
    max_rank = _LEVEL_RANK.get(_ROLE_MAX_LEVEL.get(role, "public"), 0)
    registry = _load_registry()

    access_allowed = {
        doc_id
        for doc_id in allowed
        if _LEVEL_RANK.get(registry.get(doc_id, "public"), 0) <= max_rank
    }

    logger.debug(
        f"Pre-filter: tenant='{tenant_id}' role='{role}' → {len(access_allowed)} docs"
    )
    return access_allowed


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
