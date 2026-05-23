import logging
from typing import Optional, Set

from config.settings import settings

logger = logging.getLogger(__name__)


def get_allowed_doc_ids(user_role: str) -> Optional[Set[str]]:
    """
    Return the set of doc_ids the user is allowed to see.
    Returns None when access control is off (meaning all docs are allowed).

    This runs BEFORE vector search so we never retrieve and then discard
    restricted documents — the search scope is narrowed upfront.
    """
    if not settings.enable_access_control:
        return None

    from access.access_control import _LEVEL_RANK, _ROLE_MAX_LEVEL, _load_registry
    from ingestion.indexer import get_index

    role = (user_role or "guest").lower()
    max_rank = _LEVEL_RANK.get(_ROLE_MAX_LEVEL.get(role, "public"), 0)
    registry = _load_registry()
    index = get_index()

    allowed = {
        doc_id
        for doc_id in index.documents
        if _LEVEL_RANK.get(registry.get(doc_id, "public"), 0) <= max_rank
    }
    logger.debug(f"Pre-filter: role='{role}' → {len(allowed)}/{len(index.documents)} docs accessible")
    return allowed


def apply_shard_routing(
    allowed_doc_ids: Optional[Set[str]],
    detected_intent: str,
    min_type_docs: int = 2,
) -> Optional[Set[str]]:
    """
    Narrow search scope to docs matching the detected document type.
    Only activates when intent is specific AND enough typed docs exist.

    Example: financial query → only search financial-tagged docs.
    Falls back to full allowed set when no typed docs are found.
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

    # Not enough typed docs — fall back to full allowed set
    return allowed_doc_ids
