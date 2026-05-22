import json
import logging
from enum import Enum
from typing import Dict, List

from config.settings import settings

logger = logging.getLogger(__name__)

_REGISTRY_PATH = settings.data_dir / "doc_permissions.json"


class AccessLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


# higher number = more sensitive
_LEVEL_RANK: Dict[str, int] = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
}

# each role can see up to this level
_ROLE_MAX_LEVEL: Dict[str, str] = {
    "guest": "public",
    "employee": "internal",
    "manager": "confidential",
    "admin": "restricted",
}


def _load_registry() -> Dict[str, str]:
    if _REGISTRY_PATH.exists():
        with open(_REGISTRY_PATH) as f:
            return json.load(f)
    return {}


def _save_registry(registry: Dict[str, str]):
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def set_document_access(doc_id: str, level: str):
    level = level.lower()
    if level not in _LEVEL_RANK:
        raise ValueError(f"Unknown level '{level}'. Choose from {list(_LEVEL_RANK)}")
    registry = _load_registry()
    registry[doc_id] = level
    _save_registry(registry)
    logger.info(f"Doc {doc_id} access → '{level}'")


def get_document_access(doc_id: str) -> str:
    return _load_registry().get(doc_id, AccessLevel.PUBLIC)


def filter_chunks_by_access(chunks: List[Dict], user_role: str) -> List[Dict]:
    # skip if access control is turned off
    if not settings.enable_access_control:
        return chunks

    role = user_role.lower() if user_role else "guest"
    max_rank = _LEVEL_RANK.get(_ROLE_MAX_LEVEL.get(role, "public"), 0)
    registry = _load_registry()

    allowed = [
        c for c in chunks
        if _LEVEL_RANK.get(registry.get(c.get("doc_id", ""), "public"), 0) <= max_rank
    ]

    removed = len(chunks) - len(allowed)
    if removed:
        logger.info(f"Access control removed {removed} chunks (role='{role}')")

    return allowed


def valid_access_levels() -> List[str]:
    return list(_LEVEL_RANK.keys())


def valid_roles() -> List[str]:
    return list(_ROLE_MAX_LEVEL.keys())
