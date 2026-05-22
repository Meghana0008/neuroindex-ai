import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)

# all the nasty patterns we want to block
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    # instruction overrides
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?", "instruction override"),
    (r"disregard\s+(all\s+)?(previous|prior|above)\s+", "instruction override"),
    (r"forget\s+(everything|all|your\s+previous|prior)", "instruction override"),
    (r"override\s+(all\s+)?(previous|your|system)\s+", "instruction override"),
    (r"new\s+instructions?\s*:", "instruction override"),
    # trying to read the system prompt
    (r"(reveal|print|show|output|repeat|display)\s+(your\s+)?(system\s+prompt|instructions?|prompt)", "prompt extraction"),
    (r"what\s+(are|were)\s+your\s+(initial\s+|original\s+)?instructions?", "prompt extraction"),
    (r"tell\s+me\s+your\s+(system\s+)?prompt", "prompt extraction"),
    # persona / role hijacks
    (r"you\s+are\s+now\s+(a|an)\s+\w+", "persona hijack"),
    (r"act\s+as\s+(if\s+you(\s+are|\s+were)|a|an)\s+", "persona hijack"),
    (r"pretend\s+(you\s+are|to\s+be)\s+", "persona hijack"),
    (r"roleplay\s+as\s+", "persona hijack"),
    (r"simulate\s+(being|a|an)\s+", "persona hijack"),
    # classic jailbreaks
    (r"\bdan\s+mode\b", "jailbreak"),
    (r"\bjailbreak\b", "jailbreak"),
    (r"\bdo\s+anything\s+now\b", "jailbreak"),
    (r"\bno\s+restrictions?\b", "jailbreak"),
    (r"\bunfiltered\b", "jailbreak"),
    # template / delimiter injection
    (r"<\s*system\s*>", "template injection"),
    (r"\[INST\]", "template injection"),
    (r"###\s*instruction", "template injection"),
    (r"\{\{.*\}\}", "template injection"),
    (r"-----\s*(system|human|assistant)\s*-----", "template injection"),
    # sneaky indirect injection via document content
    (r"the\s+document\s+says\s+ignore", "indirect injection"),
    (r"context\s+says?\s+you\s+should", "indirect injection"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), label) for p, label in _INJECTION_PATTERNS]

_MAX_SPECIAL_RATIO = 0.35
_MAX_QUERY_LENGTH = 2000


def scan_for_injection(query: str) -> Tuple[bool, str]:
    if not query or not query.strip():
        return False, "Empty query."

    if len(query) > _MAX_QUERY_LENGTH:
        return False, f"Query exceeds {_MAX_QUERY_LENGTH} characters."

    for pattern, label in _COMPILED:
        if pattern.search(query):
            logger.warning(f"Injection detected [{label}]: {query[:120]!r}")
            return False, f"Query blocked — detected potential {label} attempt."

    # block queries with suspiciously high special character density
    non_alnum = sum(1 for c in query if not c.isalnum() and not c.isspace())
    if len(query) > 20 and non_alnum / len(query) > _MAX_SPECIAL_RATIO:
        logger.warning(f"High special-char density: {query[:120]!r}")
        return False, "Query blocked — too many special characters."

    return True, "OK"


def sanitize_query(query: str) -> str:
    return " ".join(query.split())
