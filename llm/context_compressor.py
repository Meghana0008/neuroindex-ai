import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

_COMPRESSION_THRESHOLD = 8
_MAX_RAW_MESSAGES = 4


def compress_history(history: List[Dict]) -> List[Dict]:
    """Summarize older turns when history grows long, keeping recent turns raw."""
    if len(history) <= _COMPRESSION_THRESHOLD:
        return history

    older = history[:-_MAX_RAW_MESSAGES]
    recent = history[-_MAX_RAW_MESSAGES:]

    lines = []
    for msg in older:
        role = "User" if msg["role"] == "user" else "AI"
        content = msg["content"]
        snippet = content[:150] + "…" if len(content) > 150 else content
        lines.append(f"{role}: {snippet}")

    summary = "Conversation summary (earlier turns):\n" + "\n".join(lines)
    logger.debug(f"Compressed {len(older)} history messages into summary block")
    return [{"role": "system", "content": summary}] + recent
