import re
from datetime import datetime
from typing import Dict, List


def _format_date(ts_str: str) -> str:
    try:
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        return datetime.fromisoformat(ts_str).strftime("%Y-%m-%d")
    except Exception:
        return ts_str[:10] if len(ts_str) >= 10 else ""


def format_context(chunks: List[Dict]) -> str:
    """
    Build the context block passed to the LLM.

    Each chunk header includes version number and date so the model can:
    - Identify the newest source when multiple docs cover the same topic
    - Explicitly explain policy supersession ("v2 dated 2026-05-23 overrides v1")
    - Ground its answer in a specific, dated document version
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header_fields = [f"Source: {chunk['doc_name']}", f"Page {chunk['page_number']}"]

        version = chunk.get("version")
        if version:
            header_fields.append(f"Version: v{version}")

        uploaded_at = chunk.get("uploaded_at", "")
        if uploaded_at:
            date_str = _format_date(uploaded_at)
            if date_str:
                header_fields.append(f"Date: {date_str}")

        doc_type = chunk.get("doc_type", "")
        if doc_type and doc_type != "general":
            header_fields.append(f"Type: {doc_type.title()}")

        is_active = chunk.get("is_active", True)
        status = "ACTIVE" if is_active else "DEPRECATED"
        header_fields.append(f"Status: {status}")

        parts.append(f"[{i}] {' | '.join(header_fields)}\n{chunk['content']}")

    return "\n\n---\n\n".join(parts)


def parse_citations(answer: str, chunks: List[Dict]) -> List[Dict]:
    pattern = r'\[Doc:\s*([^,\]]+),\s*Page:\s*(\d+)\]'
    matches = re.findall(pattern, answer)

    cited: List[Dict] = []
    seen = set()

    for raw_name, raw_page in matches:
        doc_name = raw_name.strip()
        page_num = int(raw_page.strip())
        key = (doc_name, page_num)

        if key in seen:
            continue
        seen.add(key)

        match = next(
            (
                c for c in chunks
                if c["page_number"] == page_num
                and (c["doc_name"] == doc_name or doc_name in c["doc_name"])
            ),
            None,
        )

        cited.append({
            "doc_name": doc_name,
            "page_number": page_num,
            "chunk_content": match["content"] if match else "",
            "doc_id": match["doc_id"] if match else "",
        })

    return cited
