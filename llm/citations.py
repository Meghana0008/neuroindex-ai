import re
from typing import Dict, List


def format_context(chunks: List[Dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {chunk['doc_name']} | Page {chunk['page_number']}\n"
            f"{chunk['content']}"
        )
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

        # Find best-matching chunk
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
