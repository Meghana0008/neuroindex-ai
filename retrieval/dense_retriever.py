import logging
from typing import Dict, List, Optional, Set

from ingestion.embedder import embed_texts
from ingestion.indexer import get_index

logger = logging.getLogger(__name__)


def dense_retrieve(
    query: str,
    top_k: int = 20,
    allowed_doc_ids: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    FAISS inner-product (cosine) search over child chunk embeddings.

    allowed_doc_ids: if provided, only chunks from these docs are returned.
    We over-fetch to compensate for filtering, proportional to the fraction
    of the index that's allowed. This is pre-retrieval filtering at the
    Python level — at 100M+ scale, use FAISS IDSelectorBatch with IVF indexes.
    """
    index = get_index()

    if index.faiss_index is None or index.faiss_index.ntotal == 0:
        return []

    # Compute how many vectors to fetch to still return top_k after filtering
    if allowed_doc_ids is not None:
        total = len(index.child_chunks)
        allowed_count = sum(1 for c in index.child_chunks if c.get("doc_id") in allowed_doc_ids)
        if allowed_count == 0:
            return []
        ratio = total / max(1, allowed_count)
        fetch_k = min(int(top_k * max(2.0, ratio)), index.faiss_index.ntotal)
    else:
        fetch_k = min(top_k, index.faiss_index.ntotal)

    query_emb = embed_texts([query], is_query=True)
    distances, indices = index.faiss_index.search(query_emb, fetch_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(index.child_chunks):
            continue
        chunk = index.child_chunks[idx].copy()
        # Pre-filter check — skip chunks from disallowed docs
        if allowed_doc_ids is not None and chunk.get("doc_id") not in allowed_doc_ids:
            continue
        chunk["score"] = float(dist)
        chunk["retriever"] = "dense"
        results.append(chunk)
        if len(results) >= top_k:
            break

    return results
