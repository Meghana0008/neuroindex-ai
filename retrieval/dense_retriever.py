import logging
from typing import List, Dict

from ingestion.embedder import embed_texts
from ingestion.indexer import get_index

logger = logging.getLogger(__name__)


def dense_retrieve(query: str, top_k: int = 20) -> List[Dict]:
    """FAISS inner-product (cosine) search over child chunk embeddings."""
    index = get_index()

    if index.faiss_index is None or index.faiss_index.ntotal == 0:
        return []

    query_emb = embed_texts([query], is_query=True)
    n = min(top_k, index.faiss_index.ntotal)
    distances, indices = index.faiss_index.search(query_emb, n)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(index.child_chunks):
            continue
        chunk = index.child_chunks[idx].copy()
        chunk["score"] = float(dist)
        chunk["retriever"] = "dense"
        results.append(chunk)

    return results
