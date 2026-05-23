import logging
from typing import Dict, List, Optional, Set

import numpy as np

from ingestion.indexer import get_index

logger = logging.getLogger(__name__)


def sparse_retrieve(
    query: str,
    top_k: int = 20,
    allowed_doc_ids: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    BM25 keyword retrieval over child chunks.

    allowed_doc_ids: if provided, scores for chunks outside this set are zeroed
    before ranking — equivalent to searching only the allowed shard.
    """
    index = get_index()

    if index.bm25 is None or not index.child_chunks:
        return []

    scores = index.bm25.get_scores(query.lower().split())

    # Zero out scores for disallowed documents before ranking
    if allowed_doc_ids is not None:
        for i, chunk in enumerate(index.child_chunks):
            if chunk.get("doc_id") not in allowed_doc_ids:
                scores[i] = 0.0

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if idx >= len(index.child_chunks) or scores[idx] <= 0:
            continue
        chunk = index.child_chunks[idx].copy()
        chunk["score"] = float(scores[idx])
        chunk["retriever"] = "sparse"
        results.append(chunk)

    return results
