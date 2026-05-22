import logging
from typing import List, Dict

import numpy as np

from ingestion.indexer import get_index

logger = logging.getLogger(__name__)


def sparse_retrieve(query: str, top_k: int = 20) -> List[Dict]:
    """BM25 keyword retrieval over child chunks."""
    index = get_index()

    if index.bm25 is None or not index.child_chunks:
        return []

    scores = index.bm25.get_scores(query.lower().split())
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
