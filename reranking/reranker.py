import logging
from typing import Dict, List, Optional

from sentence_transformers import CrossEncoder

from config.settings import settings

logger = logging.getLogger(__name__)

_reranker: Optional[CrossEncoder] = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker: {settings.reranker_model}")
        _reranker = CrossEncoder(settings.reranker_model, max_length=512)
    return _reranker


def rerank(query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    if not chunks:
        return []

    reranker = _get_reranker()
    pairs = [(query, c["content"]) for c in chunks]
    scores = reranker.predict(pairs, show_progress_bar=False)

    ranked = sorted(
        [{**c, "rerank_score": float(s)} for c, s in zip(chunks, scores)],
        key=lambda x: x["rerank_score"],
        reverse=True,
    )
    return ranked[:top_k]
