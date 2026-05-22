import logging
from typing import Dict, List, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)


def filter_by_similarity_threshold(chunks: List[Dict], min_score: float = None) -> List[Dict]:
    # only filter dense chunks — sparse/graph scores aren't on the same scale
    threshold = min_score if min_score is not None else settings.similarity_threshold
    filtered, dropped = [], 0
    for c in chunks:
        if c.get("retriever") == "dense" and c.get("score", 1.0) < threshold:
            dropped += 1
        else:
            filtered.append(c)
    if dropped:
        logger.debug(f"Similarity filter: dropped {dropped} low-score dense chunks")
    return filtered


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def deduplicate_chunks(chunks: List[Dict], threshold: float = None) -> List[Dict]:
    sim_threshold = threshold if threshold is not None else settings.dedup_threshold
    seen_hashes: set = set()
    seen_token_sets: List[set] = []
    deduped: List[Dict] = []

    for chunk in chunks:
        content = chunk["content"]
        h = hash(content)

        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        tokens = set(content.lower().split())
        if any(_jaccard(tokens, s) >= sim_threshold for s in seen_token_sets):
            continue

        seen_token_sets.append(tokens)
        deduped.append(chunk)

    removed = len(chunks) - len(deduped)
    if removed:
        logger.info(f"Dedup removed {removed} duplicate chunk(s)")
    return deduped


def validate_reranked_chunks(chunks: List[Dict], min_score: float = None) -> List[Dict]:
    # very negative rerank scores = irrelevant, throw them out
    threshold = min_score if min_score is not None else settings.rerank_min_score
    valid = [c for c in chunks if c.get("rerank_score", 0.0) >= threshold]
    dropped = len(chunks) - len(valid)
    if dropped:
        logger.info(f"Rerank filter: dropped {dropped} low-score chunk(s)")
    return valid


def check_retrieval_sufficiency(chunks: List[Dict], min_chunks: int = None) -> Tuple[bool, str]:
    required = min_chunks if min_chunks is not None else settings.min_chunks_for_answer

    if not chunks:
        return False, "No relevant chunks found — documents may not cover this topic."

    if len(chunks) < required:
        return False, f"Only {len(chunks)} chunk(s) found (need at least {required}). Try rephrasing."

    top_score = chunks[0].get("rerank_score", 0.0)
    if top_score < settings.rerank_min_score + 1.0:
        logger.warning(f"Low top rerank score: {top_score:.3f}")

    return True, "Sufficient context retrieved."
