import re
from typing import Dict, List


def compute_confidence(chunks: List[Dict], sufficient: bool) -> float:
    if not sufficient or not chunks:
        return 0.0
    scores = [c.get("rerank_score", 0.0) for c in chunks]
    # Rerank scores typically range -10 to +5; map to 0-1
    normalized = [(min(5.0, max(-10.0, s)) + 10.0) / 15.0 for s in scores]
    avg = sum(normalized) / len(normalized)
    chunk_bonus = min(0.2, len(chunks) * 0.04)
    return round(min(1.0, avg + chunk_bonus), 3)


def check_groundedness(answer: str, chunks: List[Dict]) -> float:
    if not chunks or not answer:
        return 0.0
    sentences = re.split(r"[.!?]+", answer)
    meaningful = [s for s in sentences if len(s.strip()) > 10]
    cited = sum(1 for s in meaningful if "[Doc:" in s)
    citation_ratio = cited / max(1, len(meaningful))

    chunk_words: set = set()
    for chunk in chunks[:3]:
        words = set(chunk["content"].lower().split())
        chunk_words.update(w for w in words if len(w) > 5)
    answer_words = set(answer.lower().split())
    overlap_ratio = min(1.0, len(chunk_words & answer_words) / max(1, len(chunk_words)) * 3)

    return round(citation_ratio * 0.6 + overlap_ratio * 0.4, 3)


def detect_hallucination_risk(
    answer: str, chunks: List[Dict], sufficient: bool
) -> bool:
    # Answered despite insufficient retrieval
    if not sufficient and len(answer) > 100:
        return True
    # Long answer with decent source chunks but zero citations
    if chunks and "[Doc:" not in answer and len(answer) > 500:
        top_score = max((c.get("rerank_score", -5.0) for c in chunks), default=-5.0)
        if top_score > -3.0:
            return True
    return False
