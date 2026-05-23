import logging
from typing import Dict, List

from fastapi import APIRouter

from config.settings import settings
from monitoring.observability import load_recent_traces

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/analytics/traces")
def get_traces(limit: int = 50) -> List[Dict]:
    return load_recent_traces(settings.traces_dir, limit=limit)


@router.get("/analytics/summary")
def get_analytics_summary() -> Dict:
    traces = load_recent_traces(settings.traces_dir, limit=500)
    if not traces:
        return {"message": "No traces yet.", "total_requests": 0}

    total = len(traces)
    total_cost = sum(t.get("estimated_cost_usd", 0.0) for t in traces)
    avg_latency = sum(t.get("total_latency_ms", 0.0) for t in traces) / total
    avg_confidence = sum(t.get("confidence_score", 0.0) for t in traces) / total
    hallucination_count = sum(1 for t in traces if t.get("hallucination_risk"))
    retrieval_failures = sum(1 for t in traces if not t.get("retrieval_sufficient", True))
    injection_blocked = sum(1 for t in traces if t.get("prompt_injection_blocked"))

    agent_counts: Dict[str, int] = {}
    for t in traces:
        a = t.get("agent_type", "general")
        agent_counts[a] = agent_counts.get(a, 0) + 1

    return {
        "total_requests": total,
        "total_cost_usd": round(total_cost, 6),
        "avg_latency_ms": round(avg_latency, 1),
        "avg_confidence_score": round(avg_confidence, 3),
        "hallucination_risk_count": hallucination_count,
        "retrieval_failure_count": retrieval_failures,
        "injection_blocked_count": injection_blocked,
        "requests_by_agent": agent_counts,
    }
