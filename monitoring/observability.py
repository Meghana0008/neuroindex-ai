import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RequestTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query: str = ""
    agent_type: str = "general"
    detected_intent: str = "general"
    query_variants: List[str] = field(default_factory=list)
    num_chunks_retrieved: int = 0
    num_chunks_final: int = 0
    retrieval_sufficient: bool = True
    rerank_scores: List[float] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    confidence_score: float = 0.0
    groundedness_score: float = 0.0
    hallucination_risk: bool = False
    prompt_injection_blocked: bool = False
    error: Optional[str] = None


def save_trace(trace: RequestTrace, traces_dir: Path) -> None:
    try:
        traces_dir.mkdir(parents=True, exist_ok=True)
        trace_file = traces_dir / "traces.jsonl"
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(trace)) + "\n")
    except Exception as e:
        logger.warning(f"Failed to save trace: {e}")


def load_recent_traces(traces_dir: Path, limit: int = 100) -> List[Dict]:
    try:
        trace_file = traces_dir / "traces.jsonl"
        if not trace_file.exists():
            return []
        lines = trace_file.read_text(encoding="utf-8").strip().split("\n")
        traces = [json.loads(line) for line in lines if line.strip()]
        return traces[-limit:]
    except Exception as e:
        logger.warning(f"Failed to load traces: {e}")
        return []
