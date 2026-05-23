import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.models import Citation, QueryRequest, QueryResponse, RetrievedChunk, SafeguardInfo
from llm.chain import run_rag_pipeline, run_rag_pipeline_stream

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = run_rag_pipeline(
            query=request.query,
            use_graph=request.use_graph,
            use_hyde=request.use_hyde,
            use_multi_query=request.use_multi_query,
            user_role=request.user_role,
            min_similarity=request.min_similarity,
            min_rerank_score=request.min_rerank_score,
            conversation_history=[m.dict() for m in request.conversation_history],
            agent_type=request.agent_type,
        )
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
        query_variants=result["query_variants"],
        retrieved_chunks=[RetrievedChunk(**c) for c in result["retrieved_chunks"]],
        safeguards=SafeguardInfo(**result["safeguards"]),
        processing_time=result["processing_time"],
        confidence_score=result.get("confidence_score", 0.0),
        groundedness_score=result.get("groundedness_score", 0.0),
        hallucination_risk=result.get("hallucination_risk", False),
        estimated_cost_usd=result.get("estimated_cost_usd", 0.0),
        detected_agent_type=result.get("detected_agent_type", "general"),
    )


@router.post("/query/stream")
async def query_documents_stream(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    def generate():
        try:
            yield from run_rag_pipeline_stream(
                query=request.query,
                use_graph=request.use_graph,
                use_hyde=request.use_hyde,
                use_multi_query=request.use_multi_query,
                user_role=request.user_role,
                min_similarity=request.min_similarity,
                min_rerank_score=request.min_rerank_score,
                conversation_history=[m.dict() for m in request.conversation_history],
                agent_type=request.agent_type,
            )
        except Exception as e:
            logger.error(f"Stream error: {e}")
            import json
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
