import json
import logging
import time
from typing import Any, Dict, Generator, List, Optional

from openai import OpenAI

from config.settings import settings
from ingestion.indexer import get_index
from llm.citations import format_context, parse_citations
from llm.context_compressor import compress_history
from llm.groundedness import check_groundedness, compute_confidence, detect_hallucination_risk
from llm.prompts import build_messages
from monitoring.cost_tracker import count_tokens_approx, estimate_cost
from monitoring.observability import RequestTrace, save_trace
from reranking.reranker import rerank
from retrieval.hybrid_retriever import hybrid_retrieve, reciprocal_rank_fusion
from retrieval.intent_classifier import classify_intent
from retrieval.pre_filter import apply_shard_routing, get_allowed_doc_ids
from retrieval.query_cache import cache_get, cache_set
from retrieval.query_expander import generate_hyde_document, generate_query_variants
from retrieval.safeguards import (
    check_retrieval_sufficiency,
    deduplicate_chunks,
    filter_by_similarity_threshold,
    validate_reranked_chunks,
)
from security.prompt_guard import sanitize_query, scan_for_injection

logger = logging.getLogger(__name__)


def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def _empty_safeguards() -> Dict:
    return {
        "prompt_injection_blocked": False,
        "chunks_before_dedup": 0,
        "chunks_after_dedup": 0,
        "chunks_after_rerank_filter": 0,
        "retrieval_sufficient": True,
        "sufficiency_message": "",
    }


def _persist_trace(trace: RequestTrace) -> None:
    if not settings.enable_observability:
        return
    try:
        save_trace(trace, settings.traces_dir)
    except Exception as e:
        logger.warning(f"Trace save failed: {e}")


def _retrieve(
    query: str,
    use_graph: bool,
    use_hyde: bool,
    use_multi_query: bool,
    user_role: str,
    min_similarity: Optional[float],
    min_rerank_score: Optional[float],
    safeguards: Dict,
    allowed_doc_ids=None,
):
    """Shared retrieval logic used by both streaming and non-streaming paths."""
    index = get_index()

    variants: List[str] = (
        generate_query_variants(query, n=settings.num_query_variants)
        if use_multi_query else [query]
    )
    hyde_doc: Optional[str] = generate_hyde_document(query) if use_hyde else None

    all_result_lists: List[List[Dict]] = []
    for variant in variants:
        res = hybrid_retrieve(variant, use_graph=use_graph, allowed_doc_ids=allowed_doc_ids)
        res = filter_by_similarity_threshold(res, min_score=min_similarity)
        all_result_lists.append(res)

    if hyde_doc:
        hyde_res = hybrid_retrieve(hyde_doc, use_graph=False, allowed_doc_ids=allowed_doc_ids)
        hyde_res = filter_by_similarity_threshold(hyde_res, min_score=min_similarity)
        all_result_lists.append(hyde_res)

    combined: List[Dict] = []
    seen_ids: set = set()
    for lst in all_result_lists:
        for c in lst:
            if c["chunk_id"] not in seen_ids:
                combined.append(c)
                seen_ids.add(c["chunk_id"])

    before_dedup = len(combined)
    combined = deduplicate_chunks(combined)
    safeguards["chunks_before_dedup"] = before_dedup
    safeguards["chunks_after_dedup"] = len(combined)

    dedup_ids = {c["chunk_id"] for c in combined}
    deduped_lists = [[c for c in lst if c["chunk_id"] in dedup_ids] for lst in all_result_lists]

    fused = reciprocal_rank_fusion(deduped_lists)
    candidates = fused[:50]

    reranked = rerank(query, candidates, top_k=settings.top_k_rerank)
    filtered = validate_reranked_chunks(reranked, min_score=min_rerank_score)
    # always keep at least top 3 chunks so vague queries like "rate this resume" still work
    reranked = filtered if filtered else reranked[:3]
    safeguards["chunks_after_rerank_filter"] = len(reranked)

    sufficient, suf_msg = check_retrieval_sufficiency(reranked)
    safeguards["retrieval_sufficient"] = sufficient
    safeguards["sufficiency_message"] = suf_msg

    final_chunks: List[Dict] = []
    if sufficient:
        for child in reranked:
            parent = index.get_parent_chunk(child)
            final_chunks.append({**parent, "rerank_score": child.get("rerank_score", 0.0)})

    return variants, final_chunks, sufficient, suf_msg


def run_rag_pipeline(
    query: str,
    use_graph: bool = True,
    use_hyde: Optional[bool] = None,
    use_multi_query: Optional[bool] = None,
    user_role: str = None,
    min_similarity: Optional[float] = None,
    min_rerank_score: Optional[float] = None,
    conversation_history: list = None,
    agent_type: str = "general",
) -> Dict[str, Any]:
    t0 = time.time()
    safeguards = _empty_safeguards()

    use_hyde = settings.use_hyde if use_hyde is None else use_hyde
    use_multi_query = settings.use_multi_query if use_multi_query is None else use_multi_query
    user_role = user_role or settings.default_user_role

    detected_agent_type = classify_intent(query)
    effective_agent_type = detected_agent_type if agent_type == "general" else agent_type

    if conversation_history:
        conversation_history = compress_history(conversation_history)

    if settings.enable_prompt_guard:
        is_safe, reason = scan_for_injection(query)
        if not is_safe:
            safeguards["prompt_injection_blocked"] = True
            _persist_trace(RequestTrace(
                query=query, agent_type=agent_type, detected_intent=detected_agent_type,
                prompt_injection_blocked=True,
                total_latency_ms=round((time.time() - t0) * 1000, 1),
            ))
            return {
                "answer": f"Query blocked by security filter: {reason}",
                "citations": [], "query_variants": [query],
                "retrieved_chunks": [], "safeguards": safeguards,
                "processing_time": round(time.time() - t0, 2),
                "confidence_score": 0.0, "groundedness_score": 0.0,
                "hallucination_risk": False, "estimated_cost_usd": 0.0,
                "detected_agent_type": detected_agent_type,
                "cache_hit": False, "pre_filter_doc_count": 0, "shard_routing_applied": False,
            }
    query = sanitize_query(query)

    index = get_index()
    if not index.child_chunks:
        return {
            "answer": "No documents indexed yet. Please upload PDFs first.",
            "citations": [], "query_variants": [query],
            "retrieved_chunks": [], "safeguards": safeguards,
            "processing_time": round(time.time() - t0, 2),
            "confidence_score": 0.0, "groundedness_score": 0.0,
            "hallucination_risk": False, "estimated_cost_usd": 0.0,
            "detected_agent_type": detected_agent_type,
            "cache_hit": False, "pre_filter_doc_count": 0, "shard_routing_applied": False,
        }

    # Pre-retrieval filtering: determine allowed doc scope BEFORE vector search
    allowed_doc_ids = get_allowed_doc_ids(user_role)
    pre_filter_count = len(allowed_doc_ids) if allowed_doc_ids is not None else len(index.documents)

    # Shard routing: narrow further to docs matching detected intent
    shard_routing_applied = False
    if settings.enable_shard_routing:
        routed_ids = apply_shard_routing(
            allowed_doc_ids, detected_agent_type, settings.shard_routing_min_docs
        )
        if routed_ids is not allowed_doc_ids:
            shard_routing_applied = True
            allowed_doc_ids = routed_ids

    # Cache lookup — only for non-conversational queries
    if settings.enable_query_cache and not conversation_history:
        cached = cache_get(query, user_role, effective_agent_type, use_graph)
        if cached:
            _persist_trace(RequestTrace(
                query=query, agent_type=agent_type, detected_intent=detected_agent_type,
                cache_hit=True, pre_filter_doc_count=pre_filter_count,
                shard_routing_applied=shard_routing_applied,
                total_latency_ms=round((time.time() - t0) * 1000, 1),
            ))
            return {**cached, "cache_hit": True, "processing_time": round(time.time() - t0, 2)}

    t_r0 = time.time()
    variants, final_chunks, sufficient, suf_msg = _retrieve(
        query, use_graph, use_hyde, use_multi_query, user_role,
        min_similarity, min_rerank_score, safeguards,
        allowed_doc_ids=allowed_doc_ids,
    )
    t_retrieval_ms = round((time.time() - t_r0) * 1000, 1)

    if not sufficient and not conversation_history:
        _persist_trace(RequestTrace(
            query=query, agent_type=agent_type, detected_intent=detected_agent_type,
            query_variants=variants, retrieval_sufficient=False,
            retrieval_latency_ms=t_retrieval_ms,
            pre_filter_doc_count=pre_filter_count,
            shard_routing_applied=shard_routing_applied,
            total_latency_ms=round((time.time() - t0) * 1000, 1),
        ))
        return {
            "answer": suf_msg, "citations": [], "query_variants": variants,
            "retrieved_chunks": [], "safeguards": safeguards,
            "processing_time": round(time.time() - t0, 2),
            "confidence_score": 0.0, "groundedness_score": 0.0,
            "hallucination_risk": False, "estimated_cost_usd": 0.0,
            "detected_agent_type": detected_agent_type,
            "cache_hit": False, "pre_filter_doc_count": pre_filter_count,
            "shard_routing_applied": shard_routing_applied,
        }

    context = format_context(final_chunks) if final_chunks else "(No new document context — use conversation history to answer.)"
    messages = build_messages(
        context=context, question=query,
        conversation_history=conversation_history,
        agent_type=effective_agent_type,
    )

    t_llm0 = time.time()
    response = _client().chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.2,
        max_tokens=2000,
    )
    t_llm_ms = round((time.time() - t_llm0) * 1000, 1)

    answer = response.choices[0].message.content.strip()

    if response.usage:
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
    else:
        input_tokens = count_tokens_approx(str(messages))
        output_tokens = count_tokens_approx(answer)
    cost = estimate_cost(settings.openai_model, input_tokens, output_tokens)

    citations = parse_citations(answer, final_chunks)
    confidence = compute_confidence(final_chunks, sufficient)
    groundedness = check_groundedness(answer, final_chunks)
    hallucination_risk = detect_hallucination_risk(answer, final_chunks, sufficient)

    _persist_trace(RequestTrace(
        query=query,
        agent_type=agent_type,
        detected_intent=detected_agent_type,
        query_variants=variants,
        num_chunks_retrieved=safeguards["chunks_before_dedup"],
        num_chunks_final=len(final_chunks),
        retrieval_sufficient=sufficient,
        rerank_scores=[c.get("rerank_score", 0.0) for c in final_chunks],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost,
        retrieval_latency_ms=t_retrieval_ms,
        llm_latency_ms=t_llm_ms,
        total_latency_ms=round((time.time() - t0) * 1000, 1),
        confidence_score=confidence,
        groundedness_score=groundedness,
        hallucination_risk=hallucination_risk,
        pre_filter_doc_count=pre_filter_count,
        shard_routing_applied=shard_routing_applied,
    ))

    result = {
        "answer": answer,
        "citations": citations,
        "query_variants": variants,
        "retrieved_chunks": [
            {
                "doc_name": c["doc_name"],
                "page_number": c["page_number"],
                "rerank_score": c.get("rerank_score", 0.0),
                "content": c["content"][:250] + "...",
            }
            for c in final_chunks
        ],
        "safeguards": safeguards,
        "processing_time": round(time.time() - t0, 2),
        "confidence_score": confidence,
        "groundedness_score": groundedness,
        "hallucination_risk": hallucination_risk,
        "estimated_cost_usd": cost,
        "detected_agent_type": detected_agent_type,
        "cache_hit": False,
        "pre_filter_doc_count": pre_filter_count,
        "shard_routing_applied": shard_routing_applied,
    }

    # Store in cache for future identical queries (only non-conversational)
    if settings.enable_query_cache and sufficient and not conversation_history:
        cache_set(
            query, user_role, effective_agent_type, use_graph,
            result, ttl_seconds=settings.query_cache_ttl_seconds,
        )

    return result


def run_rag_pipeline_stream(
    query: str,
    use_graph: bool = True,
    use_hyde: Optional[bool] = None,
    use_multi_query: Optional[bool] = None,
    user_role: str = None,
    min_similarity: Optional[float] = None,
    min_rerank_score: Optional[float] = None,
    conversation_history: list = None,
    agent_type: str = "general",
) -> Generator[str, None, None]:
    """SSE streaming version — yields data: lines, ends with data: [DONE]."""
    t0 = time.time()
    safeguards = _empty_safeguards()

    def sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    use_hyde = settings.use_hyde if use_hyde is None else use_hyde
    use_multi_query = settings.use_multi_query if use_multi_query is None else use_multi_query
    user_role = user_role or settings.default_user_role

    detected_agent_type = classify_intent(query)
    effective_agent_type = detected_agent_type if agent_type == "general" else agent_type

    if conversation_history:
        conversation_history = compress_history(conversation_history)

    if settings.enable_prompt_guard:
        is_safe, reason = scan_for_injection(query)
        if not is_safe:
            safeguards["prompt_injection_blocked"] = True
            _persist_trace(RequestTrace(
                query=query, agent_type=agent_type, detected_intent=detected_agent_type,
                prompt_injection_blocked=True,
                total_latency_ms=round((time.time() - t0) * 1000, 1),
            ))
            yield sse({"type": "done", "answer": f"Query blocked: {reason}",
                       "citations": [], "query_variants": [query], "retrieved_chunks": [],
                       "safeguards": safeguards, "processing_time": round(time.time() - t0, 2),
                       "confidence_score": 0.0, "groundedness_score": 0.0,
                       "hallucination_risk": False, "estimated_cost_usd": 0.0,
                       "detected_agent_type": detected_agent_type,
                       "cache_hit": False, "pre_filter_doc_count": 0,
                       "shard_routing_applied": False})
            yield "data: [DONE]\n\n"
            return

    query = sanitize_query(query)

    index = get_index()
    if not index.child_chunks:
        msg = "No documents indexed yet. Please upload PDFs first."
        yield sse({"type": "done", "answer": msg, "citations": [], "query_variants": [query],
                   "retrieved_chunks": [], "safeguards": safeguards,
                   "processing_time": round(time.time() - t0, 2),
                   "confidence_score": 0.0, "groundedness_score": 0.0,
                   "hallucination_risk": False, "estimated_cost_usd": 0.0,
                   "detected_agent_type": detected_agent_type,
                   "cache_hit": False, "pre_filter_doc_count": 0,
                   "shard_routing_applied": False})
        yield "data: [DONE]\n\n"
        return

    # Pre-retrieval filtering and shard routing
    allowed_doc_ids = get_allowed_doc_ids(user_role)
    pre_filter_count = len(allowed_doc_ids) if allowed_doc_ids is not None else len(index.documents)

    shard_routing_applied = False
    if settings.enable_shard_routing:
        routed_ids = apply_shard_routing(
            allowed_doc_ids, detected_agent_type, settings.shard_routing_min_docs
        )
        if routed_ids is not allowed_doc_ids:
            shard_routing_applied = True
            allowed_doc_ids = routed_ids

    # Cache lookup for non-conversational queries
    if settings.enable_query_cache and not conversation_history:
        cached = cache_get(query, user_role, effective_agent_type, use_graph)
        if cached:
            _persist_trace(RequestTrace(
                query=query, agent_type=agent_type, detected_intent=detected_agent_type,
                cache_hit=True, pre_filter_doc_count=pre_filter_count,
                shard_routing_applied=shard_routing_applied,
                total_latency_ms=round((time.time() - t0) * 1000, 1),
            ))
            yield sse({**cached, "type": "done", "cache_hit": True,
                       "processing_time": round(time.time() - t0, 2)})
            yield "data: [DONE]\n\n"
            return

    t_r0 = time.time()
    variants, final_chunks, sufficient, suf_msg = _retrieve(
        query, use_graph, use_hyde, use_multi_query, user_role,
        min_similarity, min_rerank_score, safeguards,
        allowed_doc_ids=allowed_doc_ids,
    )
    t_retrieval_ms = round((time.time() - t_r0) * 1000, 1)

    if not sufficient and not conversation_history:
        _persist_trace(RequestTrace(
            query=query, agent_type=agent_type, detected_intent=detected_agent_type,
            query_variants=variants, retrieval_sufficient=False,
            retrieval_latency_ms=t_retrieval_ms,
            pre_filter_doc_count=pre_filter_count,
            shard_routing_applied=shard_routing_applied,
            total_latency_ms=round((time.time() - t0) * 1000, 1),
        ))
        yield sse({"type": "done", "answer": suf_msg, "citations": [], "query_variants": variants,
                   "retrieved_chunks": [], "safeguards": safeguards,
                   "processing_time": round(time.time() - t0, 2),
                   "confidence_score": 0.0, "groundedness_score": 0.0,
                   "hallucination_risk": False, "estimated_cost_usd": 0.0,
                   "detected_agent_type": detected_agent_type,
                   "cache_hit": False, "pre_filter_doc_count": pre_filter_count,
                   "shard_routing_applied": shard_routing_applied})
        yield "data: [DONE]\n\n"
        return

    context = format_context(final_chunks) if final_chunks else "(No new document context — use conversation history to answer.)"
    messages = build_messages(
        context=context, question=query,
        conversation_history=conversation_history,
        agent_type=effective_agent_type,
    )

    t_llm0 = time.time()
    stream = _client().chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.2,
        max_tokens=2000,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_answer = ""
    input_tokens = 0
    output_tokens = 0
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            full_answer += delta
            yield sse({"type": "token", "content": delta})
        if hasattr(chunk, "usage") and chunk.usage:
            input_tokens = chunk.usage.prompt_tokens or 0
            output_tokens = chunk.usage.completion_tokens or 0

    t_llm_ms = round((time.time() - t_llm0) * 1000, 1)

    if not input_tokens:
        input_tokens = count_tokens_approx(str(messages))
        output_tokens = count_tokens_approx(full_answer)
    cost = estimate_cost(settings.openai_model, input_tokens, output_tokens)

    citations = parse_citations(full_answer, final_chunks)
    confidence = compute_confidence(final_chunks, sufficient)
    groundedness = check_groundedness(full_answer, final_chunks)
    hallucination_risk = detect_hallucination_risk(full_answer, final_chunks, sufficient)

    _persist_trace(RequestTrace(
        query=query,
        agent_type=agent_type,
        detected_intent=detected_agent_type,
        query_variants=variants,
        num_chunks_retrieved=safeguards["chunks_before_dedup"],
        num_chunks_final=len(final_chunks),
        retrieval_sufficient=sufficient,
        rerank_scores=[c.get("rerank_score", 0.0) for c in final_chunks],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost,
        retrieval_latency_ms=t_retrieval_ms,
        llm_latency_ms=t_llm_ms,
        total_latency_ms=round((time.time() - t0) * 1000, 1),
        confidence_score=confidence,
        groundedness_score=groundedness,
        hallucination_risk=hallucination_risk,
        pre_filter_doc_count=pre_filter_count,
        shard_routing_applied=shard_routing_applied,
    ))

    done_payload = {
        "type": "done",
        "answer": full_answer,
        "citations": citations,
        "query_variants": variants,
        "retrieved_chunks": [
            {
                "doc_name": c["doc_name"],
                "page_number": c["page_number"],
                "rerank_score": c.get("rerank_score", 0.0),
                "content": c["content"][:250] + "...",
            }
            for c in final_chunks
        ],
        "safeguards": safeguards,
        "processing_time": round(time.time() - t0, 2),
        "confidence_score": confidence,
        "groundedness_score": groundedness,
        "hallucination_risk": hallucination_risk,
        "estimated_cost_usd": cost,
        "detected_agent_type": detected_agent_type,
        "cache_hit": False,
        "pre_filter_doc_count": pre_filter_count,
        "shard_routing_applied": shard_routing_applied,
    }

    # Cache the result for future identical non-conversational queries
    if settings.enable_query_cache and sufficient and not conversation_history:
        cache_set(
            query, user_role, effective_agent_type, use_graph,
            done_payload, ttl_seconds=settings.query_cache_ttl_seconds,
        )

    yield sse(done_payload)
    yield "data: [DONE]\n\n"
