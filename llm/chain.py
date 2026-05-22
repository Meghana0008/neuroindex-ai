import json
import logging
import time
from typing import Any, Dict, Generator, List, Optional

from openai import OpenAI

from access.access_control import filter_chunks_by_access
from config.settings import settings
from ingestion.indexer import get_index
from llm.citations import format_context, parse_citations
from llm.prompts import build_messages
from reranking.reranker import rerank
from retrieval.hybrid_retriever import hybrid_retrieve, reciprocal_rank_fusion
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


def _retrieve(
    query: str,
    use_graph: bool,
    use_hyde: bool,
    use_multi_query: bool,
    user_role: str,
    min_similarity: Optional[float],
    min_rerank_score: Optional[float],
    safeguards: Dict,
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
        res = hybrid_retrieve(variant, use_graph=use_graph)
        res = filter_by_similarity_threshold(res, min_score=min_similarity)
        all_result_lists.append(res)

    if hyde_doc:
        hyde_res = hybrid_retrieve(hyde_doc, use_graph=False)
        hyde_res = filter_by_similarity_threshold(hyde_res, min_score=min_similarity)
        all_result_lists.append(hyde_res)

    all_result_lists = [filter_chunks_by_access(lst, user_role) for lst in all_result_lists]

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
    reranked = validate_reranked_chunks(reranked, min_score=min_rerank_score)
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

    if settings.enable_prompt_guard:
        is_safe, reason = scan_for_injection(query)
        if not is_safe:
            safeguards["prompt_injection_blocked"] = True
            return {
                "answer": f"Query blocked by security filter: {reason}",
                "citations": [], "query_variants": [query],
                "retrieved_chunks": [], "safeguards": safeguards,
                "processing_time": round(time.time() - t0, 2),
            }
    query = sanitize_query(query)

    index = get_index()
    if not index.child_chunks:
        return {
            "answer": "No documents indexed yet. Please upload PDFs first.",
            "citations": [], "query_variants": [query],
            "retrieved_chunks": [], "safeguards": safeguards,
            "processing_time": round(time.time() - t0, 2),
        }

    variants, final_chunks, sufficient, suf_msg = _retrieve(
        query, use_graph, use_hyde, use_multi_query, user_role,
        min_similarity, min_rerank_score, safeguards,
    )

    # if retrieval failed but we have conversation history, answer from history
    if not sufficient and not conversation_history:
        return {
            "answer": suf_msg, "citations": [], "query_variants": variants,
            "retrieved_chunks": [], "safeguards": safeguards,
            "processing_time": round(time.time() - t0, 2),
        }

    context = format_context(final_chunks) if final_chunks else "(No new document context — use conversation history to answer.)"
    messages = build_messages(context=context, question=query, conversation_history=conversation_history, agent_type=agent_type)

    response = _client().chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.2,
        max_tokens=2000,
    )
    answer = response.choices[0].message.content.strip()
    citations = parse_citations(answer, final_chunks)

    return {
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
    }


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

    if settings.enable_prompt_guard:
        is_safe, reason = scan_for_injection(query)
        if not is_safe:
            safeguards["prompt_injection_blocked"] = True
            yield sse({"type": "done", "answer": f"Query blocked: {reason}",
                       "citations": [], "query_variants": [query], "retrieved_chunks": [],
                       "safeguards": safeguards, "processing_time": round(time.time() - t0, 2)})
            yield "data: [DONE]\n\n"
            return

    query = sanitize_query(query)

    index = get_index()
    if not index.child_chunks:
        msg = "No documents indexed yet. Please upload PDFs first."
        yield sse({"type": "done", "answer": msg, "citations": [], "query_variants": [query],
                   "retrieved_chunks": [], "safeguards": safeguards,
                   "processing_time": round(time.time() - t0, 2)})
        yield "data: [DONE]\n\n"
        return

    variants, final_chunks, sufficient, suf_msg = _retrieve(
        query, use_graph, use_hyde, use_multi_query, user_role,
        min_similarity, min_rerank_score, safeguards,
    )

    # if retrieval failed but we have conversation history, answer from history
    if not sufficient and not conversation_history:
        yield sse({"type": "done", "answer": suf_msg, "citations": [], "query_variants": variants,
                   "retrieved_chunks": [], "safeguards": safeguards,
                   "processing_time": round(time.time() - t0, 2)})
        yield "data: [DONE]\n\n"
        return

    context = format_context(final_chunks) if final_chunks else "(No new document context — use conversation history to answer.)"
    messages = build_messages(context=context, question=query, conversation_history=conversation_history, agent_type=agent_type)

    stream = _client().chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.2,
        max_tokens=2000,
        stream=True,
    )

    full_answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_answer += delta
            yield sse({"type": "token", "content": delta})

    citations = parse_citations(full_answer, final_chunks)
    yield sse({
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
    })
    yield "data: [DONE]\n\n"
