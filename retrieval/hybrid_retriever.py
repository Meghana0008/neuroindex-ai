import logging
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import settings
from retrieval.dense_retriever import dense_retrieve
from retrieval.graph_retriever import graph_retrieve
from retrieval.sparse_retriever import sparse_retrieve

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard RRF constant


def _parse_ts(ts_str: str) -> float:
    try:
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        return datetime.fromisoformat(ts_str).timestamp()
    except Exception:
        return 0.0


def reciprocal_rank_fusion(result_lists: List[List[Dict]]) -> List[Dict]:
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            chunk_map[cid] = chunk

    # Apply freshness bonus — newer docs rank slightly higher for same relevance
    if settings.freshness_weight > 0 and len(rrf_scores) >= 2:
        timestamps: Dict[str, float] = {
            cid: _parse_ts(chunk_map[cid].get("uploaded_at", ""))
            for cid in rrf_scores
        }
        valid_ts = {cid: ts for cid, ts in timestamps.items() if ts > 0}
        if len(valid_ts) >= 2:
            min_ts = min(valid_ts.values())
            max_ts = max(valid_ts.values())
            ts_range = max_ts - min_ts
            if ts_range > 0:
                max_rrf = max(rrf_scores.values())
                for cid, ts in valid_ts.items():
                    freshness_ratio = (ts - min_ts) / ts_range
                    rrf_scores[cid] += settings.freshness_weight * freshness_ratio * max_rrf

    fused = []
    for cid in sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True):
        chunk = chunk_map[cid].copy()
        chunk["rrf_score"] = rrf_scores[cid]
        fused.append(chunk)

    return fused


def hybrid_retrieve(
    query: str,
    top_k_dense: Optional[int] = None,
    top_k_sparse: Optional[int] = None,
    top_k_graph: Optional[int] = None,
    use_graph: bool = True,
) -> List[Dict]:
    top_k_dense = top_k_dense or settings.top_k_dense
    top_k_sparse = top_k_sparse or settings.top_k_sparse
    top_k_graph = top_k_graph or settings.top_k_graph

    dense_res = dense_retrieve(query, top_k=top_k_dense)
    sparse_res = sparse_retrieve(query, top_k=top_k_sparse)
    result_lists = [dense_res, sparse_res]

    if use_graph:
        graph_res = graph_retrieve(query, top_k=top_k_graph)
        if graph_res:
            result_lists.append(graph_res)

    fused = reciprocal_rank_fusion(result_lists)
    logger.debug(
        f"Hybrid: dense={len(dense_res)} sparse={len(sparse_res)} "
        f"graph={len(result_lists[-1]) if use_graph else 0} → fused={len(fused)}"
    )
    return fused
