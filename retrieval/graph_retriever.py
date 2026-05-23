import logging
from typing import Dict, List, Optional, Set

import networkx as nx

from graph.entity_extractor import extract_entities
from ingestion.indexer import get_index

logger = logging.getLogger(__name__)


def graph_retrieve(
    query: str,
    top_k: int = 10,
    allowed_doc_ids: Optional[Set[str]] = None,
) -> List[Dict]:
    index = get_index()

    if index.graph is None or not index.child_chunks:
        return []

    query_entities = extract_entities(query)
    if not query_entities:
        return []

    graph_nodes: Set[str] = set(index.graph.nodes())
    matched: List[str] = []

    for ent in query_entities:
        ent_lower = ent.lower()
        if ent_lower in graph_nodes:
            matched.append(ent_lower)
        else:
            for node in graph_nodes:
                if ent_lower in node or node in ent_lower:
                    matched.append(node)

    if not matched:
        return []

    relevant_ids: Set[str] = set()
    for ent in matched:
        node_data = index.graph.nodes.get(ent, {})
        for cid in node_data.get("chunk_ids", []):
            relevant_ids.add(cid)
        try:
            for neighbor in list(nx.neighbors(index.graph, ent))[:5]:
                for cid in index.graph.nodes.get(neighbor, {}).get("chunk_ids", []):
                    relevant_ids.add(cid)
        except Exception:
            pass

    id_to_chunk = {c["chunk_id"]: c for c in index.child_chunks}

    results = []
    for cid in list(relevant_ids)[:top_k * 3]:
        if cid not in id_to_chunk:
            continue
        chunk = id_to_chunk[cid].copy()
        # Pre-filter: skip chunks from disallowed docs
        if allowed_doc_ids is not None and chunk.get("doc_id") not in allowed_doc_ids:
            continue
        chunk["score"] = 1.0
        chunk["retriever"] = "graph"
        results.append(chunk)
        if len(results) >= top_k:
            break

    return results
