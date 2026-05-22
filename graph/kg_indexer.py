import logging
from typing import Dict, List

import networkx as nx

logger = logging.getLogger(__name__)


def build_knowledge_graph(
    chunks: List[Dict],
    entities_per_chunk: List[List[str]],
) -> nx.Graph:
    G = nx.Graph()

    for chunk, entities in zip(chunks, entities_per_chunk):
        cid = chunk["chunk_id"]
        unique_ents = list(dict.fromkeys(entities))  # deduplicate, keep order

        for ent in unique_ents:
            if not G.has_node(ent):
                G.add_node(ent, chunk_ids=[cid], frequency=1)
            else:
                G.nodes[ent]["chunk_ids"].append(cid)
                G.nodes[ent]["frequency"] += 1

        for i, e1 in enumerate(unique_ents):
            for e2 in unique_ents[i + 1:]:
                if e1 == e2:
                    continue
                if G.has_edge(e1, e2):
                    G[e1][e2]["weight"] += 1
                else:
                    G.add_edge(e1, e2, weight=1)

    logger.info(
        f"Knowledge graph built: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G
