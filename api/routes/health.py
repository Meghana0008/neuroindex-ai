from fastapi import APIRouter

from api.models import HealthResponse
from ingestion.indexer import get_index

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    index = get_index()
    return HealthResponse(
        status="healthy",
        indexed_chunks=len(index.child_chunks),
        faiss_vectors=index.faiss_index.ntotal if index.faiss_index else 0,
        graph_nodes=index.graph.number_of_nodes() if index.graph else 0,
        graph_edges=index.graph.number_of_edges() if index.graph else 0,
        total_documents=len(index.documents),
    )
