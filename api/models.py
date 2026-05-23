from typing import List, Optional

from pydantic import BaseModel


class ConversationMessage(BaseModel):
    role: str
    content: str


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    num_pages: int
    num_child_chunks: int
    num_parent_chunks: int
    access_level: str
    status: str


class QueryRequest(BaseModel):
    query: str
    use_graph: bool = True
    use_hyde: bool = True
    use_multi_query: bool = True
    user_role: str = "admin"
    min_similarity: Optional[float] = None
    min_rerank_score: Optional[float] = None
    conversation_history: List[ConversationMessage] = []
    agent_type: str = "general"


class Citation(BaseModel):
    doc_name: str
    page_number: int
    chunk_content: str
    doc_id: str = ""


class RetrievedChunk(BaseModel):
    doc_name: str
    page_number: int
    rerank_score: float
    content: str


class SafeguardInfo(BaseModel):
    prompt_injection_blocked: bool = False
    chunks_before_dedup: int = 0
    chunks_after_dedup: int = 0
    chunks_after_rerank_filter: int = 0
    retrieval_sufficient: bool = True
    sufficiency_message: str = ""


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    query_variants: List[str]
    retrieved_chunks: List[RetrievedChunk]
    safeguards: SafeguardInfo
    processing_time: float
    confidence_score: float = 0.0
    groundedness_score: float = 0.0
    hallucination_risk: bool = False
    estimated_cost_usd: float = 0.0
    detected_agent_type: str = "general"


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    num_pages: int
    num_child_chunks: int
    access_level: str
    uploaded_at: str


class DeleteResponse(BaseModel):
    doc_id: str
    status: str


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    faiss_vectors: int
    graph_nodes: int
    graph_edges: int
    total_documents: int
