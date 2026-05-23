from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Embeddings — always local (sentence-transformers, free, no API cost)
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-large"

    # Chunking
    child_chunk_size: int = 512
    child_chunk_overlap: int = 64
    parent_chunk_size: int = 2048
    parent_chunk_overlap: int = 128

    # Retrieval
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_graph: int = 10
    top_k_rerank: int = 5

    # Query expansion
    use_multi_query: bool = True
    num_query_variants: int = 3
    use_hyde: bool = True

    # Paths
    data_dir: Path = Path("data")
    uploads_dir: Path = Path("data/uploads")
    faiss_dir: Path = Path("data/faiss_index")
    chunks_dir: Path = Path("data/chunks")
    bm25_dir: Path = Path("data/bm25")
    graph_dir: Path = Path("data/graph")

    # Safeguards
    similarity_threshold: float = 0.30
    rerank_min_score: float = -6.0
    min_chunks_for_answer: int = 1
    dedup_threshold: float = 0.85

    # Security
    enable_prompt_guard: bool = True

    # Access control
    enable_access_control: bool = False
    default_access_level: str = "public"
    default_user_role: str = "admin"

    # PostgreSQL
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "neuroindex"
    db_user: str = "postgres"
    db_password: str = ""

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Observability
    enable_observability: bool = True
    traces_dir: Path = Path("data/traces")
    freshness_weight: float = 0.1

    # Query cache
    enable_query_cache: bool = True
    query_cache_ttl_seconds: int = 300

    # Shard routing — narrow search to docs matching detected intent
    enable_shard_routing: bool = True
    shard_routing_min_docs: int = 2

    class Config:
        env_file = ".env"


settings = Settings()

# Ensure all data directories exist at import time
for _p in [
    settings.uploads_dir,
    settings.faiss_dir,
    settings.chunks_dir,
    settings.bm25_dir,
    settings.graph_dir,
    settings.traces_dir,
]:
    _p.mkdir(parents=True, exist_ok=True)
