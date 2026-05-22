import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer = None

# BGE models need a query prefix; MiniLM and other models do not
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_BGE_MODELS = {"bge-large", "bge-base", "bge-small"}


def _is_bge(model_name: str) -> bool:
    return any(tag in model_name.lower() for tag in _BGE_MODELS)


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.local_embedding_model}")
        _model = SentenceTransformer(settings.local_embedding_model)
    return _model


def embed_texts(texts: List[str], is_query: bool = False) -> np.ndarray:
    model = _get_model()
    if is_query and _is_bge(settings.local_embedding_model):
        texts = [_BGE_QUERY_PREFIX + t for t in texts]

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)
