import json
import logging
import pickle
from datetime import datetime, timezone
from typing import Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from config.settings import settings
from ingestion.embedder import embed_texts

logger = logging.getLogger(__name__)

_DOC_REGISTRY_PATH = settings.data_dir / "documents.json"
_EMBEDDINGS_PATH = settings.faiss_dir / "embeddings.npy"


class DocumentIndex:
    def __init__(self):
        self.child_chunks: List[Dict] = []
        self.parent_chunks: Dict[str, Dict] = {}  # chunk_id → chunk dict
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.embeddings: Optional[np.ndarray] = None  # mirrors FAISS order
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus: List[List[str]] = []
        self.graph = None
        self.documents: Dict[str, Dict] = {}  # doc_id → metadata
        self._load()

    def _load(self):
        child_path = settings.chunks_dir / "child_chunks.pkl"
        parent_path = settings.chunks_dir / "parent_chunks.pkl"
        faiss_path = settings.faiss_dir / "index.faiss"
        emb_path = _EMBEDDINGS_PATH
        bm25_path = settings.bm25_dir / "bm25.pkl"
        graph_path = settings.graph_dir / "kg.pkl"

        if child_path.exists():
            with open(child_path, "rb") as f:
                self.child_chunks = pickle.load(f)
            logger.info(f"Loaded {len(self.child_chunks)} child chunks")

        if parent_path.exists():
            with open(parent_path, "rb") as f:
                parent_list = pickle.load(f)
            self.parent_chunks = {c["chunk_id"]: c for c in parent_list}
            logger.info(f"Loaded {len(self.parent_chunks)} parent chunks")

        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
            logger.info(f"Loaded FAISS index: {self.faiss_index.ntotal} vectors")

        if emb_path.exists():
            self.embeddings = np.load(str(emb_path))

        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.bm25_corpus = data["corpus"]
            logger.info(f"Loaded BM25 index: {len(self.bm25_corpus)} docs")

        if graph_path.exists():
            with open(graph_path, "rb") as f:
                self.graph = pickle.load(f)
            logger.info(
                f"Loaded KG: {self.graph.number_of_nodes()} nodes, "
                f"{self.graph.number_of_edges()} edges"
            )

        if _DOC_REGISTRY_PATH.exists():
            with open(_DOC_REGISTRY_PATH) as f:
                self.documents = json.load(f)
            logger.info(f"Loaded document registry: {len(self.documents)} docs")

    def _save(self):
        with open(settings.chunks_dir / "child_chunks.pkl", "wb") as f:
            pickle.dump(self.child_chunks, f)

        with open(settings.chunks_dir / "parent_chunks.pkl", "wb") as f:
            pickle.dump(list(self.parent_chunks.values()), f)

        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(settings.faiss_dir / "index.faiss"))

        if self.embeddings is not None:
            np.save(str(_EMBEDDINGS_PATH), self.embeddings)

        if self.bm25 is not None:
            with open(settings.bm25_dir / "bm25.pkl", "wb") as f:
                pickle.dump({"bm25": self.bm25, "corpus": self.bm25_corpus}, f)

        if self.graph is not None:
            with open(settings.graph_dir / "kg.pkl", "wb") as f:
                pickle.dump(self.graph, f)

        _DOC_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_DOC_REGISTRY_PATH, "w") as f:
            json.dump(self.documents, f, indent=2)

        # sync document registry to PostgreSQL
        try:
            from db.database import insert_document
            for doc in self.documents.values():
                insert_document(
                    doc_id=doc["doc_id"],
                    filename=doc["filename"],
                    num_pages=doc.get("num_pages", 0),
                    num_chunks=doc.get("num_child_chunks", 0),
                    access_level=doc.get("access_level", "public"),
                )
        except Exception as e:
            logger.warning(f"PostgreSQL sync skipped: {e}")

    def add_chunks(
        self,
        child_chunks: List[Dict],
        parent_chunks: List[Dict],
        doc_metadata: Optional[Dict] = None,
    ):
        if not child_chunks:
            return

        self.child_chunks.extend(child_chunks)
        for pc in parent_chunks:
            self.parent_chunks[pc["chunk_id"]] = pc

        # Stamp each chunk with indexing time and doc_type for freshness + shard routing
        uploaded_at = datetime.now(timezone.utc).isoformat()
        doc_type = doc_metadata.get("doc_type", "general") if doc_metadata else "general"
        for c in child_chunks:
            c["uploaded_at"] = uploaded_at
            c["doc_type"] = doc_type

        # Invalidate hot-query cache — new doc means cached answers may be stale
        try:
            from retrieval.query_cache import invalidate_all
            invalidate_all()
        except Exception:
            pass

        # Dense: embed and store — avoids re-embedding on document deletion
        texts = [c["content"] for c in child_chunks]
        new_embs = embed_texts(texts, is_query=False)
        dim = new_embs.shape[1]

        self.embeddings = (
            np.vstack([self.embeddings, new_embs])
            if self.embeddings is not None
            else new_embs
        )
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(new_embs)

        # Sparse: rebuild BM25
        self.bm25_corpus.extend([t.lower().split() for t in texts])
        self.bm25 = BM25Okapi(self.bm25_corpus)

        # Graph: rebuild from all child chunks
        from graph.entity_extractor import extract_entities
        from graph.kg_indexer import build_knowledge_graph

        all_texts = [c["content"] for c in self.child_chunks]
        entities_per_chunk = [extract_entities(t) for t in all_texts]
        self.graph = build_knowledge_graph(self.child_chunks, entities_per_chunk)

        # Document registry
        if doc_metadata:
            doc_id = doc_metadata["doc_id"]
            self.documents[doc_id] = {
                **doc_metadata,
                "uploaded_at": uploaded_at,
                "is_active": True,
                "version": doc_metadata.get("version", 1),
            }

        self._save()

        # Write chunks to PostgreSQL
        try:
            from db.database import insert_chunks
            insert_chunks(child_chunks + parent_chunks)
        except Exception as e:
            logger.warning(f"PostgreSQL chunk insert skipped: {e}")
        logger.info(
            f"Index updated — total child chunks: {len(self.child_chunks)}, "
            f"FAISS vectors: {self.faiss_index.ntotal}"
        )

    def remove_document(self, doc_id: str) -> bool:
        if not any(c["doc_id"] == doc_id for c in self.child_chunks):
            return False

        keep_mask = [c["doc_id"] != doc_id for c in self.child_chunks]
        keep_indices = [i for i, keep in enumerate(keep_mask) if keep]

        # Filter child chunks and their embeddings
        self.child_chunks = [c for c, k in zip(self.child_chunks, keep_mask) if k]

        if self.embeddings is not None and len(keep_indices) > 0:
            self.embeddings = self.embeddings[keep_indices]
        elif self.embeddings is not None:
            self.embeddings = None

        # Remove orphaned parent chunks
        live_parent_ids = {c["parent_id"] for c in self.child_chunks if c.get("parent_id")}
        self.parent_chunks = {pid: pc for pid, pc in self.parent_chunks.items() if pid in live_parent_ids}

        # Rebuild FAISS from stored embeddings
        if self.embeddings is not None and len(self.embeddings) > 0:
            dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.embeddings)
        else:
            self.faiss_index = None

        # Rebuild BM25
        if self.child_chunks:
            self.bm25_corpus = [c["content"].lower().split() for c in self.child_chunks]
            self.bm25 = BM25Okapi(self.bm25_corpus)
        else:
            self.bm25_corpus = []
            self.bm25 = None

        # Rebuild graph
        if self.child_chunks:
            from graph.entity_extractor import extract_entities
            from graph.kg_indexer import build_knowledge_graph
            all_texts = [c["content"] for c in self.child_chunks]
            entities_per_chunk = [extract_entities(t) for t in all_texts]
            self.graph = build_knowledge_graph(self.child_chunks, entities_per_chunk)
        else:
            self.graph = None

        self.documents.pop(doc_id, None)
        self._save()
        logger.info(f"Removed document {doc_id}. Remaining chunks: {len(self.child_chunks)}")
        return True

    def list_documents(self) -> List[Dict]:
        return list(self.documents.values())

    def get_parent_chunk(self, child_chunk: Dict) -> Dict:
        parent_id = child_chunk.get("parent_id")
        if parent_id and parent_id in self.parent_chunks:
            return self.parent_chunks[parent_id]
        return child_chunk


_index: Optional[DocumentIndex] = None


def get_index() -> DocumentIndex:
    global _index
    if _index is None:
        _index = DocumentIndex()
    return _index
