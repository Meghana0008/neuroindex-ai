import uuid
import logging
from typing import Dict, Any, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.pdf_parser import PageContent

logger = logging.getLogger(__name__)

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _make_chunk(
    content: str,
    doc_id: str,
    doc_name: str,
    page_number: int,
    chunk_type: str,
    parent_id: str = None,
) -> Dict[str, Any]:
    return {
        "chunk_id": str(uuid.uuid4()),
        "doc_id": doc_id,
        "doc_name": doc_name,
        "page_number": page_number,
        "content": content,
        "chunk_type": chunk_type,
        "parent_id": parent_id,
    }


def create_chunks(
    pages: List[PageContent],
    doc_id: str,
    doc_name: str,
    child_chunk_size: int = 512,
    child_chunk_overlap: int = 64,
    parent_chunk_size: int = 2048,
    parent_chunk_overlap: int = 128,
) -> Tuple[List[Dict], List[Dict]]:
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=_SEPARATORS,
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        separators=_SEPARATORS,
    )

    child_chunks: List[Dict] = []
    parent_chunks: List[Dict] = []

    for page in pages:
        if not page.text.strip():
            continue

        for parent_text in parent_splitter.split_text(page.text):
            parent = _make_chunk(
                content=parent_text,
                doc_id=doc_id,
                doc_name=doc_name,
                page_number=page.page_number,
                chunk_type="parent",
            )
            parent_chunks.append(parent)

            for child_text in child_splitter.split_text(parent_text):
                if len(child_text.strip()) < 20:
                    continue
                child = _make_chunk(
                    content=child_text,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page_number=page.page_number,
                    chunk_type="child",
                    parent_id=parent["chunk_id"],
                )
                child_chunks.append(child)

    logger.info(
        f"'{doc_name}': {len(child_chunks)} child chunks, "
        f"{len(parent_chunks)} parent chunks"
    )
    return child_chunks, parent_chunks
