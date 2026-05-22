import logging

from fastapi import APIRouter, HTTPException

from api.models import DeleteResponse, DocumentInfo
from db.database import delete_document, get_document, list_documents
from ingestion.indexer import get_index

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/documents", response_model=list[DocumentInfo])
def list_all_documents():
    rows = list_documents()
    return [DocumentInfo(**r) for r in rows]


@router.get("/documents/{doc_id}", response_model=DocumentInfo)
def get_one_document(doc_id: str):
    row = get_document(doc_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return DocumentInfo(**row)


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_one_document(doc_id: str):
    db_removed = delete_document(doc_id)
    index_removed = get_index().remove_document(doc_id)

    if not db_removed and not index_removed:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    logger.info(f"Document {doc_id} deleted from DB and indexes.")
    return DeleteResponse(doc_id=doc_id, status="deleted")
