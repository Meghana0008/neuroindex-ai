import logging
import shutil
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from access.access_control import set_document_access, valid_access_levels
from api.models import UploadResponse
from config.settings import settings
from ingestion.chunker import create_chunks
from ingestion.indexer import get_index
from ingestion.pdf_parser import parse_pdf

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    access_level: str = Form(default="public"),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    access_level = access_level.lower()
    if access_level not in valid_access_levels():
        raise HTTPException(
            status_code=400,
            detail=f"Invalid access_level '{access_level}'. "
                   f"Choose from: {valid_access_levels()}",
        )

    # auto-remove old version if same filename already indexed
    index = get_index()
    existing = next(
        (d for d in index.documents.values() if d["filename"] == file.filename), None
    )
    new_version = 1
    if existing:
        old_version = existing.get("version", 1)
        new_version = old_version + 1
        logger.info(
            f"Re-upload detected for '{file.filename}' — "
            f"v{old_version} → v{new_version}"
        )
        try:
            from db.database import deprecate_document
            deprecate_document(existing["doc_id"])
        except Exception as e:
            logger.warning(f"Could not mark old version deprecated in DB: {e}")
        index.remove_document(existing["doc_id"])
        try:
            from db.database import delete_document
            delete_document(existing["doc_id"])
        except Exception:
            pass

    doc_id = str(uuid.uuid4())
    doc_dir = settings.uploads_dir / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = doc_dir / file.filename
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"Saved '{file.filename}' as doc_id={doc_id} access={access_level}")

    try:
        pages = parse_pdf(pdf_path)
        child_chunks, parent_chunks = create_chunks(
            pages=pages,
            doc_id=doc_id,
            doc_name=file.filename,
            child_chunk_size=settings.child_chunk_size,
            child_chunk_overlap=settings.child_chunk_overlap,
            parent_chunk_size=settings.parent_chunk_size,
            parent_chunk_overlap=settings.parent_chunk_overlap,
        )
        get_index().add_chunks(
            child_chunks,
            parent_chunks,
            doc_metadata={
                "doc_id": doc_id,
                "filename": file.filename,
                "num_pages": len(pages),
                "num_child_chunks": len(child_chunks),
                "access_level": access_level,
                "version": new_version,
            },
        )
        set_document_access(doc_id, access_level)
    except Exception as e:
        logger.error(f"Ingestion failed for '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        num_pages=len(pages),
        num_child_chunks=len(child_chunks),
        num_parent_chunks=len(parent_chunks),
        access_level=access_level,
        status="indexed",
    )
