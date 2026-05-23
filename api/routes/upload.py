import logging
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from access.access_control import set_document_access, valid_access_levels
from api.models import UploadResponse
from config.settings import settings
from ingestion.chunker import create_chunks
from ingestion.doc_classifier import classify_document
from ingestion.hasher import compute_file_hash
from ingestion.indexer import get_index
from ingestion.pdf_parser import parse_pdf

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    access_level: str = Form(default="public"),
    tenant_id: str = Form(default="default"),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    access_level = access_level.lower()
    if access_level not in valid_access_levels():
        raise HTTPException(
            status_code=400,
            detail=f"Invalid access_level '{access_level}'. Choose from: {valid_access_levels()}",
        )

    file_bytes = await file.read()
    doc_hash = compute_file_hash(file_bytes)

    index = get_index()

    # Reject exact-duplicate content (same SHA256 + same tenant)
    exact_duplicate = next(
        (
            d for d in index.documents.values()
            if d.get("doc_hash") == doc_hash
            and d.get("tenant_id", "default") == tenant_id
            and d.get("is_active", True)
        ),
        None,
    )
    if exact_duplicate:
        logger.info(
            f"Exact duplicate detected for '{file.filename}' in tenant='{tenant_id}' "
            f"(matches doc_id={exact_duplicate['doc_id']}) — skipping reindex"
        )
        return UploadResponse(
            doc_id=exact_duplicate["doc_id"],
            filename=file.filename,
            num_pages=exact_duplicate.get("num_pages", 0),
            num_child_chunks=exact_duplicate.get("num_child_chunks", 0),
            num_parent_chunks=0,
            access_level=access_level,
            doc_type=exact_duplicate.get("doc_type", "general"),
            doc_hash=doc_hash,
            duplicate_detected=True,
            status="duplicate_skipped",
        )

    # Reject same filename per tenant — no silent versioning
    existing = next(
        (
            d for d in index.documents.values()
            if d["filename"] == file.filename
            and d.get("tenant_id", "default") == tenant_id
            and d.get("is_active", True)
        ),
        None,
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"A document named '{file.filename}' already exists for tenant '{tenant_id}'. "
                f"Delete the existing document first before uploading a new version."
            ),
        )

    doc_id = str(uuid.uuid4())
    doc_dir = settings.uploads_dir / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = doc_dir / file.filename
    with open(pdf_path, "wb") as fout:
        fout.write(file_bytes)

    logger.info(
        f"Saved '{file.filename}' as doc_id={doc_id} "
        f"tenant='{tenant_id}' access={access_level} hash={doc_hash[:12]}..."
    )

    try:
        pages = parse_pdf(pdf_path)
        doc_type = classify_document(file.filename, pages)
        logger.info(f"'{file.filename}' classified as doc_type='{doc_type}'")

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
                "version": 1,
                "doc_type": doc_type,
                "doc_hash": doc_hash,
                "tenant_id": tenant_id,
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
        doc_type=doc_type,
        doc_hash=doc_hash,
        duplicate_detected=False,
        status="indexed",
    )
