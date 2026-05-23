import logging

from fastapi import APIRouter

from ingestion.indexer import get_index

router = APIRouter()
logger = logging.getLogger(__name__)


@router.delete("/admin/reset")
def reset_all_data():
    """Wipe all indexed data — deletes all pickle files and clears PostgreSQL tables."""
    index = get_index()
    index.reset()

    try:
        from db.database import reset_all_data as db_reset
        db_reset()
    except Exception as e:
        logger.warning(f"PostgreSQL reset skipped: {e}")

    return {"status": "reset", "message": "All data cleared. System is ready for fresh uploads."}
