import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.analytics import router as analytics_router
from api.routes.documents import router as documents_router
from api.routes.health import router as health_router
from api.routes.query import router as query_router
from api.routes.upload import router as upload_router
from db.database import ensure_tables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="NeuroIndex AI",
    description=(
        "Enterprise-Grade Multimodal RAG Platform with "
        "Page-Level Citation Intelligence"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    try:
        ensure_tables()
        logging.getLogger(__name__).info("PostgreSQL connected and tables verified.")
    except Exception as e:
        logging.getLogger(__name__).warning(f"PostgreSQL not available: {e}")

app.include_router(upload_router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(query_router, prefix="/api/v1", tags=["Retrieval"])
app.include_router(documents_router, prefix="/api/v1", tags=["Documents"])
app.include_router(health_router, prefix="/api/v1", tags=["System"])
app.include_router(analytics_router, prefix="/api/v1", tags=["Analytics"])
