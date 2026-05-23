import logging
from contextlib import contextmanager
from typing import Dict, List, Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool

from config.settings import settings

logger = logging.getLogger(__name__)

_pool: Optional[psycopg2.pool.SimpleConnectionPool] = None


def _get_pool() -> psycopg2.pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=settings.db_host,
            port=settings.db_port,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
        )
        logger.info(f"DB connected → {settings.db_host}:{settings.db_port}/{settings.db_name}")
    return _pool


@contextmanager
def get_conn():
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def ensure_tables():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id        TEXT PRIMARY KEY,
                    filename      TEXT NOT NULL,
                    num_pages     INT,
                    num_chunks    INT,
                    access_level  TEXT DEFAULT 'public',
                    uploaded_at   TIMESTAMPTZ DEFAULT NOW(),
                    version       INT DEFAULT 1,
                    is_active     BOOL DEFAULT TRUE,
                    deprecated_at TIMESTAMPTZ,
                    tenant_id     TEXT DEFAULT 'default'
                );
            """)
            for col_def in [
                "version INT DEFAULT 1",
                "is_active BOOL DEFAULT TRUE",
                "deprecated_at TIMESTAMPTZ",
                "tenant_id TEXT DEFAULT 'default'",
            ]:
                col_name = col_def.split()[0]
                cur.execute(f"ALTER TABLE documents ADD COLUMN IF NOT EXISTS {col_def};")
                if col_name in ("version", "is_active"):
                    cur.execute(f"UPDATE documents SET {col_name} = %s WHERE {col_name} IS NULL;",
                                (1 if col_name == "version" else True,))
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id     TEXT PRIMARY KEY,
                    doc_id       TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
                    doc_name     TEXT,
                    page_number  INT,
                    content      TEXT,
                    chunk_type   TEXT,
                    parent_id    TEXT,
                    tenant_id    TEXT DEFAULT 'default',
                    created_at   TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tenant_id TEXT DEFAULT 'default';")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS request_traces (
                    trace_id              TEXT PRIMARY KEY,
                    timestamp             TIMESTAMPTZ,
                    query                 TEXT,
                    agent_type            TEXT,
                    detected_intent       TEXT,
                    num_chunks_final      INT,
                    retrieval_sufficient  BOOL,
                    input_tokens          INT,
                    output_tokens         INT,
                    estimated_cost_usd    FLOAT,
                    total_latency_ms      FLOAT,
                    confidence_score      FLOAT,
                    groundedness_score    FLOAT,
                    hallucination_risk    BOOL,
                    prompt_injection_blocked BOOL,
                    tenant_id             TEXT DEFAULT 'default'
                );
            """)
            cur.execute("ALTER TABLE request_traces ADD COLUMN IF NOT EXISTS tenant_id TEXT DEFAULT 'default';")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(doc_id, page_number);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_tenant ON documents(tenant_id);")
    logger.info("DB tables ready.")


def insert_document(doc_id: str, filename: str, num_pages: int,
                    num_chunks: int, access_level: str = "public",
                    version: int = 1, is_active: bool = True,
                    tenant_id: str = "default"):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents (doc_id, filename, num_pages, num_chunks, access_level, version, is_active, tenant_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE
                SET filename=EXCLUDED.filename, num_pages=EXCLUDED.num_pages,
                    num_chunks=EXCLUDED.num_chunks, access_level=EXCLUDED.access_level,
                    version=EXCLUDED.version, is_active=EXCLUDED.is_active,
                    tenant_id=EXCLUDED.tenant_id;
            """, (doc_id, filename, num_pages, num_chunks, access_level, version, is_active, tenant_id))


def delete_document(doc_id: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
            return cur.rowcount > 0


def deprecate_document(doc_id: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE documents SET is_active = FALSE, deprecated_at = NOW()
                WHERE doc_id = %s;
            """, (doc_id,))


def list_documents() -> List[Dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT doc_id, filename, num_pages, num_chunks,
                       access_level, uploaded_at::text AS uploaded_at
                FROM documents ORDER BY uploaded_at DESC;
            """)
            return [dict(r) for r in cur.fetchall()]


def get_document(doc_id: str) -> Optional[Dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT doc_id, filename, num_pages, num_chunks,
                       access_level, uploaded_at::text AS uploaded_at
                FROM documents WHERE doc_id = %s;
            """, (doc_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def reset_all_data():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE chunks, request_traces, documents CASCADE;")
    logger.info("All data reset in PostgreSQL.")


def insert_chunks(chunks: List[Dict]):
    if not chunks:
        return
    rows = [
        (c["chunk_id"], c["doc_id"], c["doc_name"], c["page_number"],
         c["content"], c.get("chunk_type", "child"), c.get("parent_id"),
         c.get("tenant_id", "default"))
        for c in chunks
    ]
    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO chunks
                   (chunk_id, doc_id, doc_name, page_number, content, chunk_type, parent_id, tenant_id)
                   VALUES %s ON CONFLICT (chunk_id) DO NOTHING;""",
                rows,
                page_size=500,
            )


def get_chunks_by_doc(doc_id: str) -> List[Dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM chunks WHERE doc_id = %s ORDER BY page_number;", (doc_id,))
            return [dict(r) for r in cur.fetchall()]


def get_stats() -> Dict:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents;")
            total_docs = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunks;")
            total_chunks = cur.fetchone()[0]
    return {"total_documents": total_docs, "total_chunks": total_chunks}
