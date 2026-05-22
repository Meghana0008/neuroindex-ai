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
    # runs on startup, safe to call multiple times
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id        TEXT PRIMARY KEY,
                    filename      TEXT NOT NULL,
                    num_pages     INT,
                    num_chunks    INT,
                    access_level  TEXT DEFAULT 'public',
                    uploaded_at   TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id     TEXT PRIMARY KEY,
                    doc_id       TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
                    doc_name     TEXT,
                    page_number  INT,
                    content      TEXT,
                    chunk_type   TEXT,
                    parent_id    TEXT,
                    created_at   TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(doc_id, page_number);")
    logger.info("DB tables ready.")


def insert_document(doc_id: str, filename: str, num_pages: int,
                    num_chunks: int, access_level: str = "public"):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents (doc_id, filename, num_pages, num_chunks, access_level)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE
                SET filename=EXCLUDED.filename, num_pages=EXCLUDED.num_pages,
                    num_chunks=EXCLUDED.num_chunks, access_level=EXCLUDED.access_level;
            """, (doc_id, filename, num_pages, num_chunks, access_level))


def delete_document(doc_id: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
            return cur.rowcount > 0


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


def insert_chunks(chunks: List[Dict]):
    if not chunks:
        return
    rows = [
        (c["chunk_id"], c["doc_id"], c["doc_name"], c["page_number"],
         c["content"], c.get("chunk_type", "child"), c.get("parent_id"))
        for c in chunks
    ]
    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO chunks
                   (chunk_id, doc_id, doc_name, page_number, content, chunk_type, parent_id)
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
