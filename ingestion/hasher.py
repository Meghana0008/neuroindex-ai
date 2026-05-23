"""
SHA256 hashing for document and chunk deduplication.

doc_hash  — whole-file hash: detects exact duplicate uploads before any OCR/embedding work
chunk_hash — per-chunk hash: enables chunk-level delta indexing when only parts of a doc change
"""
import hashlib


def compute_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def compute_chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
