"""
Auto-classify document type at upload time.
Type is stored in doc metadata and chunk metadata so it can be used for shard routing.

At scale: replace keyword scoring with a small text classifier or embedding similarity.
"""
from typing import List

from ingestion.pdf_parser import PageContent

_DOC_TYPE_KEYWORDS = {
    "financial": [
        "revenue", "profit", "loss", "balance sheet", "cash flow", "income statement",
        "quarterly", "annual report", "financial statements", "earnings", "ebitda",
        "operating expenses", "gross margin", "net income", "fiscal year", "dividends",
    ],
    "legal": [
        "contract", "agreement", "clause", "liability", "compliance", "regulation",
        "terms and conditions", "legal notice", "indemnity", "warranty", "jurisdiction",
        "arbitration", "intellectual property", "confidentiality", "breach", "statute",
    ],
    "research": [
        "abstract", "methodology", "hypothesis", "results", "conclusion",
        "experiment", "dataset", "statistical significance", "literature review",
        "peer-reviewed", "sample size", "confidence interval", "p-value",
    ],
    "hr": [
        "resume", "curriculum vitae", "candidate", "skills", "work experience",
        "education", "employment history", "qualifications", "references",
        "job application", "certifications", "salary", "recruitment",
    ],
}

_MIN_KEYWORD_HITS = 2  # require at least 2 keyword matches to classify


def classify_document(filename: str, pages: List[PageContent]) -> str:
    """
    Classify a document into one of: financial / legal / research / hr / general.
    Uses filename + first 2 pages content for speed.
    """
    sample_text = filename.lower() + " " + " ".join(
        p.text[:600].lower() for p in pages[:2]
    )
    scores = {doc_type: 0 for doc_type in _DOC_TYPE_KEYWORDS}
    for doc_type, keywords in _DOC_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in sample_text:
                scores[doc_type] += 1

    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] >= _MIN_KEYWORD_HITS else "general"
