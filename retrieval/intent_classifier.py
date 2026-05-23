from typing import Dict, List

_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "financial": [
        "revenue", "profit", "loss", "balance sheet", "cash flow", "income statement",
        "margin", "eps", "ebitda", "earnings", "financial", "fiscal", "quarterly",
        "annual report", "dividend", "investment", "capital", "debt", "equity",
        "assets", "liabilities", "valuation", "roi", "irr", "npv", "market cap",
        "operating expenses", "gross profit", "net income",
    ],
    "legal": [
        "contract", "clause", "agreement", "liability", "obligation", "compliance",
        "regulation", "statute", "jurisdiction", "indemnify", "warranty", "breach",
        "litigation", "arbitration", "intellectual property", "copyright", "patent",
        "terms and conditions", "legal", "court", "rights", "dispute", "penalty",
        "confidentiality", "non-disclosure", "nda", "termination clause",
    ],
    "research": [
        "study", "methodology", "hypothesis", "experiment", "finding",
        "conclusion", "sample size", "p-value", "confidence interval",
        "statistical significance", "analysis", "literature review", "peer-reviewed",
        "algorithm", "model performance", "accuracy", "benchmark", "abstract",
        "dataset", "training data", "evaluation", "results",
    ],
    "hr": [
        "resume", "cv", "candidate", "skills", "experience", "qualification",
        "education", "degree", "certification", "hire", "recruitment", "employee",
        "salary", "compensation", "job", "position", "interview", "performance review",
        "appraisal", "rate this", "evaluate candidate", "compare candidates",
        "work history", "references", "background check",
    ],
}


def classify_intent(query: str) -> str:
    q_lower = query.lower()
    scores: Dict[str, int] = {intent: 0 for intent in _INTENT_KEYWORDS}
    for intent, keywords in _INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                scores[intent] += 1
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "general"
