import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_nlp = None
_nlp_loaded = False

_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "LAW", "WORK_OF_ART", "FAC"}


def _get_nlp():
    global _nlp, _nlp_loaded
    if not _nlp_loaded:
        _nlp_loaded = True
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            _nlp = None
    return _nlp


def extract_entities(text: str) -> List[str]:
    nlp = _get_nlp()

    if nlp is None:
        words = text.split()
        return list({w.strip(".,;:!?") for w in words if w and w[0].isupper() and len(w) > 3})[:15]

    doc = nlp(text[:5000])
    entities: List[str] = []

    for ent in doc.ents:
        if ent.label_ in _ENTITY_LABELS:
            entities.append(ent.text.lower().strip())

    for chunk in doc.noun_chunks:
        parts = chunk.text.split()
        if 2 <= len(parts) <= 3:
            entities.append(chunk.text.lower().strip())

    return list(dict.fromkeys(entities))  # deduplicate preserving order
