import logging
from typing import List

from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def generate_query_variants(query: str, n: int = 3) -> List[str]:
    if not settings.openai_api_key:
        return [query]

    try:
        resp = _client().chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Generate exactly {n} alternative phrasings of the user's question "
                        "to improve document retrieval. Output one query per line, no numbering, "
                        "no extra text. Make them semantically diverse."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        lines = [l.strip() for l in resp.choices[0].message.content.strip().split("\n") if l.strip()]
        return [query] + lines[:n]
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return [query]


def generate_hyde_document(query: str) -> str:
    if not settings.openai_api_key:
        return query

    try:
        resp = _client().chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Write a detailed, expert-level passage from a document that would "
                        "perfectly answer the question below. Be specific. Do not mention that "
                        "you are generating a hypothetical document."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.5,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}")
        return query
