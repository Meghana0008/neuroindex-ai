SYSTEM_PROMPT = """\
You are an expert document analysis assistant with memory of the conversation.

RULES (strictly follow all):
1. Answer using information from the provided context chunks AND the conversation history.
2. Cite every factual claim with [Doc: <filename>, Page: <number>].
3. If multiple sources support a point, cite all of them.
4. For comparison questions, use information retrieved across multiple documents.
5. If the context does not contain enough information, say exactly:
   "The provided documents do not contain sufficient information to answer this question."
6. Never infer, hallucinate, or add information not in the context.
7. Be precise, structured, and comprehensive.\
"""

_HUMAN_TEMPLATE = """\
Context:
{context}

Question: {question}

Answer with inline citations in [Doc: <filename>, Page: <number>] format.\
"""


def build_messages(context: str, question: str, conversation_history: list = None) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        # keep last 6 messages (3 turns) to stay within token limits
        for msg in conversation_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": _HUMAN_TEMPLATE.format(context=context, question=question)})
    return messages
