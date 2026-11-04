"""
rag_chain.py -- the RAG query layer: given a question, retrieves the
most relevant real documents from the vector store (vector_store.py,
built from corpus_builder.py's real corpus) and turns them into an
answer, with the project's usual real-model / offline-floor split.

ask() is the single entry point api.py calls: it first checks the
deterministic tool router (tools.py) for a live-state question, then
falls back to RAG retrieval, and only reaches for a live Claude call
if one is available -- degrading to the offline paths on any
failure, same discipline as multi_day_runner.py's try/except around
generate_brief and judge_escalation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_chroma import Chroma

from rag.tools import route_offline


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: List[Dict[str, str]] = field(default_factory=list)
    mode: str = "offline"  # "offline" | "live" | "tool"


def retrieve(vector_store: Chroma, question: str, k: int = 4) -> list:
    return vector_store.similarity_search(question, k=k)


def answer_question_offline(vector_store: Chroma, question: str, k: int = 4) -> RagAnswer:
    """Deterministic extractive answer: the retrieved documents'
    real text, verbatim, with no model call and nothing summarized
    or invented -- the same 'never say more than the data supports'
    discipline as generate_brief_offline. This is the tested default
    path and the graceful degrade if a live model call fails."""
    docs = retrieve(vector_store, question, k=k)
    if not docs:
        return RagAnswer(answer="No relevant real records found for that question.", sources=[])
    body = "\n\n".join(f"[{d.metadata.get('source', '?')}] {d.page_content}" for d in docs)
    return RagAnswer(
        answer=f"(offline retrieval -- no model call) Most relevant real records:\n\n{body}",
        sources=[dict(d.metadata) for d in docs],
        mode="offline",
    )


def answer_question(
    vector_store: Chroma, question: str, api_key: Optional[str] = None, k: int = 4
) -> RagAnswer:
    """Real path: retrieves the same top-k real documents, then asks
    Claude (langchain_anthropic.ChatAnthropic) to answer using ONLY
    those documents -- same 'narrate only what's given' rule as
    llm_narrative.generate_brief and escalation_judge.judge_escalation.
    Raises if no model/network is available; callers wanting a
    no-network path should call answer_question_offline directly, or
    use ask() below, which degrades automatically."""
    from langchain_anthropic import ChatAnthropic  # noqa: PLC0415 -- optional live path

    docs = retrieve(vector_store, question, k=k)
    if not docs:
        return RagAnswer(answer="No relevant real records found for that question.", sources=[])

    context = "\n\n".join(d.page_content for d in docs)
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929", api_key=api_key, max_tokens=400, timeout=30
    )
    system = (
        "Answer the question using ONLY the real records given below -- never "
        "invent a number, a date, or a fact not present in them. If the records "
        "don't contain enough to answer, say so plainly rather than guessing."
    )
    response = llm.invoke(
        [("system", system), ("human", f"Records:\n{context}\n\nQuestion: {question}")]
    )
    return RagAnswer(answer=str(response.content), sources=[dict(d.metadata) for d in docs], mode="live")


def ask(
    vector_store: Chroma,
    question: str,
    api_key: Optional[str] = None,
    offline: bool = False,
    k: int = 4,
) -> RagAnswer:
    """Single entry point: tool router first (a live-state question
    gets answered by actually running the real detectors, not by
    retrieving a possibly-stale document), then RAG retrieval,
    live-model-with-offline-fallback unless `offline` is forced."""
    tool_hit = route_offline(question)
    if tool_hit is not None:
        return RagAnswer(answer=tool_hit.answer, sources=[{"tool": tool_hit.tool_used}], mode="tool")

    if offline:
        return answer_question_offline(vector_store, question, k=k)

    try:
        return answer_question(vector_store, question, api_key=api_key, k=k)
    except Exception:  # noqa: BLE001 -- degrade, never fail the request
        return answer_question_offline(vector_store, question, k=k)
