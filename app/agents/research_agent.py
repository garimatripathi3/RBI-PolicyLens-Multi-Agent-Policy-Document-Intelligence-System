"""
Research Agent.

Job: answer a natural-language question by (1) retrieving the most relevant
policy chunks via the hybrid retriever, then (2) synthesizing an answer from
ONLY those chunks with an LLM, with a strict grounded prompt.

If no LLM is configured, returns an extractive answer (top chunk) so the whole
pipeline runs offline.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.core.llm import get_llm
from app.core.observability import trace_span
from app.models.schemas import ResearchAnswer
from app.retrieval.hybrid_retriever import HybridRetriever

_SYSTEM_PROMPT = (
    "You are a careful policy analyst. Answer the user's question using ONLY the "
    "provided context passages. Cite the source document in brackets like [doc]. "
    "If the answer is not present in the context, reply exactly: "
    "'Not found in the provided policies.' Do not use outside knowledge."
)


class ResearchAgent(BaseAgent):
    name = "research_agent"

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.llm = get_llm()

    def run(self, query: str) -> ResearchAnswer:
        with trace_span(self.name, query=query):
            chunks = self.retriever.search(query)

            if self.llm.available and chunks:
                answer_text = self._synthesize_llm(query, chunks)
            else:
                answer_text = self._synthesize_extractive(chunks)

            return ResearchAnswer(
                query=query, answer=answer_text, supporting_chunks=chunks
            )

    def _synthesize_llm(self, query, chunks) -> str:
        context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in chunks)
        user = f"Context:\n{context}\n\nQuestion: {query}"
        return self.llm.complete(_SYSTEM_PROMPT, user, temperature=0.0)

    def _synthesize_extractive(self, chunks) -> str:
        if not chunks:
            return "Not found in the provided policies."
        top = chunks[0]
        return f"(extractive) Based on {top.doc_id}: {top.text}"
