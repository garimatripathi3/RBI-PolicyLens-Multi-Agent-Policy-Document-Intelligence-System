"""
Orchestrator: the control flow that connects the four agents.

Flow for a single query:

    Extraction Agent  (once, at index time)  -> entities per doc
    Research Agent    -> retrieve + answer
    QA Agent          -> confidence + flag
    Audit Agent       -> record every step, flush trail

The Extraction Agent runs at *ingestion* time (when documents are loaded),
while Research/QA/Audit run per *query*. The scaffold keeps a prebuilt index
in memory; TODO: persist the index + entities so restarts are cheap.
"""
from __future__ import annotations

from uuid import uuid4

from app.agents.audit_agent import AuditAgent
from app.agents.extraction_agent import ExtractionAgent
from app.agents.qa_agent import QAAgent
from app.agents.research_agent import ResearchAgent
from app.core.config import get_settings
from app.core.observability import trace_span
from app.models.schemas import Entity, PipelineResult
from app.retrieval.chunking import load_and_chunk_dir
from app.retrieval.hybrid_retriever import HybridRetriever

_settings = get_settings()


class PolicyLensPipeline:
    def __init__(self) -> None:
        self.extraction_agent = ExtractionAgent()
        self.qa_agent = QAAgent()
        self._entities: list[Entity] = []
        self._retriever: HybridRetriever | None = None
        self._research_agent: ResearchAgent | None = None

    # --------------------------------------------------------------- ingest
    def ingest(self) -> int:
        """Load + chunk the policy corpus, build the index, extract entities.

        Returns the number of chunks indexed. Call once at startup (and again
        whenever the corpus changes).
        """
        chunks = load_and_chunk_dir(_settings.policies_dir)
        self._retriever = HybridRetriever(chunks)
        self._research_agent = ResearchAgent(self._retriever)

        # Run extraction per source document (dedup by doc_id).
        seen: set[str] = set()
        self._entities = []
        for chunk in chunks:
            if chunk.doc_id in seen:
                continue
            seen.add(chunk.doc_id)
            # NOTE: extracting over the first chunk only is a scaffold shortcut.
            # TODO: extract over the full document text, not one chunk.
            self._entities.extend(
                self.extraction_agent.run(chunk.text, source_doc=chunk.doc_id)
            )
        return len(chunks)

    # ---------------------------------------------------------------- query
    def answer(self, query: str) -> PipelineResult:
        if self._research_agent is None:
            raise RuntimeError("Pipeline not ingested. Call ingest() first.")

        request_id = uuid4().hex
        audit = AuditAgent()

        with trace_span("pipeline", request_id=request_id, query=query):
            audit.record(self.extraction_agent.name, "entities_available",
                         count=len(self._entities))

            research_answer = self._research_agent.run(query)
            audit.record(
                self._research_agent.name,
                "answer_generated",
                n_chunks=len(research_answer.supporting_chunks),
            )

            verdict = self.qa_agent.run(research_answer)
            audit.record(
                self.qa_agent.name,
                "verdict",
                confidence=verdict.confidence.value,
                flagged=verdict.flagged,
            )

            trail = audit.flush()

        return PipelineResult(
            request_id=request_id,
            query=query,
            answer=research_answer,
            verdict=verdict,
            entities=self._entities,
            audit_trail=trail,
        )
