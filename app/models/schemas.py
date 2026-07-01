"""
Shared Pydantic data contracts that flow between the four agents.

These schemas are the "shared memory" of the pipeline: every agent receives and
returns typed objects, so a malformed hand-off fails loudly instead of silently
corrupting downstream steps.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Entity(BaseModel):
    """A structured entity pulled from a policy document by the Extraction Agent."""

    text: str
    label: str  # e.g. ORG, DATE, MONEY, LAW, GPE, or domain labels
    start_char: int
    end_char: int
    source_doc: str
    normalized: str | None = None  # e.g. "5%" -> "0.05", dates -> ISO


class RetrievedChunk(BaseModel):
    """A chunk returned by the retrieval layer, with its fused/reranked score."""

    chunk_id: str
    doc_id: str
    text: str
    score: float = 0.0  # final reranked score


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResearchAnswer(BaseModel):
    """The Research Agent's answer to a natural-language query."""

    query: str
    answer: str
    supporting_chunks: list[RetrievedChunk] = Field(default_factory=list)
    confidence: Confidence | None = None  # filled by the QA agent


class QAVerdict(BaseModel):
    """The QA Agent's judgement on a ResearchAnswer."""

    approved: bool
    confidence: Confidence
    score: float = 0.0  # 0..1 calibrated grounding score
    reasons: list[str] = Field(default_factory=list)
    flagged: bool = False


class AuditEvent(BaseModel):
    """One traceable event recorded by the Audit Agent."""

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    ts: datetime = Field(default_factory=_now)
    agent: str
    action: str
    payload: dict[str, Any] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    """The full result returned by the API for one query."""

    request_id: str
    query: str
    answer: ResearchAnswer
    verdict: QAVerdict
    entities: list[Entity] = Field(default_factory=list)
    audit_trail: list[AuditEvent] = Field(default_factory=list)
