"""
QA Agent.

Job: judge a ResearchAnswer and assign a confidence level. Low-confidence
answers get flagged so a human (or a downstream guardrail) can review them.

Two-signal design:
  1. Retrieval strength  - normalized top reranked score.
  2. Grounding           - is the answer actually supported by the context?
     Primary path uses an LLM-as-judge that returns a 0..1 groundedness score;
     fallback uses token-overlap grounding so it works offline.

The two signals are combined and bucketed into HIGH / MEDIUM / LOW using the
thresholds in settings. Those thresholds are the knobs you calibrate against a
labelled set.
"""
from __future__ import annotations

import json
import logging

from app.agents.base import BaseAgent
from app.core.config import get_settings
from app.core.llm import get_llm
from app.models.schemas import Confidence, QAVerdict, ResearchAnswer

logger = logging.getLogger(__name__)
_settings = get_settings()

_JUDGE_SYSTEM = (
    "You are a strict grader. Given a QUESTION, an ANSWER, and CONTEXT passages, "
    "rate how fully the ANSWER is supported by the CONTEXT. Respond with ONLY a "
    "JSON object: {\"groundedness\": <0..1 float>, \"reason\": \"<short>\"}. "
    "1.0 means every claim is directly supported; 0.0 means unsupported."
)


class QAAgent(BaseAgent):
    name = "qa_agent"

    def __init__(self) -> None:
        self.llm = get_llm()

    def run(self, answer: ResearchAnswer) -> QAVerdict:
        chunks = answer.supporting_chunks
        if not chunks:
            return QAVerdict(
                approved=False,
                confidence=Confidence.LOW,
                score=0.0,
                reasons=["No supporting chunks were retrieved."],
                flagged=True,
            )

        # Signal 1: retrieval strength (squashed to 0..1).
        top_score = max(c.score for c in chunks)
        retrieval_norm = top_score / (top_score + 1.0) if top_score > 0 else 0.0

        # Signal 2: grounding.
        if self.llm.available:
            grounded, reason = self._judge_llm(answer)
        else:
            grounded, reason = self._judge_overlap(answer), "token-overlap grounding"

        combined = 0.4 * retrieval_norm + 0.6 * grounded

        if combined < _settings.low_confidence_threshold:
            confidence = Confidence.LOW
        elif combined < _settings.medium_confidence_threshold:
            confidence = Confidence.MEDIUM
        else:
            confidence = Confidence.HIGH

        flagged = confidence == Confidence.LOW
        answer.confidence = confidence

        return QAVerdict(
            approved=not flagged,
            confidence=confidence,
            score=round(combined, 3),
            reasons=[reason, f"retrieval={retrieval_norm:.2f} grounding={grounded:.2f}"],
            flagged=flagged,
        )

    def _judge_llm(self, answer: ResearchAnswer) -> tuple[float, str]:
        context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in answer.supporting_chunks)
        user = (
            f"QUESTION: {answer.query}\n\nANSWER: {answer.answer}\n\nCONTEXT:\n{context}"
        )
        try:
            raw = self.llm.complete(_JUDGE_SYSTEM, user, temperature=0.0, max_tokens=200)
            data = json.loads(raw)
            return float(data.get("groundedness", 0.0)), str(data.get("reason", ""))
        except Exception as exc:  # pragma: no cover - network/parse dependent
            logger.warning("LLM judge failed (%s); falling back to overlap.", exc)
            return self._judge_overlap(answer), "overlap grounding (judge fallback)"

    @staticmethod
    def _judge_overlap(answer: ResearchAnswer) -> float:
        ctx = " ".join(c.text.lower() for c in answer.supporting_chunks)
        ans_tokens = [t for t in answer.answer.lower().split() if len(t) > 3]
        if not ans_tokens:
            return 0.0
        return sum(1 for t in ans_tokens if t in ctx) / len(ans_tokens)
