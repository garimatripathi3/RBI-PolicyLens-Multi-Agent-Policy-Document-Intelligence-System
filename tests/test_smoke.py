"""
Smoke tests: prove the scaffold runs end-to-end offline (no API keys, no models).

These are intentionally light. As you replace the TODO sections with real
implementations, add targeted unit tests for each agent (extraction accuracy,
retrieval recall@k, QA calibration, audit completeness).
"""
from __future__ import annotations

from app.core.pipeline import PolicyLensPipeline
from app.models.schemas import Confidence


def _build_pipeline() -> PolicyLensPipeline:
    p = PolicyLensPipeline()
    p.ingest()
    return p


def test_ingest_indexes_chunks():
    p = PolicyLensPipeline()
    n = p.ingest()
    assert n > 0


def test_query_returns_result():
    p = _build_pipeline()
    result = p.answer("What is the deductible?")
    assert result.query == "What is the deductible?"
    assert result.answer.answer  # non-empty
    assert result.verdict.confidence in set(Confidence)


def test_audit_trail_records_all_agents():
    p = _build_pipeline()
    result = p.answer("What is covered out of network?")
    agents_seen = {e.agent for e in result.audit_trail}
    # research + qa should always appear
    assert "research_agent" in agents_seen
    assert "qa_agent" in agents_seen


def test_empty_corpus_query_flags_low_confidence(tmp_path, monkeypatch):
    # Point the pipeline at an empty dir -> no chunks -> low confidence flag.
    from app.core import config

    monkeypatch.setattr(
        config.get_settings(), "policies_dir", str(tmp_path), raising=False
    )
    p = PolicyLensPipeline()
    p.ingest()
    result = p.answer("anything")
    assert result.verdict.flagged is True
