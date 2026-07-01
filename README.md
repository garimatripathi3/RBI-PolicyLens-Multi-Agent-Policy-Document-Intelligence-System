# PolicyLens — Multi-Agent Document Intelligence (Starter Scaffold)

A 4-agent platform for question-answering over policy documents. Complete,
working implementation: four agents, hybrid retrieval (BM25 + dense + RRF +
cross-encoder rerank), async FastAPI, Docker, and optional Langfuse tracing.

See `VERIFICATION.md` for exactly what was tested where. The code degrades
gracefully when a model/key is missing (logged, never silent), so it runs
offline for development and lights up fully once models and an API key are
available.

## The four agents

| Agent | Responsibility | File |
|-------|----------------|------|
| **Extraction Agent** | Pull structured entities via spaCy NER, validate through Pydantic | `app/agents/extraction_agent.py` |
| **Research Agent** | Answer NL queries from retrieved chunks (hybrid search + rerank) | `app/agents/research_agent.py` |
| **QA Agent** | Assign confidence, flag low-confidence answers | `app/agents/qa_agent.py` |
| **Audit Agent** | Record a traceable log of every step | `app/agents/audit_agent.py` |

Control flow lives in `app/core/pipeline.py`. Retrieval (chunking, BM25, dense,
cross-encoder rerank) lives in `app/retrieval/`.

## Architecture

```
                 ingest time                          query time
data/policies ──► chunking ──► HybridRetriever   POST /query ──► ResearchAgent
      │                                                             │  (retrieve + answer)
      └──► ExtractionAgent ──► entities                             ▼
                                                              QAAgent (confidence + flag)
                                                                    │
                                                                    ▼
                                                              AuditAgent (trace)
                                                                    │
                                                                    ▼
                                                              PipelineResult (JSON)
```

## Quick start (offline, no keys)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make run          # or: uvicorn app.api.main:app --reload
```

Then:

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"query":"What is the deductible?"}' | python -m json.tool
```

Run tests:

```bash
make test
```

Run in Docker:

```bash
cp .env.example .env
make docker-up
```



## Layout

```
app/
  agents/     extraction, research, qa, audit
  retrieval/  chunking, hybrid_retriever
  core/       config, observability, pipeline
  api/        FastAPI app
  models/     Pydantic schemas (shared agent contracts)
data/policies/  drop your .md/.txt/.pdf policies here
tests/        offline smoke tests
```
