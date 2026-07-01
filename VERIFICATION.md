# Verification Notes

This is the **complete** implementation — every agent has real logic, not stubs.
Here's an honest account of what was tested and where, so you know exactly what
to check on your own machine.

## Verified working (in a sandbox with no model downloads)
- Full request flow through all four agents (Extraction → Research → QA → Audit)
- **Real BM25** retrieval via `rank_bm25` (confirmed active, not the fallback)
- Reciprocal Rank Fusion combining candidate lists
- Async FastAPI app: `/health`, `/ingest`, `/query` all respond
- QA confidence scoring and low-confidence flagging
- Full audit trail recorded and returned
- All 4 pytest smoke tests pass
- Graceful degradation when models/keys are absent (logged, never silent)

## Written as real code, NOT runnable in the build sandbox (HuggingFace/OpenAI
## are network-blocked there) — verify these on your machine:
- **Dense retrieval**: `sentence-transformers` embeddings + FAISS index.
  On first run it downloads `all-MiniLM-L6-v2`. Confirm with:
  `python -c "from app.retrieval.hybrid_retriever import HybridRetriever"` after
  ingesting, and check the logs show no "Dense index unavailable" warning.
- **Cross-encoder rerank**: `cross-encoder/ms-marco-MiniLM-L-6-v2`. Same idea —
  the "CrossEncoder unavailable" warning should disappear once it downloads.
- **spaCy NER + EntityRuler**: run `python -m spacy download en_core_web_sm`
  first, then extraction uses real NER instead of the regex fallback.
- **LLM synthesis + LLM-as-judge**: set `OPENAI_API_KEY` in `.env`. The Research
  Agent will synthesize grounded answers and the QA Agent will use the judge.

## How to confirm the full stack on your machine
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env          # add OPENAI_API_KEY for LLM paths
make test                     # 4 tests should pass
make run                      # then POST to /query
```

If you see any "unavailable" warnings in the logs after installing everything,
that component fell back — check the install and network for that model.
