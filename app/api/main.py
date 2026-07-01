"""
FastAPI surface for PolicyLens.

Endpoints:
    GET  /health          -> liveness
    POST /ingest          -> (re)build the index from data/policies
    POST /query           -> run the 4-agent pipeline on a question

The pipeline is built once at startup via the lifespan handler and reused.
Endpoints are async; the CPU-bound pipeline call is offloaded to a thread so
the event loop isn't blocked (matches the 'async FastAPI' resume claim).
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.core.pipeline import PolicyLensPipeline
from app.models.schemas import PipelineResult

_pipeline: PolicyLensPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = PolicyLensPipeline()
    # Ingest at startup so the first query is fast.
    n = await asyncio.to_thread(_pipeline.ingest)
    app.state.indexed_chunks = n
    yield
    _pipeline = None


app = FastAPI(title="PolicyLens", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest() -> dict[str, int]:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    n = await asyncio.to_thread(_pipeline.ingest)
    return {"indexed_chunks": n}


@app.post("/query", response_model=PipelineResult)
async def query(req: QueryRequest) -> PipelineResult:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty.")
    # Offload the sync pipeline to a worker thread.
    return await asyncio.to_thread(_pipeline.answer, req.query)
