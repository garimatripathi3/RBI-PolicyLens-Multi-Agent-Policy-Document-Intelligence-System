"""
Hybrid retrieval with cross-encoder reranking.

Pipeline:
    query
      -> BM25 lexical search        (rank_bm25, top_k_bm25 candidates)
      -> dense vector search        (sentence-transformers + FAISS, top_k_dense)
      -> Reciprocal Rank Fusion     (combine the two candidate lists)
      -> cross-encoder rerank        (top_k_final results)

Real implementations are the primary path. If the embedding / cross-encoder
models can't be loaded (e.g. no network on first run, offline CI), the retriever
degrades to lexical-only + a lightweight rerank so the app still runs. The
degradation is logged so it's never silent.
"""
from __future__ import annotations

import logging
import re

import numpy as np

from app.core.config import get_settings
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)
_settings = get_settings()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    def __init__(self, chunks: list[RetrievedChunk]):
        self.chunks = chunks
        self._corpus_tokens = [_tokenize(c.text) for c in chunks]

        self._bm25 = None
        self._embedder = None
        self._faiss_index = None
        self._cross_encoder = None

        self._build_bm25_index()
        self._build_dense_index()
        self._load_cross_encoder()

    # ------------------------------------------------------------------ BM25
    def _build_bm25_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None
        except Exception as exc:  # pragma: no cover
            logger.warning("rank_bm25 unavailable (%s); using fallback BM25.", exc)
            self._bm25 = None

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        q = _tokenize(query)
        if self._bm25 is not None:
            scores = np.asarray(self._bm25.get_scores(q), dtype=float)
        else:
            scores = self._fallback_bm25_scores(q)
        order = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in order]

    def _fallback_bm25_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.chunks))
        qset = set(query_tokens)
        for i, toks in enumerate(self._corpus_tokens):
            if not toks:
                continue
            scores[i] = sum(toks.count(t) for t in qset) / len(toks)
        return scores

    # ------------------------------------------------------------------ dense
    def _build_dense_index(self) -> None:
        if not self.chunks:
            return
        try:
            from sentence_transformers import SentenceTransformer
            import faiss

            self._embedder = SentenceTransformer(_settings.embedding_model)
            embeddings = self._embedder.encode(
                [c.text for c in self.chunks],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")

            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            self._faiss_index = index
        except Exception as exc:  # pragma: no cover - network/model dependent
            logger.warning(
                "Dense index unavailable (%s); retrieval will be lexical-only.", exc
            )
            self._embedder = None
            self._faiss_index = None

    def _dense_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if self._embedder is None or self._faiss_index is None:
            return []
        qv = self._embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, idx = self._faiss_index.search(qv, min(top_k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]

    # ------------------------------------------------------------------ rerank
    def _load_cross_encoder(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(_settings.cross_encoder_model)
        except Exception as exc:  # pragma: no cover - network/model dependent
            logger.warning(
                "CrossEncoder unavailable (%s); using token-overlap rerank.", exc
            )
            self._cross_encoder = None

    def _rerank(self, query: str, candidate_idx: list[int]) -> list[tuple[int, float]]:
        if not candidate_idx:
            return []
        if self._cross_encoder is not None:
            pairs = [(query, self.chunks[i].text) for i in candidate_idx]
            scores = self._cross_encoder.predict(pairs)
            ranked = sorted(
                zip(candidate_idx, scores), key=lambda x: x[1], reverse=True
            )
            return [(int(i), float(s)) for i, s in ranked]
        q = set(_tokenize(query))
        ranked = sorted(
            ((i, float(len(q & set(self._corpus_tokens[i])))) for i in candidate_idx),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked

    # ------------------------------------------------------------------ fusion
    @staticmethod
    def _reciprocal_rank_fusion(
        bm25: list[tuple[int, float]],
        dense: list[tuple[int, float]],
        k: int = 60,
    ) -> list[int]:
        """Combine two ranked lists with RRF (rank-based, score-scale agnostic)."""
        scores: dict[int, float] = {}
        for ranked in (bm25, dense):
            for rank, (idx, _) in enumerate(ranked):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return [idx for idx, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    # ------------------------------------------------------------------ public
    def search(self, query: str) -> list[RetrievedChunk]:
        if not self.chunks:
            return []

        bm25 = self._bm25_search(query, _settings.top_k_bm25)
        dense = self._dense_search(query, _settings.top_k_dense)

        fused_idx = self._reciprocal_rank_fusion(bm25, dense)
        candidate_pool = fused_idx[: max(_settings.top_k_bm25, _settings.top_k_dense)]

        reranked = self._rerank(query, candidate_pool)

        results: list[RetrievedChunk] = []
        for i, score in reranked[: _settings.top_k_final]:
            results.append(self.chunks[i].model_copy(update={"score": score}))
        return results
