"""
Central configuration. Reads from environment (.env in dev).

Nothing here should hard-code secrets. Copy .env.example -> .env locally.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider ---
    openai_api_key: str | None = Field(default=None)
    llm_model: str = Field(default="gpt-4o-mini")

    # --- spaCy model used by the Extraction Agent ---
    spacy_model: str = Field(default="en_core_web_sm")

    # --- Retrieval knobs ---
    top_k_bm25: int = Field(default=20)
    top_k_dense: int = Field(default=20)
    top_k_final: int = Field(default=5)
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # --- QA thresholds ---
    low_confidence_threshold: float = Field(default=0.35)
    medium_confidence_threshold: float = Field(default=0.6)

    # --- Observability (Langfuse). Optional; pipeline runs without it. ---
    langfuse_public_key: str | None = Field(default=None)
    langfuse_secret_key: str | None = Field(default=None)
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # --- Data ---
    policies_dir: str = Field(default="data/policies")


@lru_cache
def get_settings() -> Settings:
    return Settings()
