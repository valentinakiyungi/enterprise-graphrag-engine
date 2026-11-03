"""Centralised, environment-driven configuration."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- Application -------------------------------------------------------
    app_name: str = "Enterprise GraphRAG Engine"
    environment: Literal["development", "staging", "production", "test"] = "development"
    log_level: str = "INFO"
    api_key: str | None = Field(
        default=None, description="If set, every /api request must send X-API-Key."
    )
    cors_origins: str = "*"

    # --- OpenAI ------------------------------------------------------------
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_eval_model: str | None = None  # defaults to chat model
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    llm_temperature: float = 0.0
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3

    # --- PostgreSQL / pgvector ---------------------------------------------
    database_url: str = "postgresql://graphrag:graphrag@localhost:5432/graphrag"
    pg_pool_min: int = 1
    pg_pool_max: int = 10

    # --- Neo4j -------------------------------------------------------------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_http_url: str = "http://localhost:7474"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "graphrag-password"
    neo4j_database: str = "neo4j"
    neo4j_transport: Literal["auto", "bolt", "http"] = "auto"

    # --- Ingestion ---------------------------------------------------------
    chunk_size: int = 1200  # characters
    chunk_overlap: int = 200
    extraction_concurrency: int = 4
    embedding_batch_size: int = 64

    # --- Retrieval ---------------------------------------------------------
    vector_top_k: int = 8
    keyword_top_k: int = 8
    graph_max_hops: int = 2
    graph_seed_entities: int = 8
    graph_max_facts: int = 40
    graph_top_chunks: int = 8
    entity_link_max_distance: float = 0.55  # cosine distance
    rrf_k: int = 60
    final_context_chunks: int = 8
    max_context_chars: int = 14000

    # --- Evaluation / self-correction ---------------------------------------
    evaluate_on_query: bool = True
    eval_engine: Literal["auto", "ragas", "llm_judge"] = "auto"
    faithfulness_threshold: float = 0.75
    context_precision_threshold: float = 0.5
    max_correction_attempts: int = 2
    hallucination_threshold: float = 0.5  # faithfulness below => hallucinated

    @property
    def eval_model(self) -> str:
        return self.openai_eval_model or self.openai_chat_model

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
