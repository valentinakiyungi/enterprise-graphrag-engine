from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  """Runtime configuration, read from environment variables or a .env file."""

  model_config = SettingsConfigDict(env_file=".env", extra="ignore")

  # OpenAI / gateway
  openai_api_key: str = ""
  openai_base_url: Optional[str] = None
  chat_model: str = "gpt-4o-mini"
  embedding_model: str = "text-embedding-3-small"
  embedding_dim: int = 1536

  # PostgreSQL (pgvector)
  postgres_dsn: str = "postgresql://graphrag:graphrag@localhost:5432/graphrag"

  # Neo4j
  neo4j_uri: str = "bolt://localhost:7687"
  neo4j_user: str = "neo4j"
  neo4j_password: str = "graphrag-password"
  neo4j_database: str = "neo4j"

  # Retrieval
  vector_top_k: int = 5
  graph_max_hops: int = 2
  graph_fact_limit: int = 40
  graph_chunk_limit: int = 5
  chunk_size: int = 800
  chunk_overlap: int = 100

  # Evaluation
  evaluate_by_default: bool = False
  faithfulness_threshold: float = 0.7

  # API security (empty = auth disabled)
  api_key: str = ""


@lru_cache
def get_settings() -> Settings:
  return Settings()
