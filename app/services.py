"""Wires configuration, stores, models and the workflow together."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class Services:
  settings: Settings
  vector_store: Any
  graph_store: Any
  embeddings: Any
  llm: Any
  ingestor: Any
  graph: Any
  evaluator: Optional[Any] = None

  def close(self) -> None:
    for store in (self.vector_store, self.graph_store):
      try:
        store.close()
      except Exception:
        logger.exception("Error closing %s", type(store).__name__)


def _retry(fn: Callable[[], Any], what: str, attempts: int = 15, delay: float = 3.0) -> None:
  """Databases in containers/cloud often come up after the API; wait for them."""
  for i in range(1, attempts + 1):
    try:
      fn()
      return
    except Exception as exc:
      if i == attempts:
        raise
      logger.warning("%s not ready (%s: %s); retry %d/%d", what, type(exc).__name__, exc, i, attempts)
      time.sleep(delay)


def build_services(settings: Optional[Settings] = None) -> Services:
  from app.evaluation import RagasEvaluator
  from app.graph import build_graph
  from app.ingestion import Ingestor
  from app.llm import get_chat_model, get_embeddings
  from app.stores.graph_store import Neo4jGraphStore
  from app.stores.vector_store import PgVectorStore

  settings = settings or get_settings()
  if not settings.openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

  vector_store = PgVectorStore(settings.postgres_dsn, settings.embedding_dim)
  graph_store = Neo4jGraphStore(
      settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, settings.neo4j_database
  )
  _retry(vector_store.init, "PostgreSQL")
  _retry(graph_store.init, "Neo4j")

  llm = get_chat_model(settings)
  embeddings = get_embeddings(settings)
  evaluator = RagasEvaluator(lambda: get_chat_model(settings))
  ingestor = Ingestor(
      vector_store, graph_store, embeddings, llm, settings.chunk_size, settings.chunk_overlap
  )
  graph = build_graph(
      settings=settings, vector_store=vector_store, graph_store=graph_store,
      embeddings=embeddings, llm=llm, evaluator=evaluator,
  )
  return Services(settings, vector_store, graph_store, embeddings, llm, ingestor, graph, evaluator)
