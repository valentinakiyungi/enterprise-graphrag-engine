"""Dependency container wiring settings, models, stores and pipelines together."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.evaluation.evaluator import Evaluator
from app.generation import AnswerGenerator
from app.ingestion.pipeline import IngestionPipeline
from app.llm import LLMProtocol
from app.retrieval.query_analyzer import QueryAnalyzer
from app.retrieval.retrievers import GraphRetriever, VectorRetriever

logger = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    llm: LLMProtocol
    pg: Any
    graph_store: Any
    evaluator: Evaluator
    analyzer: QueryAnalyzer = field(init=False)
    vector_retriever: VectorRetriever = field(init=False)
    graph_retriever: GraphRetriever = field(init=False)
    generator: AnswerGenerator = field(init=False)
    ingestion: IngestionPipeline = field(init=False)
    _workflow: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.analyzer = QueryAnalyzer(self.llm)
        self.vector_retriever = VectorRetriever(self.settings, self.llm, self.pg)
        self.graph_retriever = GraphRetriever(self.settings, self.llm, self.pg, self.graph_store)
        self.generator = AnswerGenerator(self.llm)
        self.ingestion = IngestionPipeline(self.settings, self.llm, self.pg, self.graph_store)

    @property
    def workflow(self):
        if self._workflow is None:
            from app.graph import build_graph

            self._workflow = build_graph(self)
        return self._workflow

    def close(self) -> None:
        for closer in (getattr(self.pg, "close", None), getattr(self.graph_store, "close", None)):
            if closer:
                try:
                    closer()
                except Exception as exc:  # pragma: no cover
                    logger.warning("Error during shutdown: %s", exc)


def create_services(settings: Settings, init_schema: bool = True) -> Services:
    """Build production services backed by OpenAI, PostgreSQL and Neo4j."""
    from app.llm import LLMClient
    from app.stores.neo4j_store import Neo4jGraphStore
    from app.stores.postgres import PgVectorStore

    llm = LLMClient(settings)
    judge_llm = (
        llm if settings.eval_model == settings.openai_chat_model
        else LLMClient(settings, model=settings.eval_model)
    )
    pg = PgVectorStore(
        settings.database_url, settings.embedding_dim, settings.pg_pool_min, settings.pg_pool_max
    )
    graph_store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        http_url=settings.neo4j_http_url,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        transport=settings.neo4j_transport,
    )
    if init_schema:
        pg.init_schema()
        graph_store.init_schema()
    return Services(
        settings=settings,
        llm=llm,
        pg=pg,
        graph_store=graph_store,
        evaluator=Evaluator(settings, judge_llm),
    )
