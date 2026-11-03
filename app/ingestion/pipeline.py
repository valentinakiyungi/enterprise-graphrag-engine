"""Document ingestion: chunk -> embed -> extract graph -> persist to both stores."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.ingestion.chunker import chunk_text
from app.ingestion.extractor import GraphExtractor
from app.llm import LLMProtocol
from app.models import Chunk

logger = logging.getLogger(__name__)


def make_document_id(title: str, content: str) -> str:
    digest = hashlib.sha256(f"{title}\n{content}".encode()).hexdigest()
    return f"doc_{digest[:16]}"


@dataclass
class IngestionResult:
    document_id: str
    title: str
    chunks: int
    entities: int
    relations: int
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class IngestionPipeline:
    def __init__(self, settings: Settings, llm: LLMProtocol, pg, graph):
        self.settings = settings
        self.llm = llm
        self.pg = pg
        self.graph = graph
        self.extractor = GraphExtractor(llm, settings.extraction_concurrency)

    def ingest(
        self,
        title: str,
        content: str,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        document_id: str | None = None,
    ) -> IngestionResult:
        started = time.perf_counter()
        metadata = metadata or {}
        doc_id = document_id or make_document_id(title, content)

        pieces = chunk_text(content, self.settings.chunk_size, self.settings.chunk_overlap)
        if not pieces:
            raise ValueError("Document has no text content")
        chunks = [
            Chunk(
                id=f"{doc_id}::{i}",
                document_id=doc_id,
                index=i,
                content=p,
                metadata={"title": title, "source": source, **metadata},
            )
            for i, p in enumerate(pieces)
        ]

        # 1. dense embeddings (contextualised with the title for better recall)
        vectors = self.llm.embed([f"{title}\n\n{c.content}" for c in chunks])
        for c, v in zip(chunks, vectors):
            c.embedding = v

        # 2. knowledge-graph extraction
        extraction = self.extractor.extract(title, chunks)
        entities = list(extraction.entities.values())
        relations = list(extraction.relations.values())
        if entities:
            ent_vectors = self.llm.embed(
                [f"{e.name} ({e.type}): {e.description}" for e in entities]
            )
            for e, v in zip(entities, ent_vectors):
                e.embedding = v

        # 3. persist: Postgres (chunks, entity vectors) then Neo4j (graph)
        self.pg.upsert_document(doc_id, title, source, metadata, chunks)
        self.pg.upsert_entities(entities)
        self.pg.link_chunk_entities(sorted(extraction.mentions))
        try:
            self.graph.upsert_document_graph(
                doc_id, title, chunks, entities, relations, sorted(extraction.mentions)
            )
        except Exception:
            logger.exception("Graph write failed; rolling back document %s", doc_id)
            self.pg.delete_document(doc_id)
            raise

        result = IngestionResult(
            document_id=doc_id,
            title=title,
            chunks=len(chunks),
            entities=len(entities),
            relations=len(relations),
            seconds=round(time.perf_counter() - started, 2),
        )
        logger.info("Ingested %s", result.to_dict())
        return result

    def delete(self, doc_id: str) -> bool:
        self.graph.delete_document(doc_id)
        return self.pg.delete_document(doc_id)
