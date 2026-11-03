"""PostgreSQL + pgvector persistence.

Holds documents, chunks (with dense embeddings and a full-text index), entity
embeddings used for graph entity-linking, query logs and evaluation runs.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from app.models import Chunk, Entity, RetrievedChunk

logger = logging.getLogger(__name__)


def to_pgvector(vec: list[float]) -> str:
    """Serialise a Python vector to pgvector's text representation."""
    return "[" + ",".join(f"{float(x):.7g}" for x in vec) + "]"


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    source      TEXT,
    metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    chunk_count INT NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    embedding   vector({dim}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);

CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    embedding   vector({dim}) NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS entities_embedding_hnsw
    ON entities USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS entities_name_lower ON entities (lower(name));

CREATE TABLE IF NOT EXISTS chunk_entities (
    chunk_id  TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (chunk_id, entity_id)
);

CREATE TABLE IF NOT EXISTS query_logs (
    id          UUID PRIMARY KEY,
    query       TEXT NOT NULL,
    mode        TEXT NOT NULL,
    answer      TEXT,
    evaluation  JSONB,
    attempts    INT,
    latency_ms  INT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id          UUID PRIMARY KEY,
    name        TEXT,
    status      TEXT NOT NULL,
    config      JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    summary     JSONB,
    results     JSONB,
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);
"""


class PgVectorStore:
    def __init__(self, dsn: str, embedding_dim: int, pool_min: int = 1, pool_max: int = 10):
        from psycopg2.pool import ThreadedConnectionPool

        self.dim = embedding_dim
        self.pool = ThreadedConnectionPool(pool_min, pool_max, dsn)

    # -- plumbing -------------------------------------------------------------
    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        from psycopg2.extras import RealDictCursor

        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.pool.putconn(conn)

    def close(self) -> None:
        self.pool.closeall()

    def init_schema(self) -> None:
        with self._cursor() as cur:
            cur.execute(SCHEMA_SQL.format(dim=self.dim))
            cur.execute(
                """SELECT atttypmod FROM pg_attribute
                   WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"""
            )
            row = cur.fetchone()
            if row and row["atttypmod"] not in (-1, self.dim):
                raise RuntimeError(
                    f"chunks.embedding has dimension {row['atttypmod']} but EMBEDDING_DIM="
                    f"{self.dim}. Re-create the database or align the setting."
                )
        logger.info("PostgreSQL schema ready (dim=%s)", self.dim)

    def health(self) -> bool:
        with self._cursor() as cur:
            cur.execute("SELECT 1 AS ok")
            return cur.fetchone()["ok"] == 1

    # -- documents & chunks -------------------------------------------------
    def upsert_document(
        self, doc_id: str, title: str, source: str | None, metadata: dict, chunks: list[Chunk]
    ) -> None:
        from psycopg2.extras import execute_values

        with self._cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
            cur.execute(
                """INSERT INTO documents (id, title, source, metadata, chunk_count)
                   VALUES (%s, %s, %s, %s::jsonb, %s)""",
                (doc_id, title, source, json.dumps(metadata), len(chunks)),
            )
            execute_values(
                cur,
                """INSERT INTO chunks (id, document_id, chunk_index, content, metadata, embedding)
                   VALUES %s""",
                [
                    (
                        c.id,
                        doc_id,
                        c.index,
                        c.content,
                        json.dumps(c.metadata),
                        to_pgvector(c.embedding or []),
                    )
                    for c in chunks
                ],
                template="(%s, %s, %s, %s, %s::jsonb, %s::vector)",
            )

    def list_documents(self, limit: int = 100, offset: int = 0) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                """SELECT id, title, source, metadata, chunk_count, created_at
                   FROM documents ORDER BY created_at DESC LIMIT %s OFFSET %s""",
                (limit, offset),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_document(self, doc_id: str) -> dict | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def delete_document(self, doc_id: str) -> bool:
        with self._cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
            deleted = cur.rowcount > 0
            # Entities no longer mentioned by any chunk are orphans.
            cur.execute(
                """DELETE FROM entities e WHERE NOT EXISTS
                   (SELECT 1 FROM chunk_entities ce WHERE ce.entity_id = e.id)"""
            )
            return deleted

    # -- retrieval ----------------------------------------------------------
    def vector_search(self, embedding: list[float], k: int) -> list[RetrievedChunk]:
        with self._cursor() as cur:
            cur.execute(
                """SELECT c.id, c.document_id, c.content, c.metadata, d.title,
                          1 - (c.embedding <=> %s::vector) AS score
                   FROM chunks c JOIN documents d ON d.id = c.document_id
                   ORDER BY c.embedding <=> %s::vector
                   LIMIT %s""",
                (to_pgvector(embedding), to_pgvector(embedding), k),
            )
            return [self._row_to_chunk(r, "vector") for r in cur.fetchall()]

    def keyword_search(self, query: str, k: int) -> list[RetrievedChunk]:
        with self._cursor() as cur:
            cur.execute(
                """SELECT c.id, c.document_id, c.content, c.metadata, d.title,
                          ts_rank_cd(c.tsv, q) AS score
                   FROM chunks c JOIN documents d ON d.id = c.document_id,
                        websearch_to_tsquery('english', %s) q
                   WHERE c.tsv @@ q
                   ORDER BY score DESC
                   LIMIT %s""",
                (query, k),
            )
            return [self._row_to_chunk(r, "keyword") for r in cur.fetchall()]

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, RetrievedChunk]:
        if not chunk_ids:
            return {}
        with self._cursor() as cur:
            cur.execute(
                """SELECT c.id, c.document_id, c.content, c.metadata, d.title, 0.0 AS score
                   FROM chunks c JOIN documents d ON d.id = c.document_id
                   WHERE c.id = ANY(%s)""",
                (list(chunk_ids),),
            )
            return {r["id"]: self._row_to_chunk(r, "graph") for r in cur.fetchall()}

    @staticmethod
    def _row_to_chunk(r: dict, retriever: str) -> RetrievedChunk:
        return RetrievedChunk(
            id=r["id"],
            document_id=r["document_id"],
            content=r["content"],
            score=float(r["score"] or 0.0),
            retriever=retriever,
            title=r.get("title") or "",
            metadata=r.get("metadata") or {},
        )

    # -- entities -----------------------------------------------------------
    def upsert_entities(self, entities: list[Entity]) -> None:
        from psycopg2.extras import execute_values

        if not entities:
            return
        with self._cursor() as cur:
            execute_values(
                cur,
                """INSERT INTO entities (id, name, type, description, embedding)
                   VALUES %s
                   ON CONFLICT (id) DO UPDATE SET
                     description = CASE
                        WHEN length(EXCLUDED.description) > length(entities.description)
                        THEN EXCLUDED.description ELSE entities.description END,
                     embedding = CASE
                        WHEN length(EXCLUDED.description) > length(entities.description)
                        THEN EXCLUDED.embedding ELSE entities.embedding END,
                     updated_at = now()""",
                [
                    (e.id, e.name, e.type, e.description, to_pgvector(e.embedding or []))
                    for e in entities
                ],
                template="(%s, %s, %s, %s, %s::vector)",
            )

    def link_chunk_entities(self, pairs: list[tuple[str, str]]) -> None:
        from psycopg2.extras import execute_values

        if not pairs:
            return
        with self._cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO chunk_entities (chunk_id, entity_id) VALUES %s ON CONFLICT DO NOTHING",
                pairs,
            )

    def existing_entity_ids(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        with self._cursor() as cur:
            cur.execute("SELECT id FROM entities WHERE id = ANY(%s)", (list(ids),))
            return {r["id"] for r in cur.fetchall()}

    def search_entities(
        self, embedding: list[float], k: int, max_distance: float
    ) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                """SELECT id, name, type, description, embedding <=> %s::vector AS distance
                   FROM entities
                   ORDER BY embedding <=> %s::vector
                   LIMIT %s""",
                (to_pgvector(embedding), to_pgvector(embedding), k),
            )
            return [dict(r) for r in cur.fetchall() if r["distance"] <= max_distance]

    def find_entities_by_name(self, names: list[str]) -> list[dict]:
        if not names:
            return []
        with self._cursor() as cur:
            cur.execute(
                """SELECT id, name, type, description, 0.0 AS distance
                   FROM entities WHERE lower(name) = ANY(%s)""",
                ([n.lower() for n in names],),
            )
            return [dict(r) for r in cur.fetchall()]

    # -- logs & eval runs ---------------------------------------------------
    def log_query(
        self,
        query: str,
        mode: str,
        answer: str,
        evaluation: dict | None,
        attempts: int,
        latency_ms: int,
    ) -> str:
        qid = str(uuid.uuid4())
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO query_logs
                   (id, query, mode, answer, evaluation, attempts, latency_ms)
                   VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)""",
                (qid, query, mode, answer, json.dumps(evaluation or {}), attempts, latency_ms),
            )
        return qid

    def create_eval_run(self, name: str | None, config: dict) -> str:
        rid = str(uuid.uuid4())
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO eval_runs (id, name, status, config) VALUES (%s, %s, 'running', %s::jsonb)",
                (rid, name, json.dumps(config)),
            )
        return rid

    def finish_eval_run(
        self, run_id: str, summary: dict | None, results: list | None, error: str | None = None
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """UPDATE eval_runs SET status = %s, summary = %s::jsonb, results = %s::jsonb,
                          error = %s, finished_at = now() WHERE id = %s""",
                (
                    "failed" if error else "completed",
                    json.dumps(summary) if summary is not None else None,
                    json.dumps(results) if results is not None else None,
                    error,
                    run_id,
                ),
            )

    def list_eval_runs(self, limit: int = 20) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                """SELECT id, name, status, config, summary, error, created_at, finished_at
                   FROM eval_runs ORDER BY created_at DESC LIMIT %s""",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_eval_run(self, run_id: str) -> dict | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM eval_runs WHERE id = %s", (run_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def stats(self) -> dict[str, Any]:
        with self._cursor() as cur:
            cur.execute(
                """SELECT (SELECT count(*) FROM documents) AS documents,
                          (SELECT count(*) FROM chunks) AS chunks,
                          (SELECT count(*) FROM entities) AS entities,
                          (SELECT count(*) FROM query_logs) AS queries,
                          (SELECT avg((evaluation->>'faithfulness')::float)
                             FROM query_logs WHERE evaluation ? 'faithfulness') AS avg_faithfulness"""
            )
            return dict(cur.fetchone())
