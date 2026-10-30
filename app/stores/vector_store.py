"""PostgreSQL + pgvector chunk store."""

from typing import Any, Dict, List, Optional, Sequence


class PgVectorStore:

  def __init__(self, dsn: str, dim: int = 1536):
    self.dsn = dsn
    self.dim = int(dim)
    self._pool = None

  def init(self) -> None:
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg_pool import ConnectionPool

    # The extension must exist before register_vector() runs on pooled connections.
    with psycopg.connect(self.dsn, autocommit=True) as conn:
      conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
      conn.execute(
          "CREATE TABLE IF NOT EXISTS chunks ("
          " id BIGSERIAL PRIMARY KEY,"
          " doc_id TEXT NOT NULL,"
          " chunk_index INT NOT NULL,"
          " content TEXT NOT NULL,"
          " metadata JSONB NOT NULL DEFAULT '{}',"
          f" embedding vector({self.dim}) NOT NULL,"
          " UNIQUE (doc_id, chunk_index))"
      )
      conn.execute(
          "CREATE INDEX IF NOT EXISTS chunks_embedding_idx"
          " ON chunks USING hnsw (embedding vector_cosine_ops)"
      )

    self._pool = ConnectionPool(
        self.dsn,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True},
        configure=register_vector,
        open=True,
    )
    self._pool.wait(timeout=30)

  def add_chunks(
      self,
      doc_id: str,
      chunks: Sequence[str],
      embeddings: Sequence[Sequence[float]],
      metadata: Optional[Dict[str, Any]] = None,
  ) -> List[int]:
    """Replace all chunks of doc_id; returns the new chunk ids in order."""
    from pgvector import Vector
    from psycopg.types.json import Jsonb

    ids: List[int] = []
    with self._pool.connection() as conn, conn.transaction():
      conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
      for i, (content, emb) in enumerate(zip(chunks, embeddings)):
        row = conn.execute(
            "INSERT INTO chunks (doc_id, chunk_index, content, metadata, embedding)"
            " VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (doc_id, i, content, Jsonb(metadata or {}), Vector(list(emb))),
        ).fetchone()
        ids.append(int(row[0]))
    return ids

  def search(self, embedding: Sequence[float], k: int = 5) -> List[Dict[str, Any]]:
    from pgvector import Vector

    vec = Vector(list(embedding))
    with self._pool.connection() as conn:
      rows = conn.execute(
          "SELECT id, doc_id, chunk_index, content, 1 - (embedding <=> %s) AS score"
          " FROM chunks ORDER BY embedding <=> %s LIMIT %s",
          (vec, vec, k),
      ).fetchall()
    return [
        {"id": r[0], "doc_id": r[1], "chunk_index": r[2], "content": r[3], "score": float(r[4])}
        for r in rows
    ]

  def get_chunks(self, ids: Sequence[int]) -> List[Dict[str, Any]]:
    if not ids:
      return []
    with self._pool.connection() as conn:
      rows = conn.execute(
          "SELECT id, doc_id, chunk_index, content FROM chunks WHERE id = ANY(%s)",
          (list(ids),),
      ).fetchall()
    by_id = {
        r[0]: {"id": r[0], "doc_id": r[1], "chunk_index": r[2], "content": r[3], "score": None}
        for r in rows
    }
    return [by_id[i] for i in ids if i in by_id]  # keep caller's ordering

  def stats(self) -> Dict[str, int]:
    with self._pool.connection() as conn:
      chunks, docs = conn.execute(
          "SELECT COUNT(*), COUNT(DISTINCT doc_id) FROM chunks"
      ).fetchone()
    return {"chunks": int(chunks), "documents": int(docs)}

  def ping(self) -> None:
    with self._pool.connection() as conn:
      conn.execute("SELECT 1")

  def close(self) -> None:
    if self._pool is not None:
      self._pool.close()
