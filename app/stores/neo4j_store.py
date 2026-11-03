"""Neo4j knowledge-graph store.

Graph model::

    (:Document {id,title})-[:HAS_CHUNK]->(:Chunk {id,document_id,index})
    (:Chunk)-[:MENTIONS]->(:Entity {id,name,type,description})
    (:Entity)-[:RELATED {type,description,weight,chunk_ids}]->(:Entity)

Two transports are supported so the engine works with only the packages in
``requirements.txt``: the official Bolt driver (used when the ``neo4j``
package is installed) and Neo4j's HTTP transactional endpoint (stdlib only).
"""

from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from typing import Any

from app.models import Chunk, Entity, GraphFact, Relation

logger = logging.getLogger(__name__)


class Neo4jError(RuntimeError):
    pass


class _HttpTransport:
    def __init__(self, base_url: str, user: str, password: str, database: str, timeout: float):
        self.url = f"{base_url.rstrip('/')}/db/{database}/tx/commit"
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
            "Accept": "application/json;charset=UTF-8",
        }
        self.timeout = timeout

    def run(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        body = json.dumps(
            {"statements": [{"statement": cypher, "parameters": params or {}}]}
        ).encode()
        req = urllib.request.Request(self.url, data=body, headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raise Neo4jError(f"Neo4j HTTP {exc.code}: {exc.read()[:300]!r}") from exc
        except urllib.error.URLError as exc:
            raise Neo4jError(f"Neo4j unreachable at {self.url}: {exc.reason}") from exc
        if payload.get("errors"):
            raise Neo4jError("; ".join(e.get("message", "") for e in payload["errors"]))
        results = payload.get("results") or [{}]
        cols = results[0].get("columns", [])
        return [dict(zip(cols, d["row"])) for d in results[0].get("data", [])]

    def close(self) -> None:
        pass


class _BoltTransport:
    def __init__(self, uri: str, user: str, password: str, database: str):
        from neo4j import GraphDatabase  # optional dependency

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def run(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(
            cypher, parameters_=params or {}, database_=self.database
        )
        return [r.data() for r in records]

    def close(self) -> None:
        self.driver.close()


class Neo4jGraphStore:
    def __init__(
        self,
        uri: str,
        http_url: str,
        user: str,
        password: str,
        database: str = "neo4j",
        transport: str = "auto",
        timeout: float = 30.0,
    ):
        self._tx: _HttpTransport | _BoltTransport
        use_bolt = transport == "bolt"
        if transport == "auto":
            try:
                import neo4j  # noqa: F401

                use_bolt = True
            except ImportError:
                use_bolt = False
        if use_bolt:
            self._tx = _BoltTransport(uri, user, password, database)
            self.transport = "bolt"
        else:
            self._tx = _HttpTransport(http_url, user, password, database, timeout)
            self.transport = "http"
        logger.info("Neo4j transport: %s", self.transport)

    def run(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self._tx.run(cypher, params)

    def close(self) -> None:
        self._tx.close()

    def health(self) -> bool:
        return self.run("RETURN 1 AS ok")[0]["ok"] == 1

    def init_schema(self) -> None:
        for stmt in (
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        ):
            self.run(stmt)
        logger.info("Neo4j schema ready")

    # -- writes -------------------------------------------------------------
    def upsert_document_graph(
        self,
        doc_id: str,
        title: str,
        chunks: list[Chunk],
        entities: list[Entity],
        relations: list[Relation],
        mentions: list[tuple[str, str]],
    ) -> None:
        self.delete_document(doc_id)  # idempotent re-ingestion
        self.run(
            """MERGE (d:Document {id: $doc_id}) SET d.title = $title
               WITH d
               UNWIND $chunks AS ch
               MERGE (c:Chunk {id: ch.id})
               SET c.document_id = $doc_id, c.index = ch.index
               MERGE (d)-[:HAS_CHUNK]->(c)""",
            {
                "doc_id": doc_id,
                "title": title,
                "chunks": [{"id": c.id, "index": c.index} for c in chunks],
            },
        )
        if entities:
            self.run(
                """UNWIND $entities AS e
                   MERGE (n:Entity {id: e.id})
                   ON CREATE SET n.name = e.name, n.type = e.type, n.description = e.description
                   ON MATCH SET n.description = CASE
                       WHEN size(coalesce(n.description, '')) < size(e.description)
                       THEN e.description ELSE n.description END""",
                {
                    "entities": [
                        {"id": e.id, "name": e.name, "type": e.type, "description": e.description}
                        for e in entities
                    ]
                },
            )
        if mentions:
            self.run(
                """UNWIND $mentions AS m
                   MATCH (c:Chunk {id: m.chunk_id})
                   MATCH (n:Entity {id: m.entity_id})
                   MERGE (c)-[:MENTIONS]->(n)""",
                {"mentions": [{"chunk_id": c, "entity_id": e} for c, e in mentions]},
            )
        if relations:
            self.run(
                """UNWIND $rels AS r
                   MATCH (a:Entity {id: r.source})
                   MATCH (b:Entity {id: r.target})
                   MERGE (a)-[x:RELATED {type: r.type}]->(b)
                   ON CREATE SET x.description = r.description, x.weight = r.weight,
                                 x.chunk_ids = r.chunk_ids
                   ON MATCH SET x.weight = coalesce(x.weight, 0) + r.weight,
                                x.chunk_ids = coalesce(x.chunk_ids, []) +
                                    [cid IN r.chunk_ids WHERE NOT cid IN coalesce(x.chunk_ids, [])],
                                x.description = CASE
                                    WHEN size(coalesce(x.description, '')) < size(r.description)
                                    THEN r.description ELSE x.description END""",
                {
                    "rels": [
                        {
                            "source": r.source,
                            "target": r.target,
                            "type": r.type,
                            "description": r.description,
                            "weight": r.weight,
                            "chunk_ids": r.chunk_ids,
                        }
                        for r in relations
                    ]
                },
            )

    def delete_document(self, doc_id: str) -> None:
        # 1. strip this document's chunk ids from relationship provenance
        self.run(
            """MATCH (:Entity)-[r:RELATED]->(:Entity)
               WHERE any(cid IN r.chunk_ids WHERE cid STARTS WITH $prefix)
               SET r.chunk_ids = [cid IN r.chunk_ids WHERE NOT cid STARTS WITH $prefix]
               WITH r WHERE size(r.chunk_ids) = 0
               DELETE r""",
            {"prefix": f"{doc_id}::"},
        )
        # 2. remove chunks and the document node
        self.run(
            """MATCH (d:Document {id: $doc_id})
               OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
               DETACH DELETE c, d""",
            {"doc_id": doc_id},
        )
        # 3. drop orphaned entities
        self.run(
            """MATCH (e:Entity)
               WHERE NOT (e)<-[:MENTIONS]-(:Chunk) AND NOT (e)-[:RELATED]-()
               DETACH DELETE e"""
        )

    # -- reads --------------------------------------------------------------
    def expand(
        self, seed_ids: list[str], max_hops: int, max_facts: int
    ) -> tuple[list[GraphFact], dict[str, float]]:
        """Multi-hop traversal around seed entities.

        Returns ranked graph facts plus a chunk-id -> relevance score map that
        points at the source passages supporting those facts.
        """
        if not seed_ids:
            return [], {}
        hops = max(1, min(int(max_hops), 4))  # bounded: variable-length can't be a parameter
        rows = self.run(
            f"""MATCH (s:Entity) WHERE s.id IN $ids
                MATCH p = (s)-[:RELATED*1..{hops}]-(:Entity)
                WITH p LIMIT $path_limit
                UNWIND range(0, length(p) - 1) AS i
                WITH relationships(p)[i] AS r, i + 1 AS hop
                WITH r, min(hop) AS hop
                RETURN startNode(r).name AS source, r.type AS relation,
                       endNode(r).name AS target, r.description AS description,
                       hop, coalesce(r.weight, 1.0) AS weight,
                       coalesce(r.chunk_ids, []) AS chunk_ids
                ORDER BY hop ASC, weight DESC
                LIMIT $limit""",
            {"ids": seed_ids, "path_limit": max_facts * 25, "limit": max_facts},
        )
        facts = [
            GraphFact(
                source=r["source"],
                relation=r["relation"],
                target=r["target"],
                description=r.get("description") or "",
                hop=int(r["hop"]),
                weight=float(r["weight"]),
                chunk_ids=list(r["chunk_ids"]),
            )
            for r in rows
        ]
        mention_rows = self.run(
            """MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE e.id IN $ids
               RETURN c.id AS chunk_id, count(DISTINCT e) AS seeds
               ORDER BY seeds DESC LIMIT 50""",
            {"ids": seed_ids},
        )
        chunk_scores: dict[str, float] = {}
        for row in mention_rows:
            chunk_scores[row["chunk_id"]] = chunk_scores.get(row["chunk_id"], 0.0) + float(
                row["seeds"]
            )
        for f in facts:
            for cid in f.chunk_ids:
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + f.weight / f.hop
        return facts, chunk_scores

    def neighbourhood(self, entity_id: str, limit: int = 50) -> dict[str, Any]:
        rows = self.run(
            """MATCH (e:Entity {id: $id})
               OPTIONAL MATCH (e)-[r:RELATED]-(n:Entity)
               RETURN e {.id, .name, .type, .description} AS entity,
                      collect(DISTINCT {relation: r.type, direction:
                          CASE WHEN startNode(r) = e THEN 'out' ELSE 'in' END,
                          neighbour: n {.id, .name, .type}})[0..$limit] AS edges""",
            {"id": entity_id, "limit": limit},
        )
        if not rows or rows[0]["entity"] is None:
            return {}
        out = rows[0]
        out["edges"] = [e for e in out["edges"] if e.get("neighbour")]
        return out

    def stats(self) -> dict[str, int]:
        # Separate statements keep this compatible with Neo4j 4.4 and 5.x.
        return {
            "entities": self.run("MATCH (e:Entity) RETURN count(e) AS n")[0]["n"],
            "relations": self.run("MATCH ()-[r:RELATED]->() RETURN count(r) AS n")[0]["n"],
            "documents": self.run("MATCH (d:Document) RETURN count(d) AS n")[0]["n"],
        }
