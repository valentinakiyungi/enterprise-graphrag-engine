"""Neo4j knowledge-graph store.

Schema: (:Entity {key, name}) -[:RELATED {predicate, doc_id, chunk_id}]-> (:Entity)
A single generic relationship type keeps Cypher free of dynamic labels (no injection
surface); the human-readable relation lives in the `predicate` property. Every edge
remembers the Postgres chunk it was extracted from, so graph hits can be traced back
to source text.
"""

from typing import Any, Dict, List, Sequence

from app.text import normalize_entity


class Neo4jGraphStore:

  def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
    self.uri, self.user, self.password, self.database = uri, user, password, database
    self._driver = None

  def init(self) -> None:
    from neo4j import GraphDatabase

    self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
    self._driver.verify_connectivity()
    with self._driver.session(database=self.database) as s:
      s.run("CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE").consume()

  def add_triples(self, doc_id: str, triples: Sequence[Dict[str, Any]]) -> int:
    """triples: dicts with subject, predicate, object, chunk_id."""
    rows = []
    for t in triples:
      s_key, o_key = normalize_entity(t["subject"]), normalize_entity(t["object"])
      if not s_key or not o_key or s_key == o_key:
        continue
      rows.append({
          "s_key": s_key, "s_name": t["subject"].strip(),
          "o_key": o_key, "o_name": t["object"].strip(),
          "predicate": t["predicate"].strip().lower()[:80],
          "chunk_id": int(t["chunk_id"]),
      })
    if not rows:
      return 0
    query = """
    UNWIND $rows AS row
    MERGE (a:Entity {key: row.s_key}) ON CREATE SET a.name = row.s_name
    MERGE (b:Entity {key: row.o_key}) ON CREATE SET b.name = row.o_name
    MERGE (a)-[:RELATED {predicate: row.predicate, doc_id: $doc_id, chunk_id: row.chunk_id}]->(b)
    """
    with self._driver.session(database=self.database) as s:
      s.execute_write(lambda tx: tx.run(query, rows=rows, doc_id=doc_id).consume())
    return len(rows)

  def delete_document(self, doc_id: str) -> None:
    with self._driver.session(database=self.database) as s:
      s.run("MATCH ()-[r:RELATED {doc_id: $doc_id}]->() DELETE r", doc_id=doc_id).consume()
      s.run("MATCH (e:Entity) WHERE NOT (e)--() DELETE e").consume()

  def resolve_entities(self, keys: Sequence[str], limit: int = 20) -> List[str]:
    """Map normalized query-entity keys to entity keys that exist in the graph."""
    keys = [k for k in keys if k]
    if not keys:
      return []
    query = """
    MATCH (e:Entity)
    WHERE e.key IN $keys
       OR any(k IN $keys WHERE size(k) >= 3 AND size(e.key) >= 3
              AND (e.key CONTAINS k OR k CONTAINS e.key))
    RETURN e.key AS key LIMIT $limit
    """
    with self._driver.session(database=self.database) as s:
      return [r["key"] for r in s.run(query, keys=list(keys), limit=limit)]

  def expand(self, keys: Sequence[str], max_hops: int = 2, limit: int = 40) -> List[Dict[str, Any]]:
    """Facts within max_hops of the seed entities, nearest first."""
    if not keys:
      return []
    hops = max(1, min(int(max_hops), 4))  # validated int, safe to format into Cypher
    query = f"""
    MATCH (s:Entity) WHERE s.key IN $keys
    MATCH p = (s)-[:RELATED*1..{hops}]-(:Entity)
    UNWIND relationships(p) AS r
    WITH r, min(length(p)) AS dist
    RETURN startNode(r).name AS subject, r.predicate AS predicate,
           endNode(r).name AS object, r.chunk_id AS chunk_id, r.doc_id AS doc_id
    ORDER BY dist
    LIMIT $limit
    """
    with self._driver.session(database=self.database) as s:
      return [dict(r) for r in s.run(query, keys=list(keys), limit=limit)]

  def stats(self) -> Dict[str, int]:
    with self._driver.session(database=self.database) as s:
      e = s.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
      r = s.run("MATCH ()-[r:RELATED]->() RETURN count(r) AS c").single()["c"]
    return {"entities": int(e), "relationships": int(r)}

  def ping(self) -> None:
    self._driver.verify_connectivity()

  def close(self) -> None:
    if self._driver is not None:
      self._driver.close()
