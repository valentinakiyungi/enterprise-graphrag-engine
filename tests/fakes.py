"""In-memory test doubles for the LLM, PostgreSQL and Neo4j layers."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Any, Callable

from app.config import Settings
from app.models import GraphFact, RetrievedChunk


def hash_embed(text: str, dim: int = 128) -> list[float]:
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine_distance(a: list[float], b: list[float]) -> float:
    return 1.0 - sum(x * y for x, y in zip(a, b))


class FakeLLM:
    def __init__(
        self,
        structured: Callable[[str, str, type], Any] | None = None,
        answer: Callable[[str, str], str] | None = None,
    ):
        self.structured = structured
        self.answer = answer or (lambda system, user: "Stub answer [S1].")
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, temperature: float | None = None) -> str:
        self.calls.append(("complete", user[:60]))
        return self.answer(system, user)

    def complete_structured(self, system: str, user: str, schema: type):
        self.calls.append((schema.__name__, user[:60]))
        if self.structured:
            out = self.structured(system, user, schema)
            if out is not None:
                return out
        return schema()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return hash_embed(text)


class FakePg:
    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.chunks: dict[str, Any] = {}
        self.entities: dict[str, Any] = {}
        self.chunk_entities: set[tuple[str, str]] = set()
        self.logs: list[dict] = []
        self.runs: dict[str, dict] = {}

    def upsert_document(self, doc_id, title, source, metadata, chunks):
        self.delete_document(doc_id)
        self.docs[doc_id] = {"id": doc_id, "title": title, "source": source,
                             "metadata": metadata, "chunk_count": len(chunks),
                             "created_at": "2026-11-03T00:00:00Z"}
        for c in chunks:
            self.chunks[c.id] = c

    def delete_document(self, doc_id):
        existed = self.docs.pop(doc_id, None) is not None
        for cid in [c for c, ch in self.chunks.items() if ch.document_id == doc_id]:
            del self.chunks[cid]
        self.chunk_entities = {p for p in self.chunk_entities if p[0] in self.chunks}
        live = {e for _, e in self.chunk_entities}
        self.entities = {k: v for k, v in self.entities.items() if k in live}
        return existed

    def _rc(self, c, score, retriever):
        return RetrievedChunk(id=c.id, document_id=c.document_id, content=c.content,
                              score=score, retriever=retriever,
                              title=self.docs[c.document_id]["title"])

    def vector_search(self, emb, k):
        ranked = sorted(self.chunks.values(), key=lambda c: cosine_distance(emb, c.embedding))
        return [self._rc(c, 1 - cosine_distance(emb, c.embedding), "vector") for c in ranked[:k]]

    def keyword_search(self, query, k):
        terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if t != "or" and len(t) > 2}
        scored = []
        for c in self.chunks.values():
            words = set(re.findall(r"[a-z0-9]+", c.content.lower()))
            s = len(terms & words)
            if s:
                scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        return [self._rc(c, float(s), "keyword") for s, c in scored[:k]]

    def get_chunks(self, ids):
        return {i: self._rc(self.chunks[i], 0.0, "graph") for i in ids if i in self.chunks}

    def upsert_entities(self, entities):
        for e in entities:
            self.entities[e.id] = e

    def link_chunk_entities(self, pairs):
        self.chunk_entities.update(pairs)

    def search_entities(self, emb, k, max_distance):
        rows = [
            {"id": e.id, "name": e.name, "type": e.type, "description": e.description,
             "distance": cosine_distance(emb, e.embedding)}
            for e in self.entities.values()
        ]
        rows.sort(key=lambda r: r["distance"])
        return [r for r in rows[:k] if r["distance"] <= max_distance]

    def find_entities_by_name(self, names):
        lower = {n.lower() for n in names}
        return [{"id": e.id, "name": e.name, "type": e.type, "description": e.description,
                 "distance": 0.0} for e in self.entities.values() if e.name.lower() in lower]

    def log_query(self, query, mode, answer, evaluation, attempts, latency_ms):
        self.logs.append(locals())
        return f"q{len(self.logs)}"

    def list_documents(self, limit=100, offset=0):
        return list(self.docs.values())[offset: offset + limit]

    def get_document(self, doc_id):
        return self.docs.get(doc_id)

    def stats(self):
        return {"documents": len(self.docs), "chunks": len(self.chunks),
                "entities": len(self.entities), "queries": len(self.logs),
                "avg_faithfulness": None}

    def health(self):
        return True

    def create_eval_run(self, name, config):
        rid = f"00000000-0000-0000-0000-{len(self.runs):012d}"
        self.runs[rid] = {"id": rid, "status": "running", "config": config}
        return rid

    def finish_eval_run(self, run_id, summary, results, error=None):
        self.runs[run_id].update(status="failed" if error else "completed",
                                 summary=summary, results=results, error=error)

    def list_eval_runs(self, limit=20):
        return list(self.runs.values())[:limit]

    def get_eval_run(self, run_id):
        return self.runs.get(run_id)

    def close(self):
        pass


class FakeGraph:
    def __init__(self):
        self.relations: list[dict] = []
        self.names: dict[str, str] = {}
        self.mentions: set[tuple[str, str]] = set()

    def upsert_document_graph(self, doc_id, title, chunks, entities, relations, mentions):
        self.delete_document(doc_id)
        for e in entities:
            self.names[e.id] = e.name
        for r in relations:
            self.relations.append({"source": r.source, "target": r.target, "type": r.type,
                                   "description": r.description, "weight": r.weight,
                                   "chunk_ids": list(r.chunk_ids)})
        self.mentions.update(mentions)

    def delete_document(self, doc_id):
        prefix = f"{doc_id}::"
        for r in self.relations:
            r["chunk_ids"] = [c for c in r["chunk_ids"] if not c.startswith(prefix)]
        self.relations = [r for r in self.relations if r["chunk_ids"]]
        self.mentions = {m for m in self.mentions if not m[0].startswith(prefix)}

    def expand(self, seed_ids, max_hops, max_facts):
        adj: dict[str, list[dict]] = defaultdict(list)
        for r in self.relations:
            adj[r["source"]].append(r)
            adj[r["target"]].append(r)
        seen_rel: dict[int, int] = {}
        frontier, visited = set(seed_ids), set(seed_ids)
        for hop in range(1, max_hops + 1):
            nxt = set()
            for node in frontier:
                for r in adj[node]:
                    seen_rel.setdefault(id(r), hop)
                    other = r["target"] if r["source"] == node else r["source"]
                    if other not in visited:
                        visited.add(other)
                        nxt.add(other)
            frontier = nxt
        facts = []
        for r in self.relations:
            if id(r) in seen_rel:
                facts.append(GraphFact(self.names[r["source"]], r["type"], self.names[r["target"]],
                                       r["description"], seen_rel[id(r)], r["weight"],
                                       r["chunk_ids"]))
        facts.sort(key=lambda f: (f.hop, -f.weight))
        facts = facts[:max_facts]
        scores: dict[str, float] = defaultdict(float)
        for cid, eid in self.mentions:
            if eid in seed_ids:
                scores[cid] += 1
        for f in facts:
            for cid in f.chunk_ids:
                scores[cid] += f.weight / f.hop
        return facts, dict(scores)

    def neighbourhood(self, entity_id, limit=50):
        if entity_id not in self.names:
            return {}
        edges = [r for r in self.relations if entity_id in (r["source"], r["target"])]
        return {"entity": {"id": entity_id, "name": self.names[entity_id]},
                "edges": [{"relation": r["type"]} for r in edges[:limit]]}

    def stats(self):
        return {"entities": len(self.names), "relations": len(self.relations), "documents": 0}

    def health(self):
        return True

    def close(self):
        pass


def make_settings(**overrides) -> Settings:
    base = dict(
        _env_file=None, openai_api_key="test", embedding_dim=128, chunk_size=400,
        chunk_overlap=60, entity_link_max_distance=0.9, eval_engine="llm_judge",
        extraction_concurrency=2,
    )
    base.update(overrides)
    return Settings(**base)
