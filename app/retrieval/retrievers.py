"""Vector (pgvector + full-text) and knowledge-graph (Neo4j) retrievers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import Settings
from app.llm import LLMProtocol
from app.models import GraphFact, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class VectorResult:
    dense: list[RetrievedChunk] = field(default_factory=list)
    keyword: list[RetrievedChunk] = field(default_factory=list)


@dataclass
class GraphResult:
    seed_entities: list[dict] = field(default_factory=list)
    facts: list[GraphFact] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)


class VectorRetriever:
    """Dense semantic search (HNSW cosine) plus Postgres full-text search."""

    def __init__(self, settings: Settings, llm: LLMProtocol, pg):
        self.s = settings
        self.llm = llm
        self.pg = pg

    def retrieve(self, query: str, keywords: list[str], top_k: int) -> VectorResult:
        embedding = self.llm.embed_query(query)
        dense = self.pg.vector_search(embedding, top_k)
        keyword_query = " OR ".join(keywords) if keywords else query
        try:
            keyword = self.pg.keyword_search(keyword_query, self.s.keyword_top_k)
        except Exception as exc:  # malformed tsquery should never fail a request
            logger.warning("Keyword search failed: %s", exc)
            keyword = []
        return VectorResult(dense=dense, keyword=keyword)


class GraphRetriever:
    """Entity linking (pgvector + exact name) then multi-hop Neo4j traversal."""

    def __init__(self, settings: Settings, llm: LLMProtocol, pg, graph):
        self.s = settings
        self.llm = llm
        self.pg = pg
        self.graph = graph

    def link_entities(self, query: str, entity_names: list[str]) -> list[dict]:
        found: dict[str, dict] = {}
        for row in self.pg.find_entities_by_name(entity_names):
            found[row["id"]] = row
        texts = [query] + entity_names
        vectors = self.llm.embed(texts)
        per_name_k = 3
        for i, vec in enumerate(vectors):
            k = self.s.graph_seed_entities if i == 0 else per_name_k
            for row in self.pg.search_entities(vec, k, self.s.entity_link_max_distance):
                prev = found.get(row["id"])
                if prev is None or row["distance"] < prev["distance"]:
                    found[row["id"]] = row
        ranked = sorted(found.values(), key=lambda r: r["distance"])
        return ranked[: self.s.graph_seed_entities]

    def retrieve(self, query: str, entity_names: list[str], max_hops: int) -> GraphResult:
        seeds = self.link_entities(query, entity_names)
        if not seeds:
            return GraphResult()
        facts, chunk_scores = self.graph.expand(
            [s["id"] for s in seeds], max_hops, self.s.graph_max_facts
        )
        top_ids = [
            cid
            for cid, _ in sorted(chunk_scores.items(), key=lambda kv: kv[1], reverse=True)[
                : self.s.graph_top_chunks
            ]
        ]
        by_id = self.pg.get_chunks(top_ids)
        chunks = []
        for cid in top_ids:
            if cid in by_id:
                c = by_id[cid]
                c.score = round(chunk_scores[cid], 4)
                chunks.append(c)
        return GraphResult(
            seed_entities=[
                {"id": s["id"], "name": s["name"], "type": s["type"],
                 "distance": round(float(s["distance"]), 4)}
                for s in seeds
            ],
            facts=facts,
            chunks=chunks,
        )
