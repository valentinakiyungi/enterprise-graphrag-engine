"""Internal domain models shared across ingestion, retrieval and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Chunk:
    id: str
    document_id: str
    index: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class Entity:
    id: str
    name: str
    type: str
    description: str = ""
    embedding: list[float] | None = None


@dataclass
class Relation:
    source: str  # entity id
    target: str  # entity id
    type: str
    description: str = ""
    weight: float = 1.0
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    """A chunk returned by any retriever, with provenance."""

    id: str
    document_id: str
    content: str
    score: float
    retriever: str  # "vector" | "keyword" | "graph"
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphFact:
    """A knowledge-graph edge rendered for the LLM."""

    source: str
    relation: str
    target: str
    description: str = ""
    hop: int = 1
    weight: float = 1.0
    chunk_ids: list[str] = field(default_factory=list)

    def render(self) -> str:
        base = f"({self.source}) -[{self.relation}]-> ({self.target})"
        return f"{base}: {self.description}" if self.description else base

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
