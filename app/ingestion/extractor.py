"""LLM-based entity and relationship extraction into a knowledge graph."""

from __future__ import annotations

import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app import prompts
from app.llm import LLMProtocol
from app.models import Chunk, Entity, Relation

logger = logging.getLogger(__name__)


# -- structured-output schema ---------------------------------------------------
class ExtractedEntity(BaseModel):
    name: str
    type: str = "CONCEPT"
    description: str = ""


class ExtractedRelationship(BaseModel):
    source: str
    target: str
    type: str = "RELATED_TO"
    description: str = ""
    strength: int = Field(default=5, ge=1, le=10)


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


# -- normalisation ----------------------------------------------------------------
def normalize_entity_id(name: str) -> str:
    """Stable id used to merge the same entity across chunks and documents."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    text = text.lower().strip()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"['’`]s\b", "", text)
    text = re.sub(r"[^a-z0-9&+]+", "_", text).strip("_")
    return text


def normalize_label(label: str, default: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", label or "").strip("_").upper()
    return text or default


def _clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", name or "").strip().strip("\"'")


@dataclass
class GraphExtraction:
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: dict[tuple[str, str, str], Relation] = field(default_factory=dict)
    mentions: set[tuple[str, str]] = field(default_factory=set)  # (chunk_id, entity_id)

    def add_entity(self, name: str, etype: str, description: str, chunk_id: str) -> str | None:
        name = _clean_name(name)
        eid = normalize_entity_id(name)
        if not eid or len(name) > 120:
            return None
        etype = normalize_label(etype, "CONCEPT")
        existing = self.entities.get(eid)
        if existing is None:
            self.entities[eid] = Entity(id=eid, name=name, type=etype, description=description)
        else:
            if len(description) > len(existing.description):
                existing.description = description
            if existing.type == "CONCEPT" and etype != "CONCEPT":
                existing.type = etype
        self.mentions.add((chunk_id, eid))
        return eid

    def add_relation(
        self, src: str, tgt: str, rtype: str, description: str, weight: float, chunk_id: str
    ) -> None:
        if src == tgt:
            return
        rtype = normalize_label(rtype, "RELATED_TO")
        key = (src, tgt, rtype)
        rel = self.relations.get(key)
        if rel is None:
            self.relations[key] = Relation(
                source=src, target=tgt, type=rtype, description=description,
                weight=weight, chunk_ids=[chunk_id],
            )
        else:
            rel.weight += weight
            if chunk_id not in rel.chunk_ids:
                rel.chunk_ids.append(chunk_id)
            if len(description) > len(rel.description):
                rel.description = description

    def merge(self, result: ExtractionResult, chunk_id: str) -> None:
        name_to_id: dict[str, str] = {}
        for ent in result.entities:
            eid = self.add_entity(ent.name, ent.type, ent.description.strip(), chunk_id)
            if eid:
                name_to_id[_clean_name(ent.name).lower()] = eid
        for rel in result.relationships:
            ids = []
            for endpoint in (rel.source, rel.target):
                eid = name_to_id.get(_clean_name(endpoint).lower())
                if eid is None:  # endpoint not listed as an entity: add it
                    eid = self.add_entity(endpoint, "CONCEPT", "", chunk_id)
                ids.append(eid)
            if all(ids):
                self.add_relation(
                    ids[0], ids[1], rel.type, rel.description.strip(),
                    rel.strength / 10.0, chunk_id,
                )


class GraphExtractor:
    def __init__(self, llm: LLMProtocol, concurrency: int = 4):
        self.llm = llm
        self.concurrency = max(1, concurrency)

    def _extract_one(self, title: str, chunk: Chunk) -> ExtractionResult:
        try:
            return self.llm.complete_structured(
                prompts.EXTRACTION_SYSTEM,
                prompts.EXTRACTION_USER.format(title=title, text=chunk.content),
                ExtractionResult,
            )
        except Exception as exc:
            logger.error("Extraction failed for chunk %s: %s", chunk.id, exc)
            return ExtractionResult()

    def extract(self, title: str, chunks: list[Chunk]) -> GraphExtraction:
        graph = GraphExtraction()
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            results = list(pool.map(lambda c: self._extract_one(title, c), chunks))
        for chunk, result in zip(chunks, results):
            graph.merge(result, chunk.id)
        logger.info(
            "Extracted %d entities, %d relations from %d chunks",
            len(graph.entities), len(graph.relations), len(chunks),
        )
        return graph
