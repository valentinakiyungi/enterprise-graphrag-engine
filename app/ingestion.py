"""Document ingestion: chunk -> embed -> pgvector, and LLM triple extraction -> Neo4j."""

import logging
from typing import Any, Dict, List, Optional

from app.schemas import TripleList
from app.text import chunk_text, normalize_entity

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = (
    "Extract factual knowledge-graph triples (subject, predicate, object) from the text.\n"
    "Rules:\n"
    "- Only extract facts explicitly stated in the text; never infer.\n"
    "- Use canonical full entity names. Resolve pronouns and abbreviations to the full name.\n"
    "- Entities are people, teams, divisions, projects, systems, vendors, policies, locations, incidents.\n"
    "- Predicates are short lowercase verb phrases such as 'reports to', 'leads', 'owns',"
    " 'depends on', 'founded', 'supplies', 'enforces', 'caused by'.\n"
    "- Prefer one atomic fact per triple. Return an empty list if there are no facts."
)


class Ingestor:

  def __init__(self, vector_store, graph_store, embeddings, llm, chunk_size: int = 800, chunk_overlap: int = 100):
    self.vector_store = vector_store
    self.graph_store = graph_store
    self.embeddings = embeddings
    self.llm = llm
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap

  def extract_triples(self, chunk: str) -> Optional[List[Dict[str, str]]]:
    """Returns triples, or None if extraction failed (so callers can count failures)."""
    try:
      structured = self.llm.with_structured_output(TripleList)
      result = structured.invoke([("system", EXTRACTION_PROMPT), ("human", chunk)])
    except Exception:
      logger.exception("Triple extraction failed")
      return None
    out = []
    for t in result.triples:
      if normalize_entity(t.subject) and normalize_entity(t.object) and t.predicate.strip():
        out.append({"subject": t.subject, "predicate": t.predicate, "object": t.object})
    return out

  def ingest(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
    if not chunks:
      return {"doc_id": doc_id, "chunks": 0, "triples": 0, "extraction_failures": 0}

    embeddings = self.embeddings.embed_documents(chunks)
    chunk_ids = self.vector_store.add_chunks(doc_id, chunks, embeddings, metadata)

    self.graph_store.delete_document(doc_id)  # re-ingesting a doc replaces its facts
    rows: List[Dict[str, Any]] = []
    failures = 0
    for chunk_id, chunk in zip(chunk_ids, chunks):
      triples = self.extract_triples(chunk)
      if triples is None:
        failures += 1
        continue
      rows.extend({**t, "chunk_id": chunk_id} for t in triples)
    stored = self.graph_store.add_triples(doc_id, rows)
    return {"doc_id": doc_id, "chunks": len(chunks), "triples": stored, "extraction_failures": failures}
