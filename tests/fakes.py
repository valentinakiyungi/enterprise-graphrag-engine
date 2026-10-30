"""In-memory test doubles for the stores, embeddings, LLM and evaluator."""

import hashlib
import math
import re
from collections import deque
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from app.schemas import EntityList, Triple, TripleList
from app.text import REFUSAL_TEXT, normalize_entity

DIM = 64


def _embed(text: str) -> List[float]:
  v = [0.0] * DIM
  for word in re.findall(r"[a-z0-9]+", text.lower()):
    v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
  norm = math.sqrt(sum(x * x for x in v)) or 1.0
  return [x / norm for x in v]


class FakeEmbeddings:

  def embed_documents(self, texts):
    return [_embed(t) for t in texts]

  def embed_query(self, text):
    return _embed(text)


class FakeVectorStore:

  def __init__(self):
    self.rows: List[Dict[str, Any]] = []
    self._next = 1

  def add_chunks(self, doc_id, chunks, embeddings, metadata=None):
    self.rows = [r for r in self.rows if r["doc_id"] != doc_id]
    ids = []
    for i, (content, emb) in enumerate(zip(chunks, embeddings)):
      self.rows.append({"id": self._next, "doc_id": doc_id, "chunk_index": i, "content": content, "emb": emb})
      ids.append(self._next)
      self._next += 1
    return ids

  def search(self, embedding, k=5):
    scored = sorted(
        ((sum(a * b for a, b in zip(embedding, r["emb"])), r) for r in self.rows),
        key=lambda x: -x[0],
    )
    return [
        {"id": r["id"], "doc_id": r["doc_id"], "chunk_index": r["chunk_index"], "content": r["content"], "score": s}
        for s, r in scored[:k]
    ]

  def get_chunks(self, ids):
    by_id = {r["id"]: r for r in self.rows}
    return [
        {"id": i, "doc_id": by_id[i]["doc_id"], "chunk_index": by_id[i]["chunk_index"],
         "content": by_id[i]["content"], "score": None}
        for i in ids if i in by_id
    ]

  def stats(self):
    return {"chunks": len(self.rows), "documents": len({r["doc_id"] for r in self.rows})}

  def ping(self):
    return None

  def close(self):
    return None


class FakeGraphStore:

  def __init__(self):
    self.edges: List[Dict[str, Any]] = []

  def add_triples(self, doc_id, triples):
    n = 0
    for t in triples:
      s_key, o_key = normalize_entity(t["subject"]), normalize_entity(t["object"])
      if s_key and o_key and s_key != o_key:
        self.edges.append({
            "doc_id": doc_id, "s_key": s_key, "o_key": o_key, "subject": t["subject"],
            "object": t["object"], "predicate": t["predicate"].strip().lower(), "chunk_id": t["chunk_id"],
        })
        n += 1
    return n

  def delete_document(self, doc_id):
    self.edges = [e for e in self.edges if e["doc_id"] != doc_id]

  def resolve_entities(self, keys, limit=20):
    known = {e["s_key"] for e in self.edges} | {e["o_key"] for e in self.edges}
    return [k for k in keys if k in known]

  def expand(self, keys, max_hops=2, limit=40):
    seen_nodes = set(keys)
    frontier = deque((k, 0) for k in keys)
    out, used = [], set()
    while frontier:
      node, depth = frontier.popleft()
      if depth >= max_hops:
        continue
      for idx, e in enumerate(self.edges):
        if idx in used or node not in (e["s_key"], e["o_key"]):
          continue
        used.add(idx)
        out.append({k: e[k] for k in ("subject", "predicate", "object", "chunk_id", "doc_id")})
        other = e["o_key"] if node == e["s_key"] else e["s_key"]
        if other not in seen_nodes:
          seen_nodes.add(other)
          frontier.append((other, depth + 1))
    return out[:limit]

  def stats(self):
    nodes = {e["s_key"] for e in self.edges} | {e["o_key"] for e in self.edges}
    return {"entities": len(nodes), "relationships": len(self.edges)}

  def ping(self):
    return None

  def close(self):
    return None


class _Structured:

  def __init__(self, llm, schema):
    self.llm, self.schema = llm, schema

  def invoke(self, messages):
    text = messages[-1][1]
    if self.schema is EntityList:
      return EntityList(entities=[e for e in self.llm.entities if e.lower() in text.lower()])
    if self.schema is TripleList:
      return TripleList(triples=[t for needle, t in self.llm.triple_script if needle in text])
    raise AssertionError(f"unexpected schema {self.schema}")


class FakeLLM:
  """Answers 'Elena Petrova' only if that name appears in the supplied context."""

  def __init__(self, entities=None, triple_script=None):
    self.entities = entities or []
    self.triple_script = triple_script or []
    self.last_prompt: Optional[str] = None

  def with_structured_output(self, schema):
    return _Structured(self, schema)

  def invoke(self, messages):
    self.last_prompt = messages[-1][1]
    context = self.last_prompt.split("Question:")[0]
    return SimpleNamespace(content="Elena Petrova" if "Elena Petrova" in context else REFUSAL_TEXT)


class FakeEvaluator:

  def evaluate(self, question, answer, contexts, reference=None):
    return {"faithfulness": 0.9, "context_precision": 0.8, "precision_basis": "response"}


DOCS = {
    "projects": "Tomas Weber leads Project Atlas.",
    "people": "Tomas Weber reports to Elena Petrova.",
    "misc": "The cafeteria opens at nine every morning.",
}

TRIPLES = [
    ("leads Project Atlas", Triple(subject="Tomas Weber", predicate="leads", object="Project Atlas")),
    ("reports to Elena Petrova", Triple(subject="Tomas Weber", predicate="reports to", object="Elena Petrova")),
]


def make_env(api_key: str = "", top_k: int = 1):
  """Build fakes, ingest DOCS, and return (Services, parts)."""
  from app.config import Settings
  from app.graph import build_graph
  from app.ingestion import Ingestor
  from app.services import Services

  settings = Settings(openai_api_key="test", vector_top_k=top_k, graph_max_hops=2, api_key=api_key)
  vs, gs, emb = FakeVectorStore(), FakeGraphStore(), FakeEmbeddings()
  llm = FakeLLM(entities=["Project Atlas"], triple_script=TRIPLES)
  evaluator = FakeEvaluator()
  ingestor = Ingestor(vs, gs, emb, llm, chunk_size=200, chunk_overlap=0)
  graph = build_graph(
      settings=settings, vector_store=vs, graph_store=gs, embeddings=emb, llm=llm, evaluator=evaluator
  )
  services = Services(settings, vs, gs, emb, llm, ingestor, graph, evaluator)
  return services
