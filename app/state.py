from typing import Any, Dict, List, Optional, TypedDict


class GraphRAGState(TypedDict, total=False):
  # inputs
  query: str
  mode: str  # "hybrid" (vector + graph) or "vector" (baseline)
  top_k: int
  evaluate: bool
  reference: Optional[str]  # optional ground-truth answer (enables reference-based precision)

  # intermediate
  vector_hits: List[Dict[str, Any]]
  entities: List[str]
  graph_facts: List[Dict[str, Any]]
  graph_hits: List[Dict[str, Any]]
  contexts: List[str]

  # outputs
  generation: str
  evaluation: Dict[str, Any]
