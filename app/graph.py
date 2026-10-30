"""LangGraph workflow: vector retrieval -> graph traversal -> synthesis -> Ragas evaluation."""

import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from app.llm import structured
from app.schemas import EntityList
from app.state import GraphRAGState
from app.text import REFUSAL_TEXT, normalize_entity

logger = logging.getLogger(__name__)

ENTITY_PROMPT = (
    "List the named entities (people, teams, divisions, projects, systems, vendors, policies,"
    " locations, incidents) mentioned in the user's question. Use the exact names as written."
    " Return an empty list if there are none."
)

ANSWER_PROMPT = (
    "You are an enterprise knowledge assistant. Answer the question using ONLY the provided context.\n"
    "The context has document excerpts and knowledge-graph facts. Combine them step by step to answer"
    " multi-hop questions.\n"
    f"If the context does not contain enough information, reply exactly: \"{REFUSAL_TEXT}\"\n"
    "Never use outside knowledge and never guess. Keep the answer concise."
)


def format_fact(fact: Dict[str, Any]) -> str:
  return f"{fact['subject']} {fact['predicate']} {fact['object']}"


def build_graph(*, settings, vector_store, graph_store, embeddings, llm, evaluator: Optional[Any] = None):
  """Compile the workflow. All collaborators are injected, which keeps it unit-testable."""

  def vector_retrieval(state: GraphRAGState) -> Dict[str, Any]:
    k = state.get("top_k") or settings.vector_top_k
    hits = vector_store.search(embeddings.embed_query(state["query"]), k)
    return {"vector_hits": hits}

  def extract_entities(state: GraphRAGState) -> Dict[str, Any]:
    try:
      extractor = structured(llm, EntityList, settings.structured_output_method)
      result = extractor.invoke([("system", ENTITY_PROMPT), ("human", state["query"])])
      return {"entities": [e for e in result.entities if e.strip()]}
    except Exception:
      logger.exception("Query entity extraction failed; falling back to vector-only context")
      return {"entities": []}

  def graph_traversal(state: GraphRAGState) -> Dict[str, Any]:
    keys = [normalize_entity(e) for e in state.get("entities", [])]
    seeds = graph_store.resolve_entities(keys)
    facts = graph_store.expand(seeds, settings.graph_max_hops, settings.graph_fact_limit)

    # Pull the source chunks the facts came from (skipping ones vector search already found).
    seen = {h["id"] for h in state.get("vector_hits", [])}
    chunk_ids: List[int] = []
    for f in facts:
      cid = f.get("chunk_id")
      if cid is not None and cid not in seen and cid not in chunk_ids:
        chunk_ids.append(cid)
    graph_hits = vector_store.get_chunks(chunk_ids[: settings.graph_chunk_limit])
    return {"graph_facts": facts, "graph_hits": graph_hits}

  def build_context(state: GraphRAGState) -> Dict[str, Any]:
    contexts: List[str] = []
    seen_text = set()
    for hit in list(state.get("vector_hits", [])) + list(state.get("graph_hits", [])):
      if hit["content"] not in seen_text:
        seen_text.add(hit["content"])
        contexts.append(hit["content"])
    facts = state.get("graph_facts", [])
    if facts:
      contexts.append("Knowledge graph facts:\n" + "\n".join(f"- {format_fact(f)}" for f in facts))
    return {"contexts": contexts}

  def generate(state: GraphRAGState) -> Dict[str, Any]:
    contexts = state.get("contexts", [])
    if not contexts:
      return {"generation": REFUSAL_TEXT}
    context_block = "\n\n---\n\n".join(contexts)
    msg = llm.invoke([
        ("system", ANSWER_PROMPT),
        ("human", f"Context:\n{context_block}\n\nQuestion: {state['query']}"),
    ])
    return {"generation": str(msg.content).strip()}

  def evaluate(state: GraphRAGState) -> Dict[str, Any]:
    if evaluator is None:
      return {"evaluation": {"faithfulness": None, "context_precision": None, "error": "evaluator not configured"}}
    return {
        "evaluation": evaluator.evaluate(
            question=state["query"],
            answer=state["generation"],
            contexts=state.get("contexts", []),
            reference=state.get("reference"),
        )
    }

  def route_after_vector(state: GraphRAGState) -> str:
    return "extract_entities" if state.get("mode", "hybrid") == "hybrid" else "build_context"

  def route_after_generate(state: GraphRAGState) -> str:
    return "evaluate" if state.get("evaluate") else END

  workflow = StateGraph(GraphRAGState)
  workflow.add_node("vector_retrieval", vector_retrieval)
  workflow.add_node("extract_entities", extract_entities)
  workflow.add_node("graph_traversal", graph_traversal)
  workflow.add_node("build_context", build_context)
  workflow.add_node("generate", generate)
  workflow.add_node("evaluate", evaluate)

  workflow.set_entry_point("vector_retrieval")
  workflow.add_conditional_edges(
      "vector_retrieval", route_after_vector,
      {"extract_entities": "extract_entities", "build_context": "build_context"},
  )
  workflow.add_edge("extract_entities", "graph_traversal")
  workflow.add_edge("graph_traversal", "build_context")
  workflow.add_edge("build_context", "generate")
  workflow.add_conditional_edges("generate", route_after_generate, {"evaluate": "evaluate", END: END})
  workflow.add_edge("evaluate", END)
  return workflow.compile()
