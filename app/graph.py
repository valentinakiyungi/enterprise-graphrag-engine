"""LangGraph workflow: hybrid retrieval with Ragas-gated self-correction.

::

                    ┌──────────────┐
                    │ analyze_query│
                    └──────┬───────┘
             ┌─────────────┴─────────────┐      (parallel fan-out)
     ┌───────▼────────┐          ┌───────▼───────┐
     │ vector_retrieve│          │ graph_retrieve│
     │ pgvector + FTS │          │ Neo4j k-hop   │
     └───────┬────────┘          └───────┬───────┘
             └─────────────┬─────────────┘      (join)
                    ┌──────▼───────┐
                    │ fuse_context │  weighted RRF + labelled context
                    └──────┬───────┘
                    ┌──────▼────────┐
           ┌───────►│generate_answer│
           │        └──────┬────────┘
           │        ┌──────▼────────┐
           │        │evaluate_answer│  Ragas faithfulness + context precision
           │        └──────┬────────┘
           │    regenerate │ expand          end
           │   ┌───────────┼───────────┐──────────► END
  ┌────────┴───▼─────┐  ┌──▼──────────────┐
  │prepare_regenerate│  │expand_retrieval │──► vector_retrieve + graph_retrieve
  └──────────────────┘  └─────────────────┘
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.nodes import make_nodes, route_after_evaluation
from app.state import GraphRAGState


def build_graph(svc):
    nodes = make_nodes(svc)
    wf = StateGraph(GraphRAGState)
    for name, fn in nodes.items():
        wf.add_node(name, fn)

    wf.add_edge(START, "analyze_query")
    wf.add_edge("analyze_query", "vector_retrieve")
    wf.add_edge("analyze_query", "graph_retrieve")
    wf.add_edge(["vector_retrieve", "graph_retrieve"], "fuse_context")
    wf.add_edge("fuse_context", "generate_answer")
    wf.add_edge("generate_answer", "evaluate_answer")
    wf.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluation(svc.settings),
        {"end": END, "expand": "expand_retrieval", "regenerate": "prepare_regeneration"},
    )
    wf.add_edge("expand_retrieval", "vector_retrieve")
    wf.add_edge("expand_retrieval", "graph_retrieve")
    wf.add_edge("prepare_regeneration", "generate_answer")
    return wf.compile()


def initial_state(
    query: str,
    mode: str = "hybrid",
    *,
    top_k: int,
    max_hops: int,
    evaluate: bool,
    reference: str | None = None,
) -> GraphRAGState:
    return GraphRAGState(
        query=query,
        mode=mode,  # type: ignore[typeddict-item]
        top_k=top_k,
        max_hops=max_hops,
        evaluate=evaluate,
        reference=reference,
        attempts=0,
        correction="",
        unsupported_claims=[],
        trace=[],
    )


def run_query(
    svc,
    query: str,
    mode: str = "hybrid",
    *,
    top_k: int | None = None,
    max_hops: int | None = None,
    evaluate: bool | None = None,
    reference: str | None = None,
) -> GraphRAGState:
    s = svc.settings
    state = initial_state(
        query,
        mode,
        top_k=top_k or s.vector_top_k,
        max_hops=max_hops or s.graph_max_hops,
        evaluate=s.evaluate_on_query if evaluate is None else evaluate,
        reference=reference,
    )
    return svc.workflow.invoke(state, {"recursion_limit": 50})
