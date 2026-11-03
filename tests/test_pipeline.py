"""End-to-end behaviour with in-memory stores and a deterministic fake LLM."""

import pytest

from app.nodes import make_nodes, route_after_evaluation


def test_ingestion_builds_graph_across_documents(services):
    pg, graph = services.pg, services.graph_store
    assert len(pg.docs) == 2
    assert {"insight_board", "orion_lake", "nimbus_cloud", "ada_lovelace"} <= set(pg.entities)
    types = {r["type"] for r in graph.relations}
    assert types == {"DEPENDS_ON", "HOSTED_BY", "SPONSORED_BY"}
    # provenance points back to chunk ids
    assert all(cid.startswith("doc_") and "::" in cid
               for r in graph.relations for cid in r["chunk_ids"])


def test_reingest_is_idempotent(services):
    before = (len(services.pg.chunks), len(services.graph_store.relations))
    from tests.conftest import CORPUS

    services.ingestion.ingest("Systems", CORPUS["Systems"], source="Systems.md")
    assert (len(services.pg.chunks), len(services.graph_store.relations)) == before


def test_delete_document_cleans_both_stores(services):
    doc_id = next(d for d, v in services.pg.docs.items() if v["title"] == "Vendors")
    assert services.ingestion.delete(doc_id)
    assert "ada_lovelace" not in services.pg.entities
    assert all(r["type"] != "SPONSORED_BY" for r in services.graph_store.relations)


def test_graph_retriever_follows_multi_hop_chain(services):
    res = services.graph_retriever.retrieve(
        "Who sponsors the vendor hosting the platform behind Insight Board?",
        ["Insight Board"], max_hops=3,
    )
    assert res.seed_entities[0]["id"] == "insight_board"
    chain = {(f.source, f.relation, f.target) for f in res.facts}
    assert ("Nimbus Cloud", "SPONSORED_BY", "Ada Lovelace") in chain
    assert {c.document_id for c in res.chunks} == set(services.pg.docs)  # both docs reached


def _run_nodes(svc, state):
    """Execute the workflow node-by-node (mirrors app/graph.py) without LangGraph."""
    nodes = make_nodes(svc)
    route = route_after_evaluation(svc.settings)
    trace = []

    def apply(update):
        trace.extend(update.pop("trace", []))
        state.update(update)

    apply(nodes["analyze_query"](state))
    nxt = "retrieve"
    while True:
        if nxt == "retrieve":
            apply(nodes["vector_retrieve"](state))
            apply(nodes["graph_retrieve"](state))
            apply(nodes["fuse_context"](state))
        apply(nodes["generate_answer"](state))
        apply(nodes["evaluate_answer"](state))
        decision = route(state)
        if decision == "end":
            break
        if decision == "expand":
            apply(nodes["expand_retrieval"](state))
            nxt = "retrieve"
        else:
            apply(nodes["prepare_regeneration"](state))
            nxt = "generate"
    state["trace"] = trace
    return state


def manual_initial(query, mode="hybrid", evaluate=True):
    return {"query": query, "mode": mode, "top_k": 5, "max_hops": 3, "evaluate": evaluate,
            "reference": None, "attempts": 0, "correction": "", "unsupported_claims": []}


def test_self_correction_removes_hallucination(services):
    state = _run_nodes(services, manual_initial(
        "Who sponsors the vendor hosting the platform behind Insight Board?"))
    assert state["attempts"] == 1
    assert "invented" not in state["generation"].lower()
    assert state["evaluation"]["faithfulness"] == 1.0
    nodes = [t["node"] for t in state["trace"]]
    assert "prepare_regeneration" in nodes
    assert any(c["label"].startswith("G") for c in state["citations"])


def test_vector_mode_skips_graph(services):
    state = _run_nodes(services, manual_initial("Insight Board platform", mode="vector",
                                                evaluate=False))
    assert state["facts"] == "(none)"
    assert all(c["retriever"] != "graph" for c in state["citations"])
    graph_trace = next(t for t in state["trace"] if t["node"] == "graph_retrieve")
    assert graph_trace.get("skipped")


def test_route_policy(settings):
    route = route_after_evaluation(settings)
    assert route({"evaluation": {}}) == "end"
    assert route({"evaluation": {"faithfulness": 0.9}, "attempts": 0}) == "end"
    assert route({"evaluation": {"faithfulness": 0.2, "context_precision": 0.1},
                  "attempts": 0, "correction": ""}) == "expand"
    assert route({"evaluation": {"faithfulness": 0.2, "context_precision": 0.1},
                  "attempts": 1, "correction": "expand"}) == "regenerate"
    assert route({"evaluation": {"faithfulness": 0.2}, "attempts": 2}) == "end"


def test_langgraph_workflow_end_to_end(services):
    pytest.importorskip("langgraph")
    from app.graph import run_query

    state = run_query(services, "Who sponsors the vendor hosting the platform behind Insight Board?",
                      max_hops=3)
    assert "Ada Lovelace" in state["generation"]
    assert state["evaluation"]["faithfulness"] == 1.0
    assert state["attempts"] == 1


def test_benchmark_runner(services):
    pytest.importorskip("langgraph")
    from app.evaluation.benchmark import BenchmarkRunner

    items = [{"id": "q1", "type": "multi_hop", "reference": "Ada Lovelace",
              "question": "Who sponsors the vendor hosting the platform behind Insight Board?"}]
    summary, results = BenchmarkRunner(services).run(items, ["vector", "hybrid"], True, 1)
    assert len(results) == 2 and not any(r.get("error") for r in results)
    assert summary["per_mode"]["hybrid"]["multi_hop_accuracy"] == 1.0
