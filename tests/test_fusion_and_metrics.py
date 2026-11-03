import pytest

from app.evaluation.benchmark import summarize
from app.evaluation.metrics import average_precision, faithfulness_score
from app.ingestion.extractor import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    GraphExtraction,
    normalize_entity_id,
    normalize_label,
)
from app.models import GraphFact, RetrievedChunk
from app.retrieval.fusion import build_context, fusion_weights, reciprocal_rank_fusion


def rc(cid, retriever="vector", content=None):
    return RetrievedChunk(id=cid, document_id="d", content=content or f"text {cid}",
                          score=0.0, retriever=retriever, title="Doc")


# --- fusion ------------------------------------------------------------------------
def test_rrf_rewards_agreement_between_retrievers():
    fused = reciprocal_rank_fusion({
        "vector": [rc("a"), rc("b"), rc("c")],
        "graph": [rc("c", "graph"), rc("d", "graph")],
    })
    assert fused[0].id == "c"  # surfaced by both retrievers
    assert fused[0].retriever == "graph+vector"
    assert {c.id for c in fused} == {"a", "b", "c", "d"}


def test_rrf_zero_weight_drops_list():
    fused = reciprocal_rank_fusion({"vector": [rc("a")], "graph": [rc("g", "graph")]},
                                   weights={"vector": 0.0, "graph": 1.0})
    assert [c.id for c in fused] == ["g"]


def test_fusion_weights_by_mode():
    assert fusion_weights("vector", True)["graph"] == 0
    assert fusion_weights("graph", False)["vector"] == 0
    assert fusion_weights("hybrid", True)["graph"] > fusion_weights("hybrid", False)["graph"]


def test_build_context_labels_and_budget():
    chunks = [rc(str(i), content="word " * 200) for i in range(10)]
    facts = [GraphFact("A", "OWNS", "B", "A owns B", 1), GraphFact("A", "OWNS", "B", "dup", 2)]
    passages, facts_block, citations, contexts = build_context(chunks, facts, 8, 3000)
    assert passages.startswith("[S1]")
    assert facts_block == "[G1] (A) -[OWNS]-> (B): A owns B"  # duplicate removed
    assert len(passages) + len(facts_block) <= 3000
    labels = [c["label"] for c in citations]
    assert "G1" in labels and labels[0] == "S1"
    assert contexts[-1].startswith("Knowledge graph facts")


# --- extraction normalisation ------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    ("The Helios Data Platform", "helios_data_platform"),
    ("Helios data-platform", "helios_data_platform"),
    ("Kestrel's  CFO", "kestrel_cfo"),
    ("AT&T", "at&t"),
])
def test_normalize_entity_id(name, expected):
    assert normalize_entity_id(name) == expected


def test_normalize_label():
    assert normalize_label("depends on", "X") == "DEPENDS_ON"
    assert normalize_label("", "RELATED_TO") == "RELATED_TO"


def test_graph_extraction_merges_across_chunks():
    g = GraphExtraction()
    g.merge(ExtractionResult(
        entities=[ExtractedEntity(name="Helios", type="system", description="Lakehouse")],
        relationships=[ExtractedRelationship(source="Helios", target="Stratus Cloud",
                                             type="hosted by", strength=10)],
    ), "doc::0")
    g.merge(ExtractionResult(
        entities=[ExtractedEntity(name="helios", type="SYSTEM",
                                  description="Central lakehouse for analytics")],
        relationships=[ExtractedRelationship(source="helios", target="Stratus Cloud",
                                             type="HOSTED_BY", strength=6)],
    ), "doc::1")
    assert set(g.entities) == {"helios", "stratus_cloud"}  # endpoint auto-added
    assert g.entities["helios"].description == "Central lakehouse for analytics"
    rel = g.relations[("helios", "stratus_cloud", "HOSTED_BY")]
    assert rel.chunk_ids == ["doc::0", "doc::1"]
    assert rel.weight == pytest.approx(1.6)


# --- metrics -----------------------------------------------------------------------
def test_faithfulness_score():
    assert faithfulness_score([True, True, False, True]) == 0.75
    assert faithfulness_score([]) == 1.0


def test_average_precision_matches_ragas_definition():
    assert average_precision([True, False, True]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([False, False]) == 0.0
    assert average_precision([True, True]) == 1.0


def test_benchmark_summary_and_comparison():
    def row(mode, typ, acc, faith):
        return {"mode": mode, "type": typ, "latency_ms": 100, "attempts": 0,
                "metrics": {"answer_correctness": acc, "faithfulness": faith,
                            "context_precision": 0.5}}
    results = [
        row("vector", "multi_hop", 0.5, 0.4), row("vector", "multi_hop", 0.5, 0.9),
        row("vector", "single_hop", 1.0, 0.9),
        row("hybrid", "multi_hop", 1.0, 0.9), row("hybrid", "multi_hop", 0.5, 0.9),
        row("hybrid", "single_hop", 1.0, 0.95),
        {"mode": "hybrid", "type": "multi_hop", "error": "boom", "metrics": {}},
    ]
    s = summarize(results)
    assert s["per_mode"]["vector"]["multi_hop_accuracy"] == 0.5
    assert s["per_mode"]["hybrid"]["multi_hop_accuracy"] == 0.75
    assert s["per_mode"]["hybrid"]["errors"] == 1
    assert s["comparison"]["multi_hop_accuracy_improvement_pct"] == 50.0
    assert s["per_mode"]["vector"]["hallucination_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["comparison"]["hallucination_reduction_pct"] == 100.0
