import pytest

pytest.importorskip("langgraph")
pytest.importorskip("pydantic_settings")

from tests.fakes import DOCS, make_env  # noqa: E402


@pytest.fixture
def services():
  s = make_env()
  for doc_id, text in DOCS.items():
    s.ingestor.ingest(doc_id, text)
  return s


def run(services, mode, **extra):
  return services.graph.invoke({
      "query": "Who is the manager of the person who leads Project Atlas?",
      "mode": mode,
      "top_k": 1,
      **extra,
  })


def test_ingestion_builds_graph(services):
  assert services.vector_store.stats()["chunks"] == 3
  assert services.graph_store.stats()["relationships"] == 2


def test_vector_mode_misses_second_hop(services):
  state = run(services, "vector")
  assert "graph_facts" not in state
  assert not any("Elena Petrova" in c for c in state["contexts"])
  assert state["generation"].lower().startswith("i don't know")


def test_hybrid_mode_follows_graph_to_second_hop(services):
  state = run(services, "hybrid")
  assert state["entities"] == ["Project Atlas"]
  assert any(f["object"] == "Elena Petrova" for f in state["graph_facts"])
  assert any("Elena Petrova" in c for c in state["contexts"])
  assert state["generation"] == "Elena Petrova"


def test_evaluation_runs_only_when_requested(services):
  assert "evaluation" not in run(services, "hybrid", evaluate=False)
  ev = run(services, "hybrid", evaluate=True)["evaluation"]
  assert ev["faithfulness"] == 0.9


def test_reingest_replaces_document_facts(services):
  before = services.graph_store.stats()["relationships"]
  services.ingestor.ingest("projects", "Tomas Weber leads Project Atlas.")
  assert services.graph_store.stats()["relationships"] == before
  assert services.vector_store.stats()["chunks"] == 3


def test_no_context_returns_refusal():
  s = make_env()  # nothing ingested
  state = s.graph.invoke({"query": "anything", "mode": "vector", "top_k": 1})
  assert state["generation"].lower().startswith("i don't know")
