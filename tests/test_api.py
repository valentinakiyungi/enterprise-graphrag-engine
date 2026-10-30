import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("langgraph")
pytest.importorskip("pydantic_settings")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402
from tests.fakes import DOCS, make_env  # noqa: E402

QUESTION = "Who is the manager of the person who leads Project Atlas?"


def client_for(api_key=""):
  return TestClient(create_app(make_env(api_key=api_key)))


def ingest_payload():
  return {"documents": [{"doc_id": k, "text": v} for k, v in DOCS.items()]}


def test_health_and_ready():
  with client_for() as c:
    assert c.get("/health").json() == {"status": "ok"}
    assert c.get("/ready").json() == {"status": "ready"}


def test_ingest_then_query_hybrid():
  with client_for() as c:
    r = c.post("/ingest", json=ingest_payload())
    assert r.status_code == 200
    assert [d["chunks"] for d in r.json()["ingested"]] == [1, 1, 1]

    r = c.post("/query", json={"query": QUESTION, "mode": "hybrid", "top_k": 1})
    body = r.json()
    assert r.status_code == 200
    assert body["generation"] == "Elena Petrova"
    assert body["entities"] == ["Project Atlas"]
    assert "Tomas Weber reports to Elena Petrova" in body["graph_facts"]
    assert {s["origin"] for s in body["sources"]} == {"vector", "graph"}
    assert body["faithfulness_score"] is None  # evaluation not requested


def test_query_vector_mode_is_baseline():
  with client_for() as c:
    c.post("/ingest", json=ingest_payload())
    body = c.post("/query", json={"query": QUESTION, "mode": "vector", "top_k": 1}).json()
    assert body["graph_facts"] == []
    assert body["generation"].lower().startswith("i don't know")


def test_query_with_evaluation():
  with client_for() as c:
    c.post("/ingest", json=ingest_payload())
    body = c.post("/query", json={"query": QUESTION, "top_k": 1, "evaluate": True}).json()
    assert body["faithfulness_score"] == 0.9
    assert body["low_confidence"] is False


def test_validation_errors():
  with client_for() as c:
    assert c.post("/query", json={"query": "x", "mode": "nope"}).status_code == 422
    assert c.post("/query", json={"query": ""}).status_code == 422
    assert c.post("/ingest", json={"documents": []}).status_code == 422


def test_api_key_enforced_when_configured():
  with client_for(api_key="secret") as c:
    assert c.get("/health").status_code == 200  # health stays open for load balancers
    assert c.get("/stats").status_code == 401
    assert c.get("/stats", headers={"X-API-Key": "wrong"}).status_code == 401
    ok = c.get("/stats", headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
    assert ok.json()["vector"]["chunks"] == 0
