import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langgraph")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(services):
    from app.main import create_app

    app = create_app(services=services)
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_query_endpoint(client):
    r = client.post("/api/query", json={
        "query": "Who sponsors the vendor hosting the platform behind Insight Board?",
        "max_hops": 3, "include_trace": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "Ada Lovelace" in body["answer"]
    assert body["evaluation"]["faithfulness"] == 1.0
    assert body["trace"]
    # v1 endpoint still works
    assert client.post("/query", json={"query": "Insight Board"}).status_code == 200


def test_document_lifecycle(client):
    r = client.post("/api/documents", json={"title": "Extra", "content": "Zeta Hub OWNS Kappa Node."})
    assert r.status_code == 200, r.text
    doc_id = r.json()["document_id"]
    assert any(d["id"] == doc_id for d in client.get("/api/documents").json())
    assert client.delete(f"/api/documents/{doc_id}").status_code == 200
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404


def test_api_key_enforced(services):
    from app.main import create_app

    app = create_app(services=services)
    app.state.settings = services.settings.model_copy(update={"api_key": "secret"})
    with TestClient(app) as c:
        assert c.get("/api/stats").status_code == 401
        assert c.get("/api/stats", headers={"X-API-Key": "secret"}).status_code == 200


def test_benchmark_dataset_path_is_sandboxed(client):
    r = client.post("/api/benchmarks", json={"dataset_path": "../../etc/passwd"})
    assert r.status_code == 404
