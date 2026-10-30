import hmac
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from app.services import Services, build_services

logger = logging.getLogger(__name__)


class IngestDocument(BaseModel):
  doc_id: str = Field(min_length=1, max_length=200)
  text: str = Field(min_length=1)
  metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
  documents: List[IngestDocument] = Field(min_length=1, max_length=100)


class QueryRequest(BaseModel):
  query: str = Field(min_length=1, max_length=2000)
  mode: Literal["hybrid", "vector"] = "hybrid"
  top_k: Optional[int] = Field(default=None, ge=1, le=20)
  evaluate: Optional[bool] = None  # None -> EVALUATE_BY_DEFAULT


def create_app(services: Optional[Services] = None) -> FastAPI:
  """App factory. Pass `services` in tests; in production they are built at startup."""

  @asynccontextmanager
  async def lifespan(app: FastAPI):
    app.state.services = services or build_services()
    yield
    if services is None:  # only close what we created
      app.state.services.close()

  app = FastAPI(
      title="Enterprise GraphRAG Engine",
      description="Hybrid vector + knowledge-graph retrieval with automated Ragas evaluation",
      version="1.0.0",
      lifespan=lifespan,
  )

  def svc(request: Request) -> Services:
    return request.app.state.services

  def require_api_key(request: Request, x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = svc(request).settings.api_key
    if expected and not (x_api_key and hmac.compare_digest(x_api_key, expected)):
      raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

  @app.get("/health")
  def health() -> Dict[str, str]:
    return {"status": "ok"}

  @app.get("/ready")
  def ready(request: Request) -> Dict[str, str]:
    s = svc(request)
    try:
      s.vector_store.ping()
      s.graph_store.ping()
    except Exception as exc:
      raise HTTPException(status_code=503, detail=f"dependency unavailable: {type(exc).__name__}")
    return {"status": "ready"}

  @app.get("/stats", dependencies=[Depends(require_api_key)])
  def stats(request: Request) -> Dict[str, Any]:
    s = svc(request)
    return {"vector": s.vector_store.stats(), "graph": s.graph_store.stats()}

  @app.post("/ingest", dependencies=[Depends(require_api_key)])
  def ingest(body: IngestRequest, request: Request) -> Dict[str, Any]:
    s = svc(request)
    results = [s.ingestor.ingest(d.doc_id, d.text, d.metadata) for d in body.documents]
    return {"ingested": results}

  @app.post("/query", dependencies=[Depends(require_api_key)])
  def query(body: QueryRequest, request: Request) -> Dict[str, Any]:
    s = svc(request)
    do_eval = s.settings.evaluate_by_default if body.evaluate is None else body.evaluate
    started = time.perf_counter()
    state = s.graph.invoke({
        "query": body.query,
        "mode": body.mode,
        "top_k": body.top_k or s.settings.vector_top_k,
        "evaluate": do_eval,
    })

    sources = [
        {"chunk_id": h["id"], "doc_id": h["doc_id"], "origin": "vector", "score": h.get("score")}
        for h in state.get("vector_hits", [])
    ] + [
        {"chunk_id": h["id"], "doc_id": h["doc_id"], "origin": "graph", "score": None}
        for h in state.get("graph_hits", [])
    ]
    evaluation = state.get("evaluation")
    faithfulness = evaluation.get("faithfulness") if evaluation else None
    return {
        "query": body.query,
        "mode": body.mode,
        "generation": state["generation"],
        "faithfulness_score": faithfulness,
        "low_confidence": faithfulness is not None and faithfulness < s.settings.faithfulness_threshold,
        "evaluation": evaluation,
        "entities": state.get("entities", []),
        "graph_facts": [
            f"{f['subject']} {f['predicate']} {f['object']}" for f in state.get("graph_facts", [])
        ],
        "sources": sources,
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }

  return app


app = create_app()
