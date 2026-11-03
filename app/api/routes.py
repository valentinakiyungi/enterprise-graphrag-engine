"""HTTP API."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request

from app.evaluation.benchmark import DEFAULT_DATASET, BenchmarkRunner, load_dataset
from app.graph import run_query
from app.schemas import (
    BatchDocumentsIn,
    BenchmarkRequest,
    Citation,
    DocumentIn,
    EvaluateRequest,
    QueryRequest,
    QueryResponse,
)

logger = logging.getLogger(__name__)

EVAL_DIR = DEFAULT_DATASET.parent

# In-memory status for background ingestion jobs (per process).
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def get_services(request: Request):
    svc = getattr(request.app.state, "services", None)
    if svc is None:
        detail = getattr(request.app.state, "startup_error", None) or "Services not initialised"
        raise HTTPException(status_code=503, detail=detail)
    return svc


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    expected = request.app.state.settings.api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])


# --- query ------------------------------------------------------------------------
@router.post("/query", response_model=QueryResponse, tags=["query"])
def query(req: QueryRequest, svc=Depends(get_services)) -> QueryResponse:
    started = time.perf_counter()
    try:
        state = run_query(
            svc, req.query, req.mode, top_k=req.top_k, max_hops=req.max_hops,
            evaluate=req.evaluate, reference=req.reference,
        )
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
    latency = int((time.perf_counter() - started) * 1000)
    evaluation = {k: v for k, v in (state.get("evaluation") or {}).items()}
    query_id = None
    try:
        query_id = svc.pg.log_query(
            req.query, req.mode, state.get("generation", ""), evaluation,
            state.get("attempts", 0), latency,
        )
    except Exception as exc:  # logging must never break answering
        logger.warning("Could not log query: %s", exc)
    return QueryResponse(
        query_id=query_id,
        query=req.query,
        mode=req.mode,
        answer=state.get("generation", ""),
        citations=[Citation(**c) for c in state.get("citations", [])],
        evaluation=evaluation,
        analysis=state.get("analysis", {}),
        seed_entities=(state.get("graph_results") or {}).get("seed_entities", []),
        attempts=state.get("attempts", 0),
        latency_ms=latency,
        trace=state.get("trace") if req.include_trace else None,
    )


# --- documents ----------------------------------------------------------------------
@router.post("/documents", tags=["documents"])
def ingest_document(doc: DocumentIn, svc=Depends(get_services)) -> dict[str, Any]:
    try:
        result = svc.ingestion.ingest(
            doc.title, doc.content, doc.source, doc.metadata, doc.document_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    return result.to_dict()


def _run_batch(job_id: str, svc, docs: list[DocumentIn]) -> None:
    for doc in docs:
        try:
            res = svc.ingestion.ingest(
                doc.title, doc.content, doc.source, doc.metadata, doc.document_id
            )
            entry = {"title": doc.title, "status": "ok", **res.to_dict()}
        except Exception as exc:
            logger.exception("Batch ingestion failed for %s", doc.title)
            entry = {"title": doc.title, "status": "error", "error": str(exc)}
        with _jobs_lock:
            _jobs[job_id]["results"].append(entry)
            _jobs[job_id]["done"] += 1
    with _jobs_lock:
        _jobs[job_id]["status"] = "completed"


@router.post("/documents/batch", tags=["documents"])
def ingest_batch(
    batch: BatchDocumentsIn, background: BackgroundTasks, svc=Depends(get_services)
) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id, "status": "running", "total": len(batch.documents),
            "done": 0, "results": [],
        }
    if batch.background:
        background.add_task(_run_batch, job_id, svc, batch.documents)
        return {"job_id": job_id, "status": "running", "total": len(batch.documents)}
    _run_batch(job_id, svc, batch.documents)
    return _jobs[job_id]


@router.get("/jobs/{job_id}", tags=["documents"])
def job_status(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(job)


@router.get("/documents", tags=["documents"])
def list_documents(limit: int = 100, offset: int = 0, svc=Depends(get_services)) -> list[dict]:
    return svc.pg.list_documents(min(limit, 500), offset)


@router.get("/documents/{doc_id}", tags=["documents"])
def get_document(doc_id: str, svc=Depends(get_services)) -> dict:
    doc = svc.pg.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/documents/{doc_id}", tags=["documents"])
def delete_document(doc_id: str, svc=Depends(get_services)) -> dict[str, Any]:
    if not svc.ingestion.delete(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": doc_id}


# --- knowledge graph ------------------------------------------------------------------
@router.get("/graph/entities/search", tags=["graph"])
def search_entities(q: str, k: int = 10, svc=Depends(get_services)) -> list[dict]:
    vec = svc.llm.embed_query(q)
    return svc.pg.search_entities(vec, min(k, 50), max_distance=2.0)


@router.get("/graph/entities/{entity_id}", tags=["graph"])
def entity_neighbourhood(entity_id: str, svc=Depends(get_services)) -> dict:
    data = svc.graph_store.neighbourhood(entity_id)
    if not data:
        raise HTTPException(status_code=404, detail="Entity not found")
    return data


# --- evaluation & benchmarks -------------------------------------------------------
@router.post("/evaluate", tags=["evaluation"])
def evaluate(req: EvaluateRequest, svc=Depends(get_services)) -> dict[str, Any]:
    metrics = ("faithfulness", "context_precision") + (
        ("answer_correctness",) if req.reference else ()
    )
    return svc.evaluator.evaluate(
        req.question, req.answer, req.contexts, req.reference, metrics=metrics
    )


def _run_benchmark(run_id: str, svc, items, req: BenchmarkRequest) -> None:
    try:
        summary, results = BenchmarkRunner(svc).run(
            items, req.modes, req.self_correct, req.concurrency
        )
        svc.pg.finish_eval_run(run_id, summary, results)
    except Exception as exc:
        logger.exception("Benchmark %s failed", run_id)
        svc.pg.finish_eval_run(run_id, None, None, error=str(exc))


@router.post("/benchmarks", tags=["evaluation"])
def start_benchmark(
    req: BenchmarkRequest, background: BackgroundTasks, svc=Depends(get_services)
) -> dict[str, Any]:
    if req.items:
        items = req.items
        for i, it in enumerate(items):
            if "question" not in it or "reference" not in it:
                raise HTTPException(422, f"items[{i}] needs 'question' and 'reference'")
            it.setdefault("id", f"q{i + 1}")
            it.setdefault("type", "single_hop")
    else:
        path = DEFAULT_DATASET
        if req.dataset_path:
            path = (EVAL_DIR / Path(req.dataset_path).name).resolve()
            if path.parent != EVAL_DIR.resolve() or not path.exists():
                raise HTTPException(404, f"Dataset not found in {EVAL_DIR.name}/")
        items = load_dataset(path)
    if req.limit:
        items = items[: req.limit]
    run_id = svc.pg.create_eval_run(
        req.name, {**req.model_dump(exclude={"items"}), "n_items": len(items)}
    )
    background.add_task(_run_benchmark, run_id, svc, items, req)
    return {"run_id": run_id, "status": "running", "items": len(items), "modes": req.modes}


@router.get("/benchmarks", tags=["evaluation"])
def list_benchmarks(limit: int = 20, svc=Depends(get_services)) -> list[dict]:
    return svc.pg.list_eval_runs(min(limit, 100))


@router.get("/benchmarks/{run_id}", tags=["evaluation"])
def get_benchmark(run_id: str, svc=Depends(get_services)) -> dict:
    try:
        uuid.UUID(run_id)
    except ValueError as exc:
        raise HTTPException(404, "Run not found") from exc
    run = svc.pg.get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


# --- ops ---------------------------------------------------------------------------
@router.get("/stats", tags=["ops"])
def stats(svc=Depends(get_services)) -> dict[str, Any]:
    out: dict[str, Any] = {"evaluation_engine": svc.evaluator.engine}
    try:
        out["postgres"] = svc.pg.stats()
    except Exception as exc:
        out["postgres"] = {"error": str(exc)}
    try:
        out["neo4j"] = svc.graph_store.stats()
    except Exception as exc:
        out["neo4j"] = {"error": str(exc)}
    return out


health_router = APIRouter(tags=["ops"])


@health_router.get("/health")
def health(request: Request) -> dict[str, Any]:
    svc = getattr(request.app.state, "services", None)
    if svc is None:
        return {
            "status": "unavailable",
            "error": getattr(request.app.state, "startup_error", "not initialised"),
        }
    checks: dict[str, Any] = {}
    for name, fn in (("postgres", svc.pg.health), ("neo4j", svc.graph_store.health)):
        try:
            checks[name] = "ok" if fn() else "fail"
        except Exception as exc:
            checks[name] = f"fail: {exc}"
    ok = all(v == "ok" for v in checks.values())
    return {"status": "ok" if ok else "degraded", "checks": checks,
            "evaluation_engine": svc.evaluator.engine}
