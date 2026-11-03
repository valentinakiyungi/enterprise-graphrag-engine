"""FastAPI entrypoint: ``uvicorn app.main:app --reload``."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app import __version__
from app.api import health_router, router, routes
from app.config import get_settings
from app.schemas import QueryResponse

STATIC_DIR = Path(__file__).parent / "static"


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "openai", "urllib3", "neo4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings
    log = logging.getLogger("app")
    if getattr(app.state, "services", None) is None:  # tests may inject services
        try:
            from app.services import create_services

            app.state.services = create_services(settings)
            log.info("GraphRAG services ready")
        except Exception as exc:
            # Keep serving /health so operators can see what is wrong.
            app.state.services = None
            app.state.startup_error = f"{type(exc).__name__}: {exc}"
            log.exception("Service initialisation failed")
    yield
    if getattr(app.state, "services", None) is not None:
        app.state.services.close()


def create_app(services=None) -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Production GraphRAG: pgvector semantic search fused with Neo4j multi-hop "
            "graph traversal, orchestrated by LangGraph, with Ragas-gated self-correction."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.services = services
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(router)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    # Backwards-compatible v1 endpoint (POST /query) served by the same handler.
    app.add_api_route(
        "/query",
        routes.query,
        methods=["POST"],
        response_model=QueryResponse,
        dependencies=[Depends(routes.require_api_key)],
        include_in_schema=False,
    )
    return app


app = create_app()
