"""Public API request/response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Mode = Literal["hybrid", "vector", "graph"]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=4000)
    mode: Mode = "hybrid"
    top_k: int | None = Field(default=None, ge=1, le=50)
    max_hops: int | None = Field(default=None, ge=1, le=4)
    evaluate: bool | None = Field(
        default=None, description="Run Ragas evaluation + self-correction (default from config)."
    )
    reference: str | None = Field(
        default=None, description="Optional ground-truth answer for reference-based metrics."
    )
    include_trace: bool = False


class Citation(BaseModel):
    label: str
    retriever: str
    title: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    snippet: str | None = None
    fact: str | None = None
    hop: int | None = None
    score: float | None = None


class QueryResponse(BaseModel):
    query_id: str | None
    query: str
    mode: Mode
    answer: str
    citations: list[Citation]
    evaluation: dict[str, Any]
    analysis: dict[str, Any]
    seed_entities: list[dict[str, Any]]
    attempts: int
    latency_ms: int
    trace: list[dict[str, Any]] | None = None


class DocumentIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]{1,128}$")


class BatchDocumentsIn(BaseModel):
    documents: list[DocumentIn] = Field(..., min_length=1, max_length=200)
    background: bool = True


class EvaluateRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    reference: str | None = None


class BenchmarkRequest(BaseModel):
    name: str | None = None
    dataset_path: str | None = Field(
        default=None, description="Server-side JSON path; defaults to data/eval/multihop_eval.json"
    )
    items: list[dict[str, Any]] | None = Field(
        default=None, description="Inline dataset: [{question, reference, type}]"
    )
    modes: list[Mode] = Field(default_factory=lambda: ["vector", "hybrid"])
    limit: int | None = Field(default=None, ge=1)
    self_correct: bool = True
    concurrency: int = Field(default=4, ge=1, le=16)
