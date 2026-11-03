"""LangGraph state for one GraphRAG request."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

RetrievalMode = Literal["hybrid", "vector", "graph"]


class GraphRAGState(TypedDict, total=False):
    # --- request ---------------------------------------------------------------
    query: str
    mode: RetrievalMode
    top_k: int
    max_hops: int
    evaluate: bool
    reference: str | None

    # --- query understanding -----------------------------------------------------
    analysis: dict[str, Any]

    # --- retrieval (written by parallel branches -> distinct keys) ---------------
    vector_results: list[dict[str, Any]]
    keyword_results: list[dict[str, Any]]
    graph_results: dict[str, Any]

    # --- fused context ---------------------------------------------------------
    fused_chunks: list[dict[str, Any]]
    passages: str
    facts: str
    contexts: list[str]
    citations: list[dict[str, Any]]

    # --- generation & evaluation -----------------------------------------------
    generation: str
    evaluation: dict[str, Any]
    unsupported_claims: list[str]
    attempts: int
    correction: str

    # --- observability: parallel nodes append, so use an additive reducer -------
    trace: Annotated[list[dict[str, Any]], operator.add]
