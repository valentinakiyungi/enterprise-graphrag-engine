"""LangGraph node implementations.

Each node is a small closure over :class:`~app.services.Services` that reads
the shared :class:`~app.state.GraphRAGState` and returns a partial update.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from app.models import GraphFact, RetrievedChunk
from app.retrieval.fusion import build_context, fusion_weights, reciprocal_rank_fusion
from app.state import GraphRAGState

logger = logging.getLogger(__name__)

Node = Callable[[GraphRAGState], dict[str, Any]]


def _trace(node: str, started: float, **info: Any) -> list[dict[str, Any]]:
    return [{"node": node, "ms": round((time.perf_counter() - started) * 1000, 1), **info}]


def make_nodes(svc) -> dict[str, Node]:
    s = svc.settings

    def analyze_query(state: GraphRAGState) -> dict[str, Any]:
        t = time.perf_counter()
        analysis = svc.analyzer.analyze(state["query"])
        return {
            "analysis": analysis.model_dump(),
            "attempts": 0,
            "trace": _trace(
                "analyze_query", t, entities=analysis.entities, multi_hop=analysis.is_multi_hop
            ),
        }

    def vector_retrieve(state: GraphRAGState) -> dict[str, Any]:
        t = time.perf_counter()
        if state.get("mode", "hybrid") == "graph":
            return {"vector_results": [], "keyword_results": [],
                    "trace": _trace("vector_retrieve", t, skipped=True)}
        analysis = state.get("analysis", {})
        query = analysis.get("rewritten_query") or state["query"]
        try:
            res = svc.vector_retriever.retrieve(
                query, analysis.get("keywords", []), state.get("top_k", s.vector_top_k)
            )
        except Exception as exc:
            logger.exception("Vector retrieval failed")
            return {"vector_results": [], "keyword_results": [],
                    "trace": _trace("vector_retrieve", t, error=str(exc))}
        return {
            "vector_results": [c.to_dict() for c in res.dense],
            "keyword_results": [c.to_dict() for c in res.keyword],
            "trace": _trace("vector_retrieve", t, dense=len(res.dense), keyword=len(res.keyword)),
        }

    def graph_retrieve(state: GraphRAGState) -> dict[str, Any]:
        t = time.perf_counter()
        if state.get("mode", "hybrid") == "vector":
            return {"graph_results": {}, "trace": _trace("graph_retrieve", t, skipped=True)}
        analysis = state.get("analysis", {})
        try:
            res = svc.graph_retriever.retrieve(
                state["query"], analysis.get("entities", []),
                state.get("max_hops", s.graph_max_hops),
            )
        except Exception as exc:
            logger.exception("Graph retrieval failed")
            return {"graph_results": {}, "trace": _trace("graph_retrieve", t, error=str(exc))}
        return {
            "graph_results": {
                "seed_entities": res.seed_entities,
                "facts": [f.to_dict() for f in res.facts],
                "chunks": [c.to_dict() for c in res.chunks],
            },
            "trace": _trace(
                "graph_retrieve", t,
                seeds=[e["name"] for e in res.seed_entities],
                facts=len(res.facts), chunks=len(res.chunks),
            ),
        }

    def fuse_context(state: GraphRAGState) -> dict[str, Any]:
        t = time.perf_counter()
        mode = state.get("mode", "hybrid")
        graph = state.get("graph_results") or {}
        lists = {
            "vector": [RetrievedChunk(**c) for c in state.get("vector_results", [])],
            "keyword": [RetrievedChunk(**c) for c in state.get("keyword_results", [])],
            "graph": [RetrievedChunk(**c) for c in graph.get("chunks", [])],
        }
        weights = fusion_weights(mode, bool(state.get("analysis", {}).get("is_multi_hop")))
        fused = reciprocal_rank_fusion(lists, k=s.rrf_k, weights=weights)
        facts = [GraphFact(**f) for f in graph.get("facts", [])] if mode != "vector" else []
        passages, facts_block, citations, contexts = build_context(
            fused, facts,
            max_chunks=s.final_context_chunks + 2 * state.get("attempts", 0),
            max_chars=s.max_context_chars,
        )
        return {
            "fused_chunks": [c.to_dict() for c in fused[: s.final_context_chunks]],
            "passages": passages,
            "facts": facts_block,
            "citations": citations,
            "contexts": contexts,
            "trace": _trace("fuse_context", t, chunks=len(fused), facts=len(facts)),
        }

    def generate_answer(state: GraphRAGState) -> dict[str, Any]:
        t = time.perf_counter()
        unsupported = (
            state.get("unsupported_claims") if state.get("correction") == "regenerate" else None
        )
        answer = svc.generator.generate(
            state["query"], state.get("passages", "(none)"), state.get("facts", "(none)"),
            unsupported,
        )
        return {"generation": answer, "trace": _trace("generate_answer", t, strict=bool(unsupported))}

    def evaluate_answer(state: GraphRAGState) -> dict[str, Any]:
        t = time.perf_counter()
        if not state.get("evaluate", s.evaluate_on_query):
            return {"evaluation": {}, "trace": _trace("evaluate_answer", t, skipped=True)}
        result = svc.evaluator.evaluate(
            state["query"], state["generation"], state.get("contexts", []),
            reference=state.get("reference"),
        )
        return {
            "evaluation": result,
            "unsupported_claims": result.get("unsupported_claims", []),
            "trace": _trace(
                "evaluate_answer", t,
                faithfulness=result.get("faithfulness"),
                context_precision=result.get("context_precision"),
                engine=result.get("engine"),
            ),
        }

    def expand_retrieval(state: GraphRAGState) -> dict[str, Any]:
        t = time.perf_counter()
        hops = min(state.get("max_hops", s.graph_max_hops) + 1, 4)
        top_k = state.get("top_k", s.vector_top_k) + 4
        return {
            "attempts": state.get("attempts", 0) + 1,
            "max_hops": hops,
            "top_k": top_k,
            "correction": "expand",
            "trace": _trace("expand_retrieval", t, max_hops=hops, top_k=top_k),
        }

    def prepare_regeneration(state: GraphRAGState) -> dict[str, Any]:
        t = time.perf_counter()
        claims = state.get("unsupported_claims") or svc.evaluator.unsupported_claims(
            state["generation"], state.get("contexts", [])
        )
        return {
            "attempts": state.get("attempts", 0) + 1,
            "unsupported_claims": claims,
            "correction": "regenerate",
            "trace": _trace("prepare_regeneration", t, unsupported=len(claims)),
        }

    return {
        "analyze_query": analyze_query,
        "vector_retrieve": vector_retrieve,
        "graph_retrieve": graph_retrieve,
        "fuse_context": fuse_context,
        "generate_answer": generate_answer,
        "evaluate_answer": evaluate_answer,
        "expand_retrieval": expand_retrieval,
        "prepare_regeneration": prepare_regeneration,
    }


def route_after_evaluation(settings) -> Callable[[GraphRAGState], str]:
    """Self-correction policy.

    * faithful enough, evaluation disabled or retry budget spent -> finish
    * unfaithful and context was poor  -> widen retrieval (more hops / chunks)
    * unfaithful but context was good  -> regenerate under strict grounding
    """

    def route(state: GraphRAGState) -> str:
        ev = state.get("evaluation") or {}
        faith = ev.get("faithfulness")
        if faith is None or state.get("attempts", 0) >= settings.max_correction_attempts:
            return "end"
        if faith >= settings.faithfulness_threshold:
            return "end"
        precision = ev.get("context_precision")
        if (
            precision is not None
            and precision < settings.context_precision_threshold
            and state.get("correction") != "expand"
        ):
            return "expand"
        return "regenerate"

    return route
