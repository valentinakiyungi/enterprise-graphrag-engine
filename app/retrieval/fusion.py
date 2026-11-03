"""Rank fusion and context assembly (pure functions, no I/O)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from app.models import GraphFact, RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[RetrievedChunk]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse several ranked chunk lists with weighted Reciprocal Rank Fusion.

    score(d) = sum_r  w_r / (k + rank_r(d))

    The returned chunks carry the fused score and a ``retriever`` label listing
    every retriever that surfaced them (e.g. ``"graph+vector"``).
    """
    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    sources: dict[str, set[str]] = defaultdict(set)
    best: dict[str, RetrievedChunk] = {}
    for name, items in ranked_lists.items():
        w = weights.get(name, 1.0)
        if w <= 0:
            continue
        seen: set[str] = set()
        rank = 0
        for item in items:
            if item.id in seen:
                continue
            seen.add(item.id)
            rank += 1
            scores[item.id] += w / (k + rank)
            sources[item.id].add(name)
            best.setdefault(item.id, item)
    fused = [
        replace(best[cid], score=round(s, 6), retriever="+".join(sorted(sources[cid])))
        for cid, s in scores.items()
    ]
    fused.sort(key=lambda c: c.score, reverse=True)
    return fused


def fusion_weights(mode: str, is_multi_hop: bool) -> dict[str, float]:
    """Retriever weights per mode. Multi-hop questions lean on the graph."""
    if mode == "vector":
        return {"vector": 1.0, "keyword": 0.6, "graph": 0.0}
    if mode == "graph":
        return {"vector": 0.0, "keyword": 0.0, "graph": 1.0}
    return {"vector": 1.0, "keyword": 0.6, "graph": 1.3 if is_multi_hop else 0.9}


def select_facts(facts: list[GraphFact], limit: int) -> list[GraphFact]:
    seen: set[tuple[str, str, str]] = set()
    out: list[GraphFact] = []
    for f in sorted(facts, key=lambda f: (f.hop, -f.weight)):
        key = (f.source.lower(), f.relation, f.target.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
        if len(out) >= limit:
            break
    return out


def build_context(
    chunks: list[RetrievedChunk],
    facts: list[GraphFact],
    max_chunks: int,
    max_chars: int,
    max_facts: int = 25,
) -> tuple[str, str, list[dict], list[str]]:
    """Render labelled passages/facts within a character budget.

    Returns (passages_block, facts_block, citations, contexts) where
    ``contexts`` is the flat list of strings used for evaluation.
    """
    budget = max_chars
    passages: list[str] = []
    citations: list[dict] = []
    contexts: list[str] = []

    chosen_facts = select_facts(facts, max_facts)
    fact_lines = [f"[G{i}] {f.render()}" for i, f in enumerate(chosen_facts, 1)]
    facts_block = "\n".join(fact_lines) if fact_lines else "(none)"
    budget -= len(facts_block)

    for chunk in chunks[:max_chunks]:
        label = f"S{len(passages) + 1}"
        header = f"[{label}] (source: {chunk.title or chunk.document_id})"
        text = chunk.content
        if len(header) + len(text) + 3 > budget:
            room = budget - len(header) - 7
            if room < 200:
                break
            text = text[:room] + " ..."
        passages.append(f"{header}\n{text}")
        contexts.append(text)
        budget -= len(header) + len(text) + 3  # header newline + passage separator
        citations.append(
            {
                "label": label,
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "title": chunk.title,
                "retriever": chunk.retriever,
                "score": chunk.score,
                "snippet": chunk.content[:280],
            }
        )

    for i, f in enumerate(chosen_facts, 1):
        citations.append(
            {
                "label": f"G{i}",
                "fact": f.render(),
                "hop": f.hop,
                "retriever": "graph",
                "chunk_ids": f.chunk_ids,
            }
        )
    if fact_lines:
        contexts.append("Knowledge graph facts:\n" + "\n".join(f.render() for f in chosen_facts))

    passages_block = "\n\n".join(passages) if passages else "(none)"
    return passages_block, facts_block, citations, contexts
