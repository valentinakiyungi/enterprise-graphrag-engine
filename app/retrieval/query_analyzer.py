"""Query understanding: entities to link, keywords, and multi-hop detection."""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from app import prompts
from app.llm import LLMProtocol

logger = logging.getLogger(__name__)

_MULTI_HOP_HINTS = re.compile(
    r"\b(who (manages|owns|leads|runs)|which .* (that|whose|of the)|depend|upstream|downstream|"
    r"responsible for|reports? to|both|and also|through|via|indirect|chain|impact)\b",
    re.I,
)
_STOP = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "is", "are", "was",
    "were", "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "does", "do", "did",
    "that", "this", "with", "by", "from", "at", "be", "it", "its", "as", "has", "have",
}


class QueryAnalysis(BaseModel):
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    is_multi_hop: bool = False
    rewritten_query: str = ""


def heuristic_analysis(query: str) -> QueryAnalysis:
    """Cheap fallback used if the LLM call fails."""
    caps = re.findall(r"\b([A-Z][\w&-]*(?:\s+[A-Z][\w&-]*)*)", query)
    entities = [c for c in caps if c.lower() not in _STOP]
    words = [w for w in re.findall(r"[A-Za-z0-9-]+", query) if w.lower() not in _STOP]
    return QueryAnalysis(
        entities=list(dict.fromkeys(entities))[:6],
        keywords=list(dict.fromkeys(words))[:8],
        is_multi_hop=bool(_MULTI_HOP_HINTS.search(query)),
        rewritten_query=query,
    )


class QueryAnalyzer:
    def __init__(self, llm: LLMProtocol):
        self.llm = llm

    def analyze(self, query: str) -> QueryAnalysis:
        try:
            result = self.llm.complete_structured(
                prompts.QUERY_ANALYSIS_SYSTEM, query, QueryAnalysis
            )
        except Exception as exc:
            logger.warning("Query analysis failed (%s); using heuristics", exc)
            return heuristic_analysis(query)
        if not result.rewritten_query.strip():
            result.rewritten_query = query
        # Combine model judgement with lexical cues; either can flag multi-hop.
        result.is_multi_hop = result.is_multi_hop or bool(_MULTI_HOP_HINTS.search(query))
        return result
