"""LLM-as-judge implementations of the core RAG metrics.

These mirror the Ragas definitions and are used automatically when Ragas is
not importable or fails at runtime, so evaluation never silently disappears.
Scoring math is kept in pure functions for unit testing.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app import prompts
from app.llm import LLMProtocol

logger = logging.getLogger(__name__)


# -- schemas --------------------------------------------------------------------
class ClaimVerdict(BaseModel):
    claim: str
    supported: bool


class FaithfulnessJudgement(BaseModel):
    claims: list[ClaimVerdict] = Field(default_factory=list)


class ContextVerdict(BaseModel):
    index: int
    useful: bool


class ContextPrecisionJudgement(BaseModel):
    verdicts: list[ContextVerdict] = Field(default_factory=list)


class CorrectnessJudgement(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


# -- pure scoring ---------------------------------------------------------------
def faithfulness_score(verdicts: list[bool]) -> float:
    """Fraction of answer claims supported by the context (1.0 if no claims)."""
    if not verdicts:
        return 1.0
    return sum(verdicts) / len(verdicts)


def average_precision(relevance: list[bool]) -> float:
    """Ragas-style context precision: mean of precision@k over relevant ranks."""
    hits = 0
    total = 0.0
    for k, rel in enumerate(relevance, 1):
        if rel:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


# -- judge ------------------------------------------------------------------------
class LLMJudge:
    def __init__(self, llm: LLMProtocol):
        self.llm = llm

    def faithfulness(self, answer: str, contexts: list[str]) -> tuple[float, list[str]]:
        judgement = self.llm.complete_structured(
            prompts.FAITHFULNESS_SYSTEM,
            prompts.FAITHFULNESS_USER.format(context="\n\n".join(contexts), answer=answer),
            FaithfulnessJudgement,
        )
        verdicts = [c.supported for c in judgement.claims]
        unsupported = [c.claim for c in judgement.claims if not c.supported]
        return faithfulness_score(verdicts), unsupported

    def context_precision(
        self, question: str, contexts: list[str], answer: str, reference: str | None = None
    ) -> float:
        if not contexts:
            return 0.0
        items = "\n\n".join(f"[{i}] {c[:1500]}" for i, c in enumerate(contexts))
        judgement = self.llm.complete_structured(
            prompts.CONTEXT_PRECISION_SYSTEM,
            prompts.CONTEXT_PRECISION_USER.format(
                question=question, answer=reference or answer, items=items
            ),
            ContextPrecisionJudgement,
        )
        by_index = {v.index: v.useful for v in judgement.verdicts}
        return average_precision([by_index.get(i, False) for i in range(len(contexts))])

    def answer_correctness(self, question: str, answer: str, reference: str) -> float:
        judgement = self.llm.complete_structured(
            prompts.CORRECTNESS_SYSTEM,
            prompts.CORRECTNESS_USER.format(question=question, reference=reference, answer=answer),
            CorrectnessJudgement,
        )
        return float(judgement.score)
