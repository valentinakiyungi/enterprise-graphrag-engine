"""Ragas-backed evaluation with an automatic LLM-judge fallback.

Metrics
-------
faithfulness        share of answer claims supported by the retrieved context
context_precision   ranking quality of retrieved context (useful items ranked first)
answer_correctness  agreement with a reference answer (benchmarks only)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
from typing import Any, Coroutine

from app.config import Settings
from app.evaluation.metrics import LLMJudge
from app.llm import LLMProtocol

logger = logging.getLogger(__name__)


def _run_async(coro: Coroutine) -> Any:
    """Run a coroutine from sync code, even if called inside an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _clean(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else round(v, 4)


class _RagasBackend:
    def __init__(self, settings: Settings):
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithoutReference,
            LLMContextPrecisionWithReference,
        )

        kwargs: dict[str, Any] = {"api_key": settings.openai_api_key, "temperature": 0}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        llm = LangchainLLMWrapper(ChatOpenAI(model=settings.eval_model, **kwargs))
        self.faithfulness = Faithfulness(llm=llm)
        self.cp_with_ref = LLMContextPrecisionWithReference(llm=llm)
        self.cp_without_ref = LLMContextPrecisionWithoutReference(llm=llm)

    async def _score(
        self, question: str, answer: str, contexts: list[str], reference: str | None,
        metrics: tuple[str, ...],
    ) -> dict[str, float | None]:
        from ragas import SingleTurnSample

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts or [""],
            reference=reference,
        )
        tasks: dict[str, Any] = {}
        if "faithfulness" in metrics:
            tasks["faithfulness"] = self.faithfulness.single_turn_ascore(sample)
        if "context_precision" in metrics:
            metric = self.cp_with_ref if reference else self.cp_without_ref
            tasks["context_precision"] = metric.single_turn_ascore(sample)
        values = await asyncio.gather(*tasks.values())
        return {k: _clean(v) for k, v in zip(tasks.keys(), values)}

    def score(self, question, answer, contexts, reference, metrics) -> dict[str, float | None]:
        return _run_async(self._score(question, answer, contexts, reference, metrics))


class Evaluator:
    def __init__(self, settings: Settings, judge_llm: LLMProtocol):
        self.settings = settings
        self.judge = LLMJudge(judge_llm)
        self._ragas: _RagasBackend | None = None
        self.engine = "llm_judge"
        if settings.eval_engine in ("auto", "ragas"):
            try:
                self._ragas = _RagasBackend(settings)
                self.engine = "ragas"
            except Exception as exc:
                if settings.eval_engine == "ragas":
                    raise
                logger.warning("Ragas unavailable (%s); using built-in LLM judge", exc)
        logger.info("Evaluation engine: %s", self.engine)

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        reference: str | None = None,
        metrics: tuple[str, ...] = ("faithfulness", "context_precision"),
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"engine": self.engine, "unsupported_claims": []}
        scores: dict[str, float | None] = {}
        if self._ragas is not None:
            try:
                scores = self._ragas.score(question, answer, contexts, reference, metrics)
            except Exception as exc:
                logger.warning("Ragas scoring failed (%s); falling back to LLM judge", exc)
                result["engine"] = "llm_judge(fallback)"
                scores = {}
        try:
            if "faithfulness" in metrics and scores.get("faithfulness") is None:
                f, unsupported = self.judge.faithfulness(answer, contexts)
                scores["faithfulness"] = round(f, 4)
                result["unsupported_claims"] = unsupported
            if "context_precision" in metrics and scores.get("context_precision") is None:
                scores["context_precision"] = round(
                    self.judge.context_precision(question, contexts, answer, reference), 4
                )
            if "answer_correctness" in metrics and reference:
                scores["answer_correctness"] = round(
                    self.judge.answer_correctness(question, answer, reference), 4
                )
        except Exception as exc:
            logger.error("Evaluation failed: %s", exc)
            result["error"] = str(exc)
        result.update(scores)
        faith = result.get("faithfulness")
        result["hallucinated"] = (
            faith is not None and faith < self.settings.hallucination_threshold
        )
        return result

    def unsupported_claims(self, answer: str, contexts: list[str]) -> list[str]:
        try:
            return self.judge.faithfulness(answer, contexts)[1]
        except Exception as exc:
            logger.warning("Could not extract unsupported claims: %s", exc)
            return []
