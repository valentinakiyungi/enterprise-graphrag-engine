"""Automated Ragas evaluation: faithfulness and context precision.

Written against the ragas 0.2/0.3 API (SingleTurnSample + metric.single_turn_ascore).
Fresh LLM clients are created per call because each evaluation runs in its own
event loop (asyncio.run), and async HTTP clients must not be shared across loops.
"""

import asyncio
import logging
import math
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _clean(score: Any) -> Optional[float]:
  try:
    f = float(score)
  except (TypeError, ValueError):
    return None
  return None if math.isnan(f) else round(f, 4)


class RagasEvaluator:

  def __init__(self, llm_factory: Callable[[], Any]):
    self._llm_factory = llm_factory

  def evaluate(
      self,
      question: str,
      answer: str,
      contexts: List[str],
      reference: Optional[str] = None,
  ) -> Dict[str, Any]:
    if not contexts or not answer:
      return {"faithfulness": None, "context_precision": None, "error": "no contexts or answer to evaluate"}
    try:
      return asyncio.run(self._aevaluate(question, answer, contexts, reference))
    except Exception as exc:  # evaluation must never break the query path
      logger.exception("Ragas evaluation failed")
      return {"faithfulness": None, "context_precision": None, "error": f"{type(exc).__name__}: {exc}"}

  async def _aevaluate(self, question, answer, contexts, reference) -> Dict[str, Any]:
    from ragas import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextPrecisionWithReference,
    )

    llm = LangchainLLMWrapper(self._llm_factory())
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference,
    )
    faithfulness = Faithfulness(llm=llm)
    precision_cls = LLMContextPrecisionWithReference if reference else LLMContextPrecisionWithoutReference
    precision = precision_cls(llm=llm)

    f_score = await faithfulness.single_turn_ascore(sample)
    p_score = await precision.single_turn_ascore(sample)
    return {
        "faithfulness": _clean(f_score),
        "context_precision": _clean(p_score),
        "precision_basis": "reference" if reference else "response",
    }
