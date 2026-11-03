"""Thin OpenAI wrapper (via langchain-openai) used by every component.

Everything that talks to a model goes through :class:`LLMClient`, which keeps
the rest of the code base easy to test with a fake implementation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from app.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMProtocol(Protocol):
    def complete(self, system: str, user: str, *, temperature: float | None = None) -> str: ...

    def complete_structured(self, system: str, user: str, schema: type[T]) -> T: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _extract_json(text: str) -> Any:
    """Best-effort JSON extraction from a model reply (handles ```json fences)."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidate = fence.group(1) if fence else text
    start = min([i for i in (candidate.find("{"), candidate.find("[")) if i != -1], default=-1)
    if start == -1:
        raise ValueError("No JSON object found in model output")
    return json.loads(candidate[start:])


class LLMClient:
    """OpenAI chat + embeddings with structured output support."""

    def __init__(self, settings: Settings, *, model: str | None = None):
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.settings = settings
        common: dict[str, Any] = {
            "api_key": settings.openai_api_key,
            "max_retries": settings.llm_max_retries,
            "timeout": settings.llm_timeout_s,
        }
        if settings.openai_base_url:
            common["base_url"] = settings.openai_base_url
        self.model_name = model or settings.openai_chat_model
        self.chat = ChatOpenAI(
            model=self.model_name, temperature=settings.llm_temperature, **common
        )
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            dimensions=settings.embedding_dim
            if settings.openai_embedding_model.startswith("text-embedding-3")
            else None,
            **common,
        )

    # -- chat ---------------------------------------------------------------
    def complete(self, system: str, user: str, *, temperature: float | None = None) -> str:
        llm = self.chat if temperature is None else self.chat.bind(temperature=temperature)
        msg = llm.invoke([("system", system), ("human", user)])
        return str(msg.content)

    def complete_structured(self, system: str, user: str, schema: type[T]) -> T:
        try:
            # function calling tolerates optional fields / numeric bounds that
            # OpenAI's strict json_schema mode rejects
            structured = self.chat.with_structured_output(schema, method="function_calling")
            result = structured.invoke([("system", system), ("human", user)])
            if isinstance(result, schema):
                return result
            return schema.model_validate(result)
        except Exception as exc:  # fall back to plain JSON prompting
            logger.warning("Structured output failed (%s); falling back to JSON parse", exc)
            hint = json.dumps(schema.model_json_schema())
            raw = self.complete(
                system + "\nRespond ONLY with JSON matching this schema:\n" + hint, user
            )
            return schema.model_validate(_extract_json(raw))

    # -- embeddings -----------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        bs = self.settings.embedding_batch_size
        for i in range(0, len(texts), bs):
            out.extend(self.embeddings.embed_documents(texts[i : i + bs]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)
