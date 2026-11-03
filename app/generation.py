"""Grounded answer synthesis with inline citations."""

from __future__ import annotations

import re

from app import prompts
from app.llm import LLMProtocol

_CITATION_RE = re.compile(r"\[(S\d+|G\d+)\]")


def cited_labels(answer: str) -> list[str]:
    return list(dict.fromkeys(_CITATION_RE.findall(answer)))


class AnswerGenerator:
    def __init__(self, llm: LLMProtocol):
        self.llm = llm

    def generate(
        self,
        question: str,
        passages: str,
        facts: str,
        unsupported_claims: list[str] | None = None,
    ) -> str:
        system = prompts.ANSWER_SYSTEM
        if unsupported_claims:
            bullet_list = "\n".join(f"- {c}" for c in unsupported_claims[:15])
            system += prompts.STRICT_ANSWER_ADDENDUM.format(unsupported=bullet_list)
        if passages == "(none)" and facts == "(none)":
            return (
                "I could not find any relevant information in the knowledge base to answer "
                "this question."
            )
        user = prompts.ANSWER_USER.format(question=question, passages=passages, facts=facts)
        return self.llm.complete(system, user, temperature=0.0).strip()
