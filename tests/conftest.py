from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.metrics import (  # noqa: E402
    ClaimVerdict,
    ContextPrecisionJudgement,
    ContextVerdict,
    CorrectnessJudgement,
    FaithfulnessJudgement,
)
from app.ingestion.extractor import (  # noqa: E402
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from app.retrieval.query_analyzer import QueryAnalysis  # noqa: E402
from tests.fakes import FakeGraph, FakeLLM, FakePg, make_settings  # noqa: E402

# A tiny corpus where the answer needs a 3-hop chain:
#   Insight Board -DEPENDS_ON-> Orion Lake -HOSTED_BY-> Nimbus Cloud -SPONSORED_BY-> Ada Lovelace
CORPUS = {
    "Systems": (
        "# Systems\n\nInsight Board DEPENDS_ON Orion Lake. The Insight Board is a sales dashboard.\n\n"
        "Orion Lake HOSTED_BY Nimbus Cloud. Orion Lake is the analytics platform."
    ),
    "Vendors": (
        "# Vendors\n\nNimbus Cloud SPONSORED_BY Ada Lovelace. Nimbus Cloud provides compute "
        "with a 99.9% SLA."
    ),
}

REL_RE = re.compile(r"([A-Z][\w]*(?: [A-Z][\w]*)*) ([A-Z_]{4,}) ([A-Z][\w]*(?: [A-Z][\w]*)*)\.")


def rule_based_structured(system: str, user: str, schema: type):
    """Deterministic stand-in for the LLM's structured outputs."""
    if schema is ExtractionResult:
        text = user.split('"""')[1] if '"""' in user else user
        ents, rels = {}, []
        for src, rtype, tgt in REL_RE.findall(text):
            for n in (src, tgt):
                ents[n] = ExtractedEntity(name=n, type="SYSTEM", description=f"{n} entity")
            rels.append(ExtractedRelationship(source=src, target=tgt, type=rtype,
                                              description=f"{src} {rtype} {tgt}", strength=8))
        return ExtractionResult(entities=list(ents.values()), relationships=rels)
    if schema is QueryAnalysis:
        names = re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", user)
        return QueryAnalysis(entities=names, keywords=names, is_multi_hop=True,
                             rewritten_query=user)
    if schema is FaithfulnessJudgement:
        answer = user.split("Answer:")[-1]
        bad = "invented" in answer.lower()
        return FaithfulnessJudgement(claims=[ClaimVerdict(claim="claim one", supported=True),
                                             ClaimVerdict(claim="invented claim", supported=not bad)])
    if schema is ContextPrecisionJudgement:
        n = len(re.findall(r"^\[\d+\]", user, re.M))
        return ContextPrecisionJudgement(verdicts=[ContextVerdict(index=i, useful=True)
                                                   for i in range(n)])
    if schema is CorrectnessJudgement:
        return CorrectnessJudgement(score=1.0 if "ada lovelace" in user.lower().split(
            "candidate answer:")[-1] else 0.0)
    return None


def grounded_answer(system: str, user: str) -> str:
    if "Ada Lovelace" in user and "IMPORTANT" in system:
        return "Ada Lovelace sponsors Nimbus Cloud [G3]."
    if "Ada Lovelace" in user:
        # first draft deliberately contains an unsupported claim
        return "Ada Lovelace sponsors Nimbus Cloud [G3]. She also invented the SLA."
    return "The context does not say."


@pytest.fixture
def settings():
    return make_settings()


@pytest.fixture
def fake_llm():
    return FakeLLM(structured=rule_based_structured, answer=grounded_answer)


@pytest.fixture
def services(settings, fake_llm):
    from app.evaluation.evaluator import Evaluator
    from app.services import Services

    svc = Services(settings=settings, llm=fake_llm, pg=FakePg(), graph_store=FakeGraph(),
                   evaluator=Evaluator(settings, fake_llm))
    for title, text in CORPUS.items():
        svc.ingestion.ingest(title, text, source=f"{title}.md")
    return svc
