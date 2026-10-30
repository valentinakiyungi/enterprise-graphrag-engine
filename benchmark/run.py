"""Benchmark: pure vector RAG vs hybrid GraphRAG on multi-hop questions.

Usage (needs running Postgres + Neo4j and OPENAI_API_KEY):
  python -m benchmark.run --corpus benchmark/corpus --questions benchmark/questions.jsonl

Metrics per mode: answer accuracy (LLM judge; for unanswerable questions, correct = refused),
Ragas faithfulness and context precision, and hallucination rate
(answered, not refused, with faithfulness below FAITHFULNESS_THRESHOLD).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

from app.config import get_settings
from app.services import build_services
from app.text import is_refusal
from benchmark.metrics import compare, summarize, to_markdown

logger = logging.getLogger("benchmark")

JUDGE_PROMPT = (
    "You grade answers to questions. Given a question, a reference answer and a candidate answer,"
    " decide whether the candidate states the same key facts as the reference (extra detail is fine,"
    " contradictions or missing key facts are not). A refusal such as \"I don't know\" is incorrect."
)


class Verdict(BaseModel):
  correct: bool
  reason: str = ""


def load_questions(path: Path) -> List[Dict[str, Any]]:
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def judge(llm, question: str, reference: str, answer: str) -> bool:
  verdict = llm.with_structured_output(Verdict).invoke([
      ("system", JUDGE_PROMPT),
      ("human", f"Question: {question}\nReference answer: {reference}\nCandidate answer: {answer}"),
  ])
  return bool(verdict.correct)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--corpus", default="benchmark/corpus")
  parser.add_argument("--questions", default="benchmark/questions.jsonl")
  parser.add_argument("--out", default="benchmark/results")
  parser.add_argument("--skip-ingest", action="store_true")
  # The sample corpus is tiny: small chunks and a small top-k keep the vector baseline from
  # simply retrieving the whole corpus, which would make the comparison meaningless.
  parser.add_argument("--chunk-size", type=int, default=200)
  parser.add_argument("--top-k", type=int, default=2)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO)

  services = build_services()
  settings = get_settings()
  services.ingestor.chunk_size = args.chunk_size
  services.ingestor.chunk_overlap = 0
  try:
    if not args.skip_ingest:
      for path in sorted(Path(args.corpus).glob("*.txt")):
        print("ingest:", services.ingestor.ingest(path.stem, path.read_text()))

    rows: List[Dict[str, Any]] = []
    for q in load_questions(Path(args.questions)):
      for mode in ("vector", "hybrid"):
        state = services.graph.invoke({
            "query": q["question"], "mode": mode, "top_k": args.top_k,
            "evaluate": True, "reference": q.get("reference") or None,
        })
        answer = state["generation"]
        refused = is_refusal(answer)
        if q["answerable"]:
          correct = (not refused) and judge(services.llm, q["question"], q["reference"], answer)
        else:
          correct = refused  # the only right answer to an unanswerable question is to decline
        ev = state.get("evaluation") or {}
        faith = ev.get("faithfulness")
        rows.append({
            "id": q["id"], "mode": mode, "answerable": q["answerable"], "answer": answer,
            "refused": refused, "correct": correct,
            "faithfulness": faith, "context_precision": ev.get("context_precision"),
            "hallucinated": (not refused) and faith is not None and faith < settings.faithfulness_threshold,
        })
        print(f"{q['id']} {mode:6} correct={correct} faithfulness={faith}")

    summary = summarize(rows)
    comparison = compare(summary)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"summary": summary, "comparison": comparison, "rows": rows}, indent=2)
    )
    report = to_markdown(summary, comparison)
    (out_dir / "results.md").write_text(report + "\n")
    print("\n" + report)
  finally:
    services.close()


if __name__ == "__main__":
  main()
