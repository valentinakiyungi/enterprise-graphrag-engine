"""Benchmark harness: pure vector RAG vs. GraphRAG on a labelled question set.

Produces exactly the numbers needed to substantiate claims such as
"multi-hop accuracy +X% vs. vector RAG" and "hallucinations reduced by Y%".
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "data" / "eval" / "multihop_eval.json"


def load_dataset(path: str | Path | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else DEFAULT_DATASET
    items = json.loads(p.read_text(encoding="utf-8"))
    for i, item in enumerate(items):
        if "question" not in item or "reference" not in item:
            raise ValueError(f"Dataset item {i} needs 'question' and 'reference'")
        item.setdefault("id", f"q{i + 1}")
        item.setdefault("type", "single_hop")
    return items


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def _relative_change(new: float | None, base: float | None) -> float | None:
    if new is None or base is None or base == 0:
        return None
    return round((new - base) / base * 100, 1)


def summarize(results: list[dict[str, Any]], hallucination_threshold: float = 0.5) -> dict:
    """Aggregate per-mode metrics and compute GraphRAG-vs-vector deltas."""
    modes = sorted({r["mode"] for r in results})
    per_mode: dict[str, dict[str, Any]] = {}
    for mode in modes:
        rows = [r for r in results if r["mode"] == mode and not r.get("error")]
        multi = [r for r in rows if r.get("type") == "multi_hop"]
        faith = [r["metrics"].get("faithfulness") for r in rows]
        scored = [f for f in faith if f is not None]
        per_mode[mode] = {
            "n": len(rows),
            "errors": sum(1 for r in results if r["mode"] == mode and r.get("error")),
            "answer_accuracy": _mean([r["metrics"].get("answer_correctness") for r in rows]),
            "multi_hop_accuracy": _mean([r["metrics"].get("answer_correctness") for r in multi]),
            "faithfulness": _mean(faith),
            "context_precision": _mean([r["metrics"].get("context_precision") for r in rows]),
            "hallucination_rate": round(
                sum(1 for f in scored if f < hallucination_threshold) / len(scored), 4
            ) if scored else None,
            "avg_latency_ms": _mean([r.get("latency_ms") for r in rows]),
            "avg_attempts": _mean([r.get("attempts") for r in rows]),
        }

    comparison: dict[str, Any] = {}
    base, cand = per_mode.get("vector"), per_mode.get("hybrid")
    if base and cand:
        comparison = {
            "baseline": "vector",
            "candidate": "hybrid",
            "multi_hop_accuracy_improvement_pct": _relative_change(
                cand["multi_hop_accuracy"], base["multi_hop_accuracy"]
            ),
            "answer_accuracy_improvement_pct": _relative_change(
                cand["answer_accuracy"], base["answer_accuracy"]
            ),
            "faithfulness_improvement_pct": _relative_change(
                cand["faithfulness"], base["faithfulness"]
            ),
            "context_precision_improvement_pct": _relative_change(
                cand["context_precision"], base["context_precision"]
            ),
            "hallucination_reduction_pct": (
                None
                if base["hallucination_rate"] in (None, 0) or cand["hallucination_rate"] is None
                else round(
                    (base["hallucination_rate"] - cand["hallucination_rate"])
                    / base["hallucination_rate"] * 100, 1
                )
            ),
        }
    return {"per_mode": per_mode, "comparison": comparison}


class BenchmarkRunner:
    def __init__(self, svc):
        self.svc = svc

    def _run_one(self, item: dict, mode: str, self_correct: bool) -> dict[str, Any]:
        from app.graph import run_query

        started = time.perf_counter()
        row: dict[str, Any] = {
            "id": item["id"], "type": item["type"], "mode": mode,
            "question": item["question"], "reference": item["reference"],
        }
        try:
            state = run_query(
                self.svc, item["question"], mode,
                evaluate=self_correct, reference=item["reference"],
            )
            row["latency_ms"] = int((time.perf_counter() - started) * 1000)
            row["answer"] = state.get("generation", "")
            row["attempts"] = state.get("attempts", 0)
            row["metrics"] = self.svc.evaluator.evaluate(
                item["question"], row["answer"], state.get("contexts", []),
                reference=item["reference"],
                metrics=("faithfulness", "context_precision", "answer_correctness"),
            )
        except Exception as exc:
            logger.exception("Benchmark item %s/%s failed", item["id"], mode)
            row.update(error=str(exc), metrics={})
        return row

    def run(
        self,
        items: list[dict[str, Any]],
        modes: list[str] = ("vector", "hybrid"),
        self_correct: bool = True,
        concurrency: int = 4,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        jobs = [(item, mode) for item in items for mode in modes]
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            results = list(pool.map(lambda j: self._run_one(j[0], j[1], self_correct), jobs))
        summary = summarize(results, self.svc.settings.hallucination_threshold)
        summary["config"] = {
            "modes": list(modes), "items": len(items), "self_correct": self_correct,
            "eval_engine": self.svc.evaluator.engine,
            "chat_model": self.svc.settings.openai_chat_model,
            "eval_model": self.svc.settings.eval_model,
        }
        return summary, results
