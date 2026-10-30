"""Aggregation for benchmark rows (stdlib only)."""

from typing import Any, Dict, List, Optional


def _mean(values: List[Optional[float]]) -> Optional[float]:
  vals = [v for v in values if v is not None]
  return round(sum(vals) / len(vals), 4) if vals else None


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
  """rows: dicts with mode, correct, faithfulness, context_precision, hallucinated."""
  out: Dict[str, Dict[str, Any]] = {}
  for mode in sorted({r["mode"] for r in rows}):
    mrows = [r for r in rows if r["mode"] == mode]
    n = len(mrows)
    out[mode] = {
        "questions": n,
        "accuracy": round(sum(1 for r in mrows if r["correct"]) / n, 4) if n else None,
        "mean_faithfulness": _mean([r.get("faithfulness") for r in mrows]),
        "mean_context_precision": _mean([r.get("context_precision") for r in mrows]),
        "hallucination_rate": round(sum(1 for r in mrows if r.get("hallucinated")) / n, 4) if n else None,
    }
  return out


def relative_change(new: Optional[float], base: Optional[float]) -> Optional[float]:
  """(new - base) / base; None when undefined."""
  if new is None or base in (None, 0):
    return None
  return round((new - base) / base, 4)


def compare(summary: Dict[str, Dict[str, Any]], base: str = "vector", new: str = "hybrid") -> Dict[str, Any]:
  if base not in summary or new not in summary:
    return {}
  b, n = summary[base], summary[new]
  return {
      "accuracy_relative_change": relative_change(n["accuracy"], b["accuracy"]),
      "hallucination_rate_relative_change": relative_change(n["hallucination_rate"], b["hallucination_rate"]),
      "faithfulness_relative_change": relative_change(n["mean_faithfulness"], b["mean_faithfulness"]),
      "context_precision_relative_change": relative_change(n["mean_context_precision"], b["mean_context_precision"]),
  }


def to_markdown(summary: Dict[str, Dict[str, Any]], comparison: Dict[str, Any]) -> str:
  def fmt(v):
    return "n/a" if v is None else f"{v:.3f}"

  lines = [
      "| Mode | Questions | Accuracy | Faithfulness | Context precision | Hallucination rate |",
      "|---|---|---|---|---|---|",
  ]
  for mode, s in summary.items():
    lines.append(
        f"| {mode} | {s['questions']} | {fmt(s['accuracy'])} | {fmt(s['mean_faithfulness'])} |"
        f" {fmt(s['mean_context_precision'])} | {fmt(s['hallucination_rate'])} |"
    )
  if comparison:
    lines += ["", "Relative change, hybrid vs vector (positive = higher):"]
    lines += [f"- {k}: {'n/a' if v is None else f'{v:+.1%}'}" for k, v in comparison.items()]
  return "\n".join(lines)
