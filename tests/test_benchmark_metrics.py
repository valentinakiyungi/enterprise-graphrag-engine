from benchmark.metrics import compare, relative_change, summarize, to_markdown


def _rows():
  return [
      {"mode": "vector", "correct": True, "faithfulness": 0.9, "context_precision": 0.5, "hallucinated": False},
      {"mode": "vector", "correct": False, "faithfulness": 0.2, "context_precision": 0.3, "hallucinated": True},
      {"mode": "hybrid", "correct": True, "faithfulness": 0.9, "context_precision": 0.7, "hallucinated": False},
      {"mode": "hybrid", "correct": True, "faithfulness": None, "context_precision": 0.5, "hallucinated": False},
  ]


def test_summarize():
  s = summarize(_rows())
  assert s["vector"]["accuracy"] == 0.5
  assert s["hybrid"]["accuracy"] == 1.0
  assert s["vector"]["hallucination_rate"] == 0.5
  assert s["hybrid"]["hallucination_rate"] == 0.0
  assert s["hybrid"]["mean_faithfulness"] == 0.9  # None values are ignored
  assert s["vector"]["mean_context_precision"] == 0.4


def test_relative_change():
  assert relative_change(1.0, 0.5) == 1.0
  assert relative_change(0.0, 0.5) == -1.0
  assert relative_change(1.0, 0) is None
  assert relative_change(None, 0.5) is None


def test_compare_and_markdown():
  s = summarize(_rows())
  c = compare(s)
  assert c["accuracy_relative_change"] == 1.0
  assert c["hallucination_rate_relative_change"] == -1.0
  md = to_markdown(s, c)
  assert "| hybrid |" in md and "+100.0%" in md


def test_compare_missing_mode():
  assert compare(summarize([{"mode": "vector", "correct": True}])) == {}
