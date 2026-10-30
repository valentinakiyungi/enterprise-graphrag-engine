import pytest

from app.text import REFUSAL_TEXT, chunk_text, is_refusal, normalize_entity


def test_normalize_entity():
  assert normalize_entity("  Project   ATLAS ") == "project atlas"
  assert normalize_entity("") == ""
  assert normalize_entity(None) == ""


def test_is_refusal():
  assert is_refusal(REFUSAL_TEXT)
  assert is_refusal("I don’t know based on the context")
  assert not is_refusal("Elena Petrova")
  assert not is_refusal("")


def test_chunk_empty():
  assert chunk_text("   \n\n ") == []


def test_chunk_short_text_is_single_chunk():
  assert chunk_text("One short paragraph.", 200, 20) == ["One short paragraph."]


def test_chunk_rejects_bad_overlap():
  with pytest.raises(ValueError):
    chunk_text("abc", chunk_size=10, overlap=10)


def test_chunk_bounds_and_coverage():
  paras = [f"Paragraph {i}. " + "word " * 30 for i in range(10)]
  text = "\n\n".join(paras)
  chunks = chunk_text(text, chunk_size=200, overlap=40)
  assert len(chunks) > 1
  assert all(c.strip() for c in chunks)
  assert all(len(c) <= 200 + 40 + 2 for c in chunks)
  assert chunks[0].startswith("Paragraph 0.")
  assert chunks[-1].rstrip().endswith("word")
  joined = " ".join(chunks)
  for i in range(10):
    assert f"Paragraph {i}." in joined


def test_chunk_splits_oversized_single_sentence():
  chunks = chunk_text("x" * 1000, chunk_size=300, overlap=0)
  assert len(chunks) == 4
  assert "".join(chunks) == "x" * 1000
