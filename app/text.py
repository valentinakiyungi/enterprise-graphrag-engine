"""Pure text helpers (no third-party dependencies)."""

import re
from typing import List

REFUSAL_TEXT = "I don't know based on the provided context."


def normalize_entity(name: str) -> str:
  """Canonical key for an entity name: lowercase, collapsed whitespace."""
  return re.sub(r"\s+", " ", name or "").strip().lower()


def is_refusal(answer: str) -> bool:
  """True when the model declined to answer from the given context."""
  a = (answer or "").strip().lower().replace("’", "'")
  return a.startswith("i don't know") or a.startswith("i do not know")


def _split_long(paragraph: str, size: int) -> List[str]:
  sentences = re.split(r"(?<=[.!?])\s+", paragraph)
  pieces: List[str] = []
  cur = ""
  for s in sentences:
    while len(s) > size:  # a single oversized sentence: hard-slice it
      if cur:
        pieces.append(cur)
        cur = ""
      pieces.append(s[:size])
      s = s[size:]
    if not cur:
      cur = s
    elif len(cur) + 1 + len(s) <= size:
      cur += " " + s
    else:
      pieces.append(cur)
      cur = s
  if cur:
    pieces.append(cur)
  return pieces


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
  """Paragraph-aware chunking with a character overlap between chunks.

  Chunks are at most roughly chunk_size + overlap characters long.
  """
  if overlap >= chunk_size:
    raise ValueError("overlap must be smaller than chunk_size")
  text = (text or "").strip()
  if not text:
    return []

  pieces: List[str] = []
  for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
    if not para:
      continue
    if len(para) <= chunk_size:
      pieces.append(para)
    else:
      pieces.extend(_split_long(para, chunk_size))

  chunks: List[str] = []
  cur = ""
  for piece in pieces:
    if not cur:
      cur = piece
    elif len(cur) + 2 + len(piece) <= chunk_size:
      cur += "\n\n" + piece
    else:
      chunks.append(cur)
      tail = cur[-overlap:] if overlap else ""
      if tail and " " in tail:
        tail = tail[tail.index(" ") + 1:]  # start the overlap on a word boundary
      cur = (tail + "\n\n" + piece) if tail else piece
  if cur:
    chunks.append(cur)
  return chunks
