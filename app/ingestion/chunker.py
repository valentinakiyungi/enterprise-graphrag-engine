"""Structure-aware text chunking.

Splits on headings and paragraphs first, then sentences, and finally hard
character windows, packing pieces into chunks up to ``chunk_size`` with a
sentence-aligned ``overlap`` carried into the next chunk.
"""

from __future__ import annotations

import re

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_HEADING_RE = re.compile(r"^#{1,6}\s+.+$", re.M)


def _split_units(text: str, chunk_size: int) -> list[tuple[str, str]]:
    """Break text into (unit, separator) pairs no larger than chunk_size.

    The separator is what joins the unit to the previous one: a blank line
    between paragraphs, a space between sentences of the same paragraph.
    """
    units: list[tuple[str, str]] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            units.append((para, "\n\n"))
            continue
        sep = "\n\n"
        for sent in _SENTENCE_RE.split(para):
            sent = sent.strip()
            if not sent:
                continue
            while len(sent) > chunk_size:  # pathological long sentence
                cut = sent.rfind(" ", 0, chunk_size)
                cut = cut if cut > chunk_size // 2 else chunk_size
                units.append((sent[:cut].strip(), sep))
                sep = " "
                sent = sent[cut:].strip()
            if sent:
                units.append((sent, sep))
                sep = " "
    return units


def _tail_overlap(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    # start the overlap on a sentence or word boundary
    m = _SENTENCE_RE.search(tail)
    if m:
        return tail[m.end():]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size // 2))
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    chunks: list[str] = []
    current = ""
    current_heading = ""
    for unit, sep in _split_units(text, chunk_size):
        if _HEADING_RE.match(unit):
            current_heading = unit
        candidate = f"{current}{sep}{unit}" if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            carry = _tail_overlap(current, overlap)
            # keep the active section heading as context for the next chunk
            prefix = (
                current_heading
                if current_heading and not carry.startswith(current_heading) and unit != current_heading
                else ""
            )
            seed = "\n\n".join(p for p in (prefix, carry) if p)
            current = f"{seed}{sep}{unit}" if seed else unit
            if len(current) > chunk_size:  # overlap would overflow; drop it
                current = unit
        else:
            current = unit
    if current:
        chunks.append(current)
    return chunks
