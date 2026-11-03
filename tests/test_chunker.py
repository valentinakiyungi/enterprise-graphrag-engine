import re

from app.ingestion.chunker import chunk_text


def test_short_text_single_chunk():
    assert chunk_text("Hello world.", 100, 10) == ["Hello world."]


def test_empty_text():
    assert chunk_text("   \n  ", 100, 10) == []


def test_chunks_respect_size_and_cover_text():
    paras = [f"Paragraph {i}. " + "Sentence about systems and owners. " * 6 for i in range(12)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, chunk_size=500, overlap=80)
    assert len(chunks) > 3
    assert all(len(c) <= 500 for c in chunks)
    for i in range(12):
        assert any(f"Paragraph {i}." in c for c in chunks)


def test_overlap_carries_context():
    sentences = " ".join(f"Fact number {i} is important." for i in range(60))
    chunks = chunk_text(sentences, chunk_size=300, overlap=100)
    assert len(chunks) > 2
    # the tail of each chunk should reappear at the start of the next
    for a, b in zip(chunks, chunks[1:]):
        last_sentence = re.findall(r"Fact number \d+ is important\.", a)[-1]
        assert last_sentence in b


def test_long_unbroken_text_is_split():
    chunks = chunk_text("x" * 2500, chunk_size=1000, overlap=0)
    assert len(chunks) == 3 and all(len(c) <= 1000 for c in chunks)


def test_heading_kept_with_section():
    text = "# Vendors\n\n" + "\n\n".join("Vendor detail sentence goes here. " * 8 for _ in range(4))
    chunks = chunk_text(text, chunk_size=400, overlap=50)
    assert chunks[0].startswith("# Vendors")
    assert all(c.startswith("# Vendors") for c in chunks[1:])
