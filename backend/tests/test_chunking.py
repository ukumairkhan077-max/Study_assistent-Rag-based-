import pytest

from app.services.chunking import chunk_pages, fixed_split, recursive_split, semantic_split
from app.services.document_loader import PageText, sanitize_filename
from app.services.embeddings import HashingEmbedder

TEXT = " ".join(f"Sentence number {i} talks about regression and variance." for i in range(40))


def test_recursive_chunks_respect_size_limit():
    chunks = recursive_split(TEXT, 200, 40)
    assert len(chunks) > 3
    assert all(len(c) <= 200 for c in chunks)


def test_recursive_overlap_is_shared_between_neighbours():
    chunks = recursive_split(TEXT, 200, 60)
    shared = 0
    for a, b in zip(chunks, chunks[1:]):
        # the tail of a chunk reappears at the start of the next one
        if any(a.endswith(b[:n]) for n in range(15, 61) if n <= len(b)):
            shared += 1
    assert shared >= len(chunks) - 2


def test_recursive_prefers_paragraph_boundaries():
    text = ("A" * 80 + "\n\n" + "B" * 80 + "\n\n" + "C" * 80)
    chunks = recursive_split(text, 100, 0)
    assert chunks == ["A" * 80, "B" * 80, "C" * 80]


def test_fixed_split_overlap_and_coverage():
    text = "".join(str(i % 10) for i in range(1000))
    chunks = fixed_split(text, 100, 20)
    assert chunks[0] == text[:100] and chunks[1][:20] == chunks[0][-20:]
    assert chunks[-1].endswith(text[-5:])


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        recursive_split(TEXT, 100, 100)


def test_empty_text_gives_no_chunks():
    assert recursive_split("   ", 100, 10) == []


def test_page_metadata_ids_and_parent_links():
    pages = [PageText(1, TEXT), PageText(3, "Short page about clustering.")]
    children, parents = chunk_pages(pages, "doc1", "notes.pdf", chunk_size=200, overlap=30, parent_size=600)
    assert {c.page for c in children} == {1, 3}
    assert all(c.filename == "notes.pdf" and c.doc_id == "doc1" for c in children)
    assert len({c.chunk_id for c in children}) == len(children)  # unique, stable IDs
    parent_ids = {p.parent_id for p in parents}
    assert all(c.parent_id in parent_ids for c in children)
    # a child's text really is contained in its parent, on the same page
    by_id = {p.parent_id: p for p in parents}
    for c in children:
        assert c.text[:50] in by_id[c.parent_id].text and by_id[c.parent_id].page == c.page


def test_chunk_ids_are_deterministic():
    pages = [PageText(1, TEXT)]
    a, _ = chunk_pages(pages, "d", "f.pdf", chunk_size=200, overlap=30)
    b, _ = chunk_pages(pages, "d", "f.pdf", chunk_size=200, overlap=30)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_semantic_split_makes_multiple_chunks_on_topic_shift():
    emb = HashingEmbedder()
    text = ("Regression models predictors. Regression coefficients matter. Regression residuals are checked. "
            "Pizza dough needs flour. Pizza ovens are hot. Pizza toppings vary widely.")
    chunks = semantic_split(text, emb.embed, max_chars=400, percentile=30)
    assert len(chunks) >= 2
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_semantic_strategy_requires_embedder():
    with pytest.raises(ValueError):
        chunk_pages([PageText(1, TEXT)], "d", "f.pdf", strategy="semantic")


def test_sanitize_filename_blocks_path_traversal():
    assert sanitize_filename("../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_filename("C:\\Users\\me\\notes<>.pdf") == "notes__.pdf"
