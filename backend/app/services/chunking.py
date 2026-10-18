"""Chunking. Pure functions - testable without an LLM or a vector DB.

Why chunk? Embedding models and LLM context windows are finite, and one vector
for a whole chapter blurs many topics together. Small chunks give precise
matches but may lose surrounding context; large chunks keep context but dilute
the embedding and waste LLM tokens. We therefore index SMALL child chunks for
precision and keep a LARGER parent passage for context (parent-document
retrieval). Chunks never cross page boundaries so every chunk has one exact
page number (trade-off: a sentence split across pages is split in two chunks).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .document_loader import PageText

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    index: int
    text: str
    parent_id: str = ""
    collection: str = "General"

    def metadata(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page": self.page,
            "chunk_index": self.index,
            "parent_id": self.parent_id,
            "collection": self.collection,
        }


@dataclass
class ParentPassage:
    parent_id: str
    doc_id: str
    filename: str
    page: int
    text: str


# ----------------------------------------------------------------- splitters
def fixed_split(text: str, size: int, overlap: int) -> list[str]:
    """Baseline: sliding window of `size` characters, stepping size-overlap."""
    _check(size, overlap)
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    step = size - overlap
    out = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            out.append(piece)
        if start + size >= len(text):
            break
    return out


def recursive_split(text: str, size: int, overlap: int, separators: Sequence[str] = SEPARATORS) -> list[str]:
    """Split on the largest natural boundary (paragraph > line > sentence > word)
    that keeps pieces <= size, then merge small pieces back up to `size` with
    `overlap` characters carried between consecutive chunks."""
    _check(size, overlap)
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    sep_idx, parts = len(separators) - 1, None
    for i, sep in enumerate(separators):
        if sep == "":
            sep_idx = i
            break
        if sep in text:
            sep_idx, parts = i, text.split(sep)
            break
    sep = separators[sep_idx]
    if parts is None or sep == "":  # no natural boundary left: hard split
        return fixed_split(text, size, overlap)
    pieces: list[str] = []
    for p in parts:
        if len(p) > size:
            pieces.extend(recursive_split(p, size, overlap, separators[sep_idx + 1 :]))
        elif p.strip():
            pieces.append(p)
    return _merge(pieces, sep, size, overlap)


def _merge(pieces: list[str], sep: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    total = 0
    for p in pieces:
        add = len(p) + (len(sep) if cur else 0)
        if cur and total + add > size:
            chunks.append(sep.join(cur).strip())
            # keep a tail of at most `overlap` chars as the start of the next chunk
            while cur and (total > overlap or total + len(p) + len(sep) > size):
                total -= len(cur[0]) + (len(sep) if len(cur) > 1 else 0)
                cur.pop(0)
            total = max(total, 0)
            add = len(p) + (len(sep) if cur else 0)
        cur.append(p)
        total += add
    if cur:
        chunks.append(sep.join(cur).strip())
    return [c for c in chunks if c]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p and p.strip()]


def semantic_split(
    text: str,
    embed: Callable[[list[str]], np.ndarray],
    max_chars: int,
    percentile: float = 25.0,
) -> list[str]:
    """Start a new chunk where adjacent-sentence similarity drops into the lowest
    `percentile` of the document (a topic shift), or when `max_chars` is reached.
    Needs an embedder, so it is slower than recursive splitting."""
    sents = split_sentences(text)
    if len(sents) < 3:
        return recursive_split(text, max_chars, 0)
    vecs = np.asarray(embed(sents), dtype=float)
    sims = np.array([float(vecs[i] @ vecs[i + 1]) for i in range(len(sents) - 1)])
    cut = np.percentile(sims, percentile)
    chunks, cur, length = [], [sents[0]], len(sents[0])
    for i in range(1, len(sents)):
        if (sims[i - 1] <= cut and length > max_chars * 0.3) or length + len(sents[i]) > max_chars:
            chunks.append(" ".join(cur))
            cur, length = [], 0
        cur.append(sents[i])
        length += len(sents[i]) + 1
    chunks.append(" ".join(cur))
    out: list[str] = []
    for c in chunks:  # a single very long sentence can still exceed max_chars
        out.extend(recursive_split(c, max_chars, 0) if len(c) > max_chars else [c])
    return out


def _check(size: int, overlap: int) -> None:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")


# ------------------------------------------------------- document -> chunks
def chunk_pages(
    pages: list[PageText],
    doc_id: str,
    filename: str,
    *,
    strategy: str = "recursive",
    chunk_size: int = 500,
    overlap: int = 80,
    parent_size: int = 1800,
    collection: str = "General",
    embed: Optional[Callable[[list[str]], np.ndarray]] = None,
) -> tuple[list[Chunk], list[ParentPassage]]:
    """Build parent passages (large, recursive, no overlap) and split each into
    child chunks using `strategy`. Only children are embedded for retrieval."""
    if strategy == "semantic" and embed is None:
        raise ValueError("semantic chunking requires an embedding function")
    if strategy not in ("recursive", "fixed", "semantic"):
        raise ValueError(f"Unknown chunk strategy: {strategy}")
    children: list[Chunk] = []
    parents: list[ParentPassage] = []
    for pg in pages:
        parent_texts = recursive_split(pg.text, max(parent_size, chunk_size), 0)
        for pi, ptext in enumerate(parent_texts):
            parent_id = f"{doc_id}:p{pg.page}:P{pi}"
            parents.append(ParentPassage(parent_id, doc_id, filename, pg.page, ptext))
            if strategy == "fixed":
                texts = fixed_split(ptext, chunk_size, overlap)
            elif strategy == "semantic":
                texts = semantic_split(ptext, embed, chunk_size)  # type: ignore[arg-type]
            else:
                texts = recursive_split(ptext, chunk_size, overlap)
            for ci, t in enumerate(texts):
                children.append(
                    Chunk(
                        chunk_id=f"{parent_id}:c{ci}",
                        doc_id=doc_id,
                        filename=filename,
                        page=pg.page,
                        index=len(children),
                        text=t,
                        parent_id=parent_id,
                        collection=collection,
                    )
                )
    return children, parents
