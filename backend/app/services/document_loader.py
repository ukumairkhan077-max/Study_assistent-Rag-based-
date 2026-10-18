"""PDF text extraction with page-level metadata.

PyMuPDF extracts the *text layer* of a PDF. A scanned PDF (page images only) has
no text layer, so extraction yields nothing. We detect that case and report it
instead of pretending to index the document. To support scans you would add OCR
(e.g. `ocrmypdf` or `pytesseract` + Tesseract installed on Windows) as a
pre-processing step that produces a searchable PDF; that is NOT bundled here.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf  # PyMuPDF


class DocumentError(Exception):
    """Base class for user-facing document problems."""


class UnreadableDocumentError(DocumentError):
    pass


class EmptyDocumentError(DocumentError):
    pass


@dataclass
class PageText:
    page: int  # 1-based, matches what a student sees in a PDF viewer
    text: str


def sanitize_filename(name: str) -> str:
    """Drop any directory part and unsafe characters (path traversal defence)."""
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    return name[:150] or "document.pdf"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_pdf_bytes(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(path: Path) -> list[PageText]:
    """Return non-empty pages. Raises a DocumentError with a helpful message."""
    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # corrupt / not a PDF
        raise UnreadableDocumentError(f"Could not open PDF: {exc}") from exc
    try:
        if doc.needs_pass:
            raise UnreadableDocumentError("PDF is password protected.")
        pages: list[PageText] = []
        total = doc.page_count
        for i in range(total):
            text = clean_text(doc.load_page(i).get_text("text"))
            if text:
                pages.append(PageText(page=i + 1, text=text))
    finally:
        doc.close()
    if total == 0:
        raise EmptyDocumentError("The PDF has no pages.")
    if not pages:
        raise EmptyDocumentError(
            "No extractable text found. This looks like a scanned PDF; run OCR first "
            "(see README: 'Scanned PDFs')."
        )
    return pages
