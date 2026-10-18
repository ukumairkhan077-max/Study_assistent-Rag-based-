"""Upload -> extract -> chunk -> embed -> store. Idempotent: the doc_id is derived
from the file's SHA-256, so uploading the same bytes twice never indexes twice."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..core.config import Settings
from ..schemas.document import DocumentInfo
from .chunking import chunk_pages
from .document_loader import DocumentError, extract_pages, is_pdf_bytes, sanitize_filename, sha256_bytes
from .embeddings import Embedder
from .registry import DocumentRegistry, ParentStore
from .vector_store import VectorStore

log = logging.getLogger("ingest")


class InvalidUpload(Exception):
    pass


class IngestionService:
    def __init__(self, s: Settings, registry: DocumentRegistry, store: VectorStore, parents: ParentStore, embedder: Embedder):
        self.s, self.registry, self.store, self.parents, self.embedder = s, registry, store, parents, embedder
        s.uploads_dir.mkdir(parents=True, exist_ok=True)

    def validate_and_register(self, filename: str, data: bytes, collection: str = "General",
                              strategy: Optional[str] = None) -> tuple[DocumentInfo, bool, Optional[Path]]:
        """Fast, synchronous part: validate, de-duplicate, save, register as 'processing'.
        Returns (record, is_duplicate, path_to_index_or_None)."""
        name = sanitize_filename(filename)
        if not name.lower().endswith(".pdf"):
            raise InvalidUpload("Only .pdf files are supported.")
        if not data:
            raise InvalidUpload("The file is empty.")
        if len(data) > self.s.max_upload_mb * 1024 * 1024:
            raise InvalidUpload(f"File exceeds the {self.s.max_upload_mb} MB limit.")
        if not is_pdf_bytes(data):
            raise InvalidUpload("The file content is not a PDF.")
        sha = sha256_bytes(data)
        existing = self.registry.find_by_hash(sha)
        if existing and existing.status != "failed":
            return existing, True, None
        doc_id = sha[:16]
        path = self.s.uploads_dir / f"{doc_id}.pdf"  # server-chosen name: no user-controlled path
        path.write_bytes(data)
        doc = DocumentInfo(
            doc_id=doc_id, filename=name, collection=(collection or "General")[:60], sha256=sha, status="processing",
            chunk_strategy=strategy or self.s.chunk_strategy, embedding_model=self.embedder.model_name,
            uploaded_at=DocumentRegistry.now(),
        )
        self.registry.upsert(doc)
        return doc, False, path

    def index(self, doc_id: str) -> None:
        """Slow part (runs in a background task). Failures are recorded, not raised."""
        doc = self.registry.get(doc_id)
        if doc is None:
            return
        path = self.s.uploads_dir / f"{doc_id}.pdf"
        try:
            pages = extract_pages(path)
            children, parents = chunk_pages(
                pages, doc_id, doc.filename, strategy=doc.chunk_strategy, chunk_size=self.s.chunk_size,
                overlap=self.s.chunk_overlap, parent_size=self.s.parent_chunk_size, collection=doc.collection,
                embed=self.embedder.embed,
            )
            if not children:
                raise DocumentError("No chunks could be created from this document.")
            vectors = self.embedder.embed([c.text for c in children])
            self.store.delete_document(doc_id)  # clean slate => retry-safe
            self.store.add_chunks(children, vectors)
            self.parents.save(doc_id, parents)
            self.registry.update(doc_id, status="indexed", error=None, pages=len(pages),
                                 chunk_count=len(children), parent_count=len(parents))
        except DocumentError as exc:
            self._fail(doc_id, str(exc), path)
        except Exception as exc:  # embedding download failure, disk error, ...
            log.exception("Indexing failed for %s", doc_id)
            self._fail(doc_id, f"Indexing failed: {type(exc).__name__}: {exc}", path)

    def _fail(self, doc_id: str, message: str, path: Path) -> None:
        self.store.delete_document(doc_id)
        self.parents.delete(doc_id)
        self.registry.update(doc_id, status="failed", error=message)

    def delete(self, doc_id: str) -> bool:
        """Remove vectors, parents, upload and registry entry => no stale content."""
        if self.registry.get(doc_id) is None:
            return False
        self.store.delete_document(doc_id)
        self.parents.delete(doc_id)
        (self.s.uploads_dir / f"{doc_id}.pdf").unlink(missing_ok=True)
        self.registry.delete(doc_id)
        return True
