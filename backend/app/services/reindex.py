"""Rebuild the vector index after changing EMBEDDING_MODEL (or chunk settings).

Vectors from different embedding models live in different spaces and must never
share a collection, so reindexing drops the collection and re-embeds every stored
upload. Original PDFs in data/uploads are the source of truth."""
from __future__ import annotations

from typing import Optional

import chromadb

from ..api.deps import build_services
from ..core.config import Settings
from .embeddings import Embedder
from .registry import DocumentRegistry
from .vector_store import COLLECTION


def reindex_all(settings: Settings, embedder: Optional[Embedder] = None) -> dict:
    docs = DocumentRegistry(settings.registry_path).list()
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # collection did not exist yet
        pass
    sv = build_services(settings, embedder=embedder)
    report = {"embedding_model": sv.embedder.model_name, "indexed": 0, "failed": []}
    for d in docs:
        if not (settings.uploads_dir / f"{d.doc_id}.pdf").exists():
            sv.registry.update(d.doc_id, status="failed", error="Original PDF missing; re-upload it.")
            report["failed"].append(d.filename)
            continue
        sv.registry.update(d.doc_id, status="processing", error=None, embedding_model=sv.embedder.model_name)
        sv.ingestion.index(d.doc_id)
        ok = sv.registry.get(d.doc_id).status == "indexed"
        report["indexed" if ok else "failed"] += 1 if ok else 0
        if not ok:
            report["failed"].append(d.filename)
    return report
