"""Small JSON-backed stores that live next to ChromaDB:

* DocumentRegistry - one record per uploaded document (status, counts, hash).
* ParentStore      - parent passages, one JSON file per document, so deleting a
                     document removes its parents with a single unlink.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..schemas.document import DocumentInfo
from .chunking import ParentPassage


def _atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


class DocumentRegistry:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._docs: dict[str, DocumentInfo] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                self._docs = {k: DocumentInfo(**v) for k, v in raw.items()}
            except Exception:  # corrupt registry: start empty rather than crash
                self._docs = {}
        # a crash mid-indexing leaves "processing" records that can never finish
        for d in self._docs.values():
            if d.status == "processing":
                d.status, d.error = "failed", "Indexing was interrupted; delete and re-upload."

    def _save(self) -> None:
        _atomic_write(self.path, {k: v.model_dump() for k, v in self._docs.items()})

    def get(self, doc_id: str) -> Optional[DocumentInfo]:
        return self._docs.get(doc_id)

    def find_by_hash(self, sha: str) -> Optional[DocumentInfo]:
        return next((d for d in self._docs.values() if d.sha256 == sha), None)

    def list(self) -> list[DocumentInfo]:
        return sorted(self._docs.values(), key=lambda d: d.uploaded_at, reverse=True)

    def upsert(self, doc: DocumentInfo) -> None:
        with self._lock:
            self._docs[doc.doc_id] = doc
            self._save()

    def update(self, doc_id: str, **fields) -> Optional[DocumentInfo]:
        with self._lock:
            doc = self._docs.get(doc_id)
            if doc is None:
                return None
            for k, v in fields.items():
                setattr(doc, k, v)
            self._save()
            return doc

    def delete(self, doc_id: str) -> None:
        with self._lock:
            self._docs.pop(doc_id, None)
            self._save()

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ParentStore:
    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def _file(self, doc_id: str) -> Path:
        return self.dir / f"{doc_id}.json"

    def save(self, doc_id: str, parents: list[ParentPassage]) -> None:
        _atomic_write(self._file(doc_id), {p.parent_id: p.__dict__ for p in parents})

    def load(self, doc_id: str) -> dict[str, dict]:
        f = self._file(doc_id)
        if not f.exists():
            return {}
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def get(self, doc_id: str, parent_id: str) -> Optional[dict]:
        """Missing/deleted parents return None; callers fall back to the child."""
        return self.load(doc_id).get(parent_id)

    def delete(self, doc_id: str) -> None:
        self._file(doc_id).unlink(missing_ok=True)
