"""Persistent ChromaDB storage.

Collection schema (collection name: ``study_chunks``, cosine space):
  id         -> chunk_id  "<doc_id>:p<page>:P<parent_no>:c<child_no>"  (stable => idempotent upserts)
  document   -> chunk text (the CHILD chunk that was embedded)
  embedding  -> vector from the configured embedding model (we pass vectors in; Chroma never embeds)
  metadata   -> doc_id, filename, page (1-based), chunk_index, parent_id, collection
Collection metadata records ``embedding_model`` and ``embedding_dim``; opening the
collection with a different model raises EmbeddingModelMismatch instead of
silently mixing incompatible vector spaces. To switch models, run a reindex
(see README: "Changing the embedding model").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chromadb
import numpy as np

from .chunking import Chunk

COLLECTION = "study_chunks"


class EmbeddingModelMismatch(RuntimeError):
    pass


@dataclass
class StoredChunk:
    chunk_id: str
    text: str
    metadata: dict
    distance: Optional[float] = None
    embedding: Optional[np.ndarray] = None

    @property
    def similarity(self) -> Optional[float]:
        # Chroma's cosine "distance" = 1 - cosine_similarity
        return None if self.distance is None else 1.0 - self.distance


class VectorStore:
    def __init__(self, path, embedding_model: str, embedding_dim: Optional[int] = None):
        self.client = chromadb.PersistentClient(path=str(path))
        self.embedding_model = embedding_model
        self.version = 0  # bumped on every write; the BM25 index watches it
        existing = [c.name if hasattr(c, "name") else c for c in self.client.list_collections()]
        if COLLECTION in existing:
            self.col = self.client.get_collection(COLLECTION, embedding_function=None)
            stored = (self.col.metadata or {}).get("embedding_model")
            if stored and stored != embedding_model:
                raise EmbeddingModelMismatch(
                    f"Collection was built with '{stored}' but the app is configured for "
                    f"'{embedding_model}'. Reindex or set EMBEDDING_MODEL back."
                )
        else:
            meta = {"hnsw:space": "cosine", "embedding_model": embedding_model}
            if embedding_dim:
                meta["embedding_dim"] = int(embedding_dim)
            self.col = self.client.create_collection(COLLECTION, metadata=meta, embedding_function=None)

    # ------------------------------------------------------------- writes
    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray, batch: int = 256) -> None:
        """Upsert => re-ingesting the same document overwrites, never duplicates."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        dim = (self.col.metadata or {}).get("embedding_dim")
        if dim and len(embeddings) and embeddings.shape[1] != int(dim):
            raise EmbeddingModelMismatch(f"Embedding dimension {embeddings.shape[1]} != collection {dim}")
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            self.col.upsert(
                ids=[c.chunk_id for c in part],
                documents=[c.text for c in part],
                embeddings=embeddings[i : i + batch].tolist(),
                metadatas=[c.metadata() for c in part],
            )
        self.version += 1

    def delete_document(self, doc_id: str) -> None:
        self.col.delete(where={"doc_id": doc_id})
        self.version += 1

    # -------------------------------------------------------------- reads
    @staticmethod
    def build_where(doc_ids: Optional[list[str]] = None, collection: Optional[str] = None) -> Optional[dict]:
        conds = []
        if doc_ids:
            conds.append({"doc_id": {"$in": list(doc_ids)}})
        if collection:
            conds.append({"collection": collection})
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def count(self, where: Optional[dict] = None) -> int:
        if where is None:
            return self.col.count()
        return len(self.col.get(where=where, include=[])["ids"])

    def query(
        self, embedding: np.ndarray, k: int, where: Optional[dict] = None, with_embeddings: bool = False
    ) -> list[StoredChunk]:
        n = self.count(where)
        if n == 0 or k <= 0:
            return []
        include = ["documents", "metadatas", "distances"] + (["embeddings"] if with_embeddings else [])
        res = self.col.query(
            query_embeddings=[np.asarray(embedding, dtype=float).tolist()],
            n_results=min(k, n),
            where=where,
            include=include,
        )
        out = []
        for i, cid in enumerate(res["ids"][0]):
            emb = np.asarray(res["embeddings"][0][i]) if with_embeddings else None
            out.append(StoredChunk(cid, res["documents"][0][i], res["metadatas"][0][i], res["distances"][0][i], emb))
        return out

    def get_by_ids(self, ids: list[str]) -> list[StoredChunk]:
        if not ids:
            return []
        res = self.col.get(ids=ids, include=["documents", "metadatas"])
        return [StoredChunk(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def list_chunks(self, where: Optional[dict] = None) -> list[StoredChunk]:
        res = self.col.get(where=where, include=["documents", "metadatas"])
        items = [StoredChunk(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
        return sorted(items, key=lambda c: (c.metadata.get("doc_id", ""), c.metadata.get("chunk_index", 0)))

    def info(self) -> dict:
        return {
            "collection": COLLECTION,
            "chunks": self.col.count(),
            "metadata": dict(self.col.metadata or {}),
        }
