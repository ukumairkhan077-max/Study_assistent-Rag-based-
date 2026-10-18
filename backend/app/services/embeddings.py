"""Text -> vector.

Terminology: *text* is a string; *tokens* are the sub-word pieces the model's
tokenizer produces; the model maps the token sequence to an *embedding*, which
is a *vector* of floats (e.g. 384 dims). Queries and chunks MUST be embedded
with the same model or their vectors live in unrelated spaces.

All vectors are L2-normalised, so cosine similarity == dot product and
Euclidean distance is a monotone function of cosine similarity. Similarity
scores are NOT probabilities: 0.62 does not mean "62% likely to be correct".
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model_name: str
    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local CPU embeddings. The model is downloaded from Hugging Face on first
    use and cached; after that no network is needed."""

    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._dimension: int | None = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy: heavy import

            self._model = SentenceTransformer(self.model_name, device="cpu")
            self._dimension = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dimension(self) -> int:  # type: ignore[override]
        self._load()
        return int(self._dimension)  # type: ignore[arg-type]

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vecs = self._load().encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder.

    NOT semantic (it only knows shared words). Exists so tests and pipeline
    smoke-runs work offline and without a model download. Never use it to
    report retrieval quality."""

    def __init__(self, dimension: int = 256, model_name: str = "hashing-test-embedder"):
        self.dimension = dimension
        self.model_name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dimension] += 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


def build_embedder(model_name: str) -> Embedder:
    if model_name == "hashing-test-embedder":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model_name)
