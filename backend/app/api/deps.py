"""Builds the object graph once. Tests build their own graph with fakes."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from ..core.config import Settings, get_settings
from ..services.embeddings import Embedder, build_embedder
from ..services.ingestion import IngestionService
from ..services.llm_service import LLMService
from ..services.rag_pipeline import RAGPipeline
from ..services.registry import DocumentRegistry, ParentStore
from ..services.retriever import CrossEncoderReranker, Reranker, Retriever
from ..services.study_tools import StudyService
from ..services.vector_store import VectorStore


@dataclass
class Services:
    settings: Settings
    embedder: Embedder
    store: VectorStore
    registry: DocumentRegistry
    parents: ParentStore
    retriever: Retriever
    llm: LLMService
    pipeline: RAGPipeline
    ingestion: IngestionService
    study: StudyService


def build_services(settings: Settings, embedder: Optional[Embedder] = None, llm=None,
                   reranker: Optional[Reranker] = None) -> Services:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    embedder = embedder or build_embedder(settings.embedding_model)
    llm = llm or LLMService(settings)
    store = VectorStore(settings.chroma_path, embedder.model_name, embedder.dimension)  # loads the model once at startup
    registry = DocumentRegistry(settings.registry_path)
    parents = ParentStore(settings.parents_dir)
    reranker = reranker or CrossEncoderReranker(settings.reranker_model)  # model loads on first use only
    retriever = Retriever(store, embedder, parents, reranker, query_llm=llm if llm.configured else None)
    pipeline = RAGPipeline(settings, retriever, llm)
    return Services(settings, embedder, store, registry, parents, retriever, llm, pipeline,
                    IngestionService(settings, registry, store, parents, embedder), StudyService(settings, pipeline, llm))


@lru_cache
def get_services() -> Services:
    return build_services(get_settings())
