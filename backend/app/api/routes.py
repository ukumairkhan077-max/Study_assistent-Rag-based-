"""Thin HTTP layer: validate input, call a service, map errors to status codes."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from ..schemas.chat import (ChatRequest, ChatResponse, ExplainAnswerRequest, RetrieveRequest, RetrieveResponse,
                            StudyRequest, StudyResponse)
from ..schemas.document import ChunkInfo, DocumentInfo, UploadResponse
from ..services.evaluation_service import EvaluationService
from ..services.ingestion import InvalidUpload
from ..services.llm_service import LLMError, LLMNotConfiguredError, LLMRequestError, LLMUnavailableError
from .deps import Services, get_services

router = APIRouter(prefix="/api")
_evaluation = EvaluationService()


def get_evaluation() -> EvaluationService:
    return _evaluation


def llm_guard(fn):
    """Run an LLM-dependent call and translate provider failures to HTTP errors."""
    try:
        return fn()
    except LLMNotConfiguredError as exc:
        raise HTTPException(503, str(exc))
    except LLMUnavailableError as exc:
        raise HTTPException(503, str(exc))
    except LLMRequestError as exc:
        raise HTTPException(502, str(exc))
    except LLMError as exc:
        raise HTTPException(502, str(exc))


# ------------------------------------------------------------------ health
@router.get("/health")
def health(sv: Services = Depends(get_services)) -> dict:
    s = sv.settings
    return {  # safe subset only: no keys, no base URL, no filesystem paths
        "status": "ok",
        "llm_configured": s.llm_configured,
        "llm_model": s.llm_model or None,
        "embedding_model": sv.embedder.model_name,
        "reranker_model": s.reranker_model,
        "chunk_strategy": s.chunk_strategy,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "parent_chunk_size": s.parent_chunk_size,
        "indexed_chunks": sv.store.count(),
        "documents": len(sv.registry.list()),
    }


# --------------------------------------------------------------- documents
@router.post("/documents", response_model=UploadResponse, status_code=202)
def upload_document(background: BackgroundTasks, file: UploadFile = File(...), collection: str = Form("General"),
                    chunk_strategy: Optional[str] = Form(None), sv: Services = Depends(get_services)):
    if chunk_strategy not in (None, "", "recursive", "fixed", "semantic"):
        raise HTTPException(422, "chunk_strategy must be recursive, fixed or semantic")
    limit = sv.settings.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)  # never read more than limit+1 bytes
    try:
        doc, duplicate, path = sv.ingestion.validate_and_register(file.filename or "document.pdf", data, collection,
                                                                  chunk_strategy or None)
    except InvalidUpload as exc:
        code = 413 if "limit" in str(exc) else 400
        raise HTTPException(code, str(exc))
    if path is not None:
        background.add_task(sv.ingestion.index, doc.doc_id)
    return UploadResponse(document=doc, duplicate=duplicate)


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents(sv: Services = Depends(get_services)):
    return sv.registry.list()


@router.get("/documents/{doc_id}", response_model=DocumentInfo)
def get_document(doc_id: str, sv: Services = Depends(get_services)):
    doc = sv.registry.get(doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    return doc


@router.get("/documents/{doc_id}/chunks", response_model=list[ChunkInfo])
def get_chunks(doc_id: str, sv: Services = Depends(get_services)):
    if sv.registry.get(doc_id) is None:
        raise HTTPException(404, "Document not found")
    chunks = sv.store.list_chunks(sv.store.build_where([doc_id]))
    return [ChunkInfo(chunk_id=c.chunk_id, doc_id=doc_id, page=int(c.metadata["page"]),
                      parent_id=c.metadata.get("parent_id", ""), text=c.text) for c in chunks]


@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, sv: Services = Depends(get_services)):
    if not sv.ingestion.delete(doc_id):
        raise HTTPException(404, "Document not found")


# -------------------------------------------------------------------- chat
@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, sv: Services = Depends(get_services)):
    return llm_guard(lambda: sv.pipeline.answer(req))


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest, sv: Services = Depends(get_services)):
    """Works without any LLM: inspect exactly what retrieval returns."""
    return sv.pipeline.inspect(req.query, req.retrieval)


# ------------------------------------------------------------------- study
@router.post("/study/summarize", response_model=StudyResponse)
def summarize(req: StudyRequest, sv: Services = Depends(get_services)):
    return llm_guard(lambda: sv.study.summarize(req))


@router.post("/study/explain", response_model=StudyResponse)
def explain(req: StudyRequest, sv: Services = Depends(get_services)):
    return llm_guard(lambda: sv.study.explain(req))


@router.post("/study/quiz", response_model=StudyResponse)
def quiz(req: StudyRequest, sv: Services = Depends(get_services)):
    return llm_guard(lambda: sv.study.generate("mcq", req))


@router.post("/study/short-answer", response_model=StudyResponse)
def short_answer(req: StudyRequest, sv: Services = Depends(get_services)):
    return llm_guard(lambda: sv.study.generate("short_answer", req))


@router.post("/study/flashcards", response_model=StudyResponse)
def flashcards(req: StudyRequest, sv: Services = Depends(get_services)):
    return llm_guard(lambda: sv.study.generate("flashcards", req))


@router.post("/study/explain-answer", response_model=StudyResponse)
def explain_answer(req: ExplainAnswerRequest, sv: Services = Depends(get_services)):
    return llm_guard(lambda: sv.study.explain_answer(req))


# -------------------------------------------------------------- evaluation
@router.get("/evaluation/results")
def evaluation_results(ev: EvaluationService = Depends(get_evaluation)):
    results = ev.latest_results()
    if results is None:
        raise HTTPException(404, "No evaluation results yet. Run the evaluation first.")
    return results


@router.post("/evaluation/run", status_code=202)
def evaluation_run(generate: bool = False, ev: EvaluationService = Depends(get_evaluation)):
    if not ev.start(with_generation=generate):
        raise HTTPException(409, "An evaluation is already running.")
    return {"state": "running"}


@router.get("/evaluation/status")
def evaluation_status(ev: EvaluationService = Depends(get_evaluation)):
    return ev.status()
