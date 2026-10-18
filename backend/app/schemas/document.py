from typing import Literal, Optional

from pydantic import BaseModel

DocStatus = Literal["processing", "indexed", "failed"]


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    collection: str = "General"
    sha256: str
    status: DocStatus
    error: Optional[str] = None
    pages: int = 0
    chunk_count: int = 0
    parent_count: int = 0
    chunk_strategy: str = "recursive"
    embedding_model: str = ""
    uploaded_at: str


class UploadResponse(BaseModel):
    document: DocumentInfo
    duplicate: bool = False


class ChunkInfo(BaseModel):
    chunk_id: str
    doc_id: str
    page: int
    parent_id: str
    text: str
