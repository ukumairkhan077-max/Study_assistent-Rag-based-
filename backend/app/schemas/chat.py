from typing import Literal, Optional

from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    """One model describes every retrieval variant so experiments are comparable.

    `strategy` selects the *candidate generator* (alternatives). `rerank`,
    `parent_document` and `compression` are *sequential stages* that can be
    combined with any strategy.
    """

    strategy: Literal["similarity", "mmr", "hybrid", "expansion", "multi_query"] = "similarity"
    top_k: int = Field(5, ge=1, le=20)
    candidate_k: int = Field(20, ge=1, le=100, description="Candidate pool for MMR/rerank/fusion")
    mmr_lambda: float = Field(0.5, ge=0.0, le=1.0, description="1=pure relevance, 0=pure diversity")
    num_queries: int = Field(3, ge=1, le=6, description="Generated queries for multi_query / expansion")
    hybrid_base: bool = Field(False, description="expansion/multi_query use dense+BM25 per query")
    rerank: bool = False
    parent_document: bool = False
    compression: bool = False
    compression_threshold: float = Field(0.25, ge=0.0, le=1.0)
    doc_ids: Optional[list[str]] = None
    collection: Optional[str] = None


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[HistoryTurn] = []
    retrieval: RetrievalConfig = RetrievalConfig()
    detailed: bool = False


class Source(BaseModel):
    source_id: str  # "S1" - the token the LLM cites
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    excerpt: str
    similarity: Optional[float] = None  # dense cosine similarity (NOT a probability)
    score: float  # score of the stage that produced the final order
    rerank_score: Optional[float] = None
    cited: bool = False
    parent_expanded: bool = False
    compressed: bool = False


class ChatResponse(BaseModel):
    answer: str
    supported: bool  # False when evidence was insufficient / no valid citation
    sources: list[Source]
    retrieved_chunk_ids: list[str]
    invalid_citations_removed: list[str] = []
    retrieval: RetrievalConfig
    diagnostics: dict = {}
    warnings: list[str] = []


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    retrieval: RetrievalConfig = RetrievalConfig()


class RetrieveResponse(BaseModel):
    sources: list[Source]
    top_dense_similarity: Optional[float]
    retrieval: RetrievalConfig
    diagnostics: dict = {}


class StudyRequest(BaseModel):
    topic: str = Field("", max_length=500)
    doc_id: Optional[str] = None
    count: int = Field(5, ge=1, le=15)
    retrieval: RetrievalConfig = RetrievalConfig()


class ExplainAnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    selected_answer: str = Field(min_length=1, max_length=500)
    correct_answer: str = Field(min_length=1, max_length=500)
    retrieval: RetrievalConfig = RetrievalConfig()


class StudyResponse(BaseModel):
    kind: str
    text: Optional[str] = None
    items: list[dict] = []
    sources: list[Source] = []
    warnings: list[str] = []
