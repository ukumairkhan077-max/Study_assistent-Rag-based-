"""Centralised configuration. Every tunable value lives here and is read from
environment variables / the backend/.env file. Secrets never leave the backend."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    # --- LLM (any OpenAI-compatible endpoint) ---
    llm_provider: str = "openai-compatible"
    llm_api_key: str = ""
    llm_base_url: str = ""  # empty -> SDK default
    llm_model: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800
    llm_max_tokens_param: str = "max_tokens"  # some providers want max_completion_tokens
    llm_send_temperature: bool = True  # some reasoning models reject temperature
    llm_timeout_seconds: float = 45.0
    llm_max_attempts: int = 3

    # --- Embeddings / reranker (local, CPU friendly) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Storage ---
    chroma_path: Path = BACKEND_DIR / "chroma_db"
    data_dir: Path = BACKEND_DIR / "data"

    # --- Chunking ---
    chunk_strategy: str = "recursive"  # recursive | fixed | semantic
    chunk_size: int = 500  # characters (child chunks, used for retrieval)
    chunk_overlap: int = 80
    parent_chunk_size: int = 1800  # characters (parent passages)

    # --- Retrieval / generation ---
    min_relevance: float = 0.25  # min dense cosine similarity of best chunk to attempt an answer
    context_token_budget: int = 3000  # approx tokens (chars/4) of retrieved context sent to the LLM
    history_turns: int = 4

    # --- API ---
    frontend_origin: str = "http://localhost:5173"
    max_upload_mb: int = 25
    log_level: str = "INFO"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def parents_dir(self) -> Path:
        return self.data_dir / "parents"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "registry.json"

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()
