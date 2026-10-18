import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .core.config import get_settings

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# httpx/openai log full URLs at INFO; keep them quiet so nothing sensitive is printed
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

app = FastAPI(
    title="AI Study Assistant",
    version="1.0.0",
    description="Chat with your documents using a RAG pipeline. Interactive docs: /docs",
)

# CORS: only the configured frontend origin (default: Vite dev server), not "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, settings.frontend_origin.replace("localhost", "127.0.0.1")],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)
app.include_router(router)
