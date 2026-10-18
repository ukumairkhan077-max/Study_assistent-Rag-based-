"""Shared fixtures. No network, no model downloads, no paid API: embeddings use a
deterministic hashing embedder and the LLM is a scripted fake."""
import sys
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.deps import Services, build_services, get_services  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.embeddings import HashingEmbedder  # noqa: E402
from app.services.llm_service import LLMUnavailableError  # noqa: E402


class FakeLLM:
    """Scripted LLM. `responses` are returned in order (last one repeats)."""

    def __init__(self, responses=None, configured=True, error=None):
        self.responses, self.configured, self.error = list(responses or ["Fake answer [S1]"]), configured, error
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    def complete(self, system, user):
        return self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])


class FakeReranker:
    """Scores by how many query words a passage contains (deterministic)."""

    def score(self, query, texts):
        q = set(query.lower().split())
        return [float(len(q & set(t.lower().split()))) for t in texts]


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


REGRESSION = ("Multiple regression models a response variable as a linear combination of several predictors. "
              "Each coefficient measures the change in the response for a one unit change in that predictor while "
              "the other predictors are held constant. The adjusted R-squared penalises useless predictors.")
CLUSTER = ("K-means clustering partitions observations into k groups by minimising within-cluster variance. "
           "The algorithm alternates between assigning points to the nearest centroid and recomputing centroids.")
FORMULA = "The F-statistic tests whether at least one coefficient is non-zero. The identifier VIF_2024 flags multicollinearity."


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(chroma_path=tmp_path / "chroma", data_dir=tmp_path / "data", embedding_model="hashing-test-embedder",
                    llm_api_key="", llm_model="", chunk_size=200, chunk_overlap=30, parent_chunk_size=600,
                    min_relevance=0.1, _env_file=None)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def services(settings, fake_llm) -> Services:
    return build_services(settings, embedder=HashingEmbedder(), llm=fake_llm, reranker=FakeReranker())


@pytest.fixture
def client(services):
    app.dependency_overrides[get_services] = lambda: services
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def indexed(services):
    """Index a 3-page PDF through the real ingestion path; returns the DocumentInfo."""
    data = make_pdf([REGRESSION, CLUSTER, FORMULA])
    doc, dup, path = services.ingestion.validate_and_register("lecture.pdf", data)
    assert not dup and path is not None
    services.ingestion.index(doc.doc_id)
    return services.registry.get(doc.doc_id)
