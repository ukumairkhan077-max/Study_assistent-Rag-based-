import numpy as np
import pytest

from app.schemas.chat import RetrievalConfig
from app.services.chunking import Chunk
from app.services.embeddings import HashingEmbedder
from app.services.retriever import RetrievedChunk, mmr_select, reciprocal_rank_fusion
from app.services.vector_store import EmbeddingModelMismatch, VectorStore
from tests.conftest import CLUSTER, FORMULA, REGRESSION, FakeLLM, make_pdf


def rc(cid, score=0.0, sim=None):
    return RetrievedChunk(cid, "d", "f.pdf", 1, cid, score=score, similarity=sim)


# ----------------------------------------------------------------- embeddings
def test_embeddings_have_consistent_dimension_and_unit_norm():
    e = HashingEmbedder(dimension=64)
    v = e.embed(["multiple regression", "k means clustering", ""])
    assert v.shape == (3, 64)
    assert np.allclose(np.linalg.norm(v[:2], axis=1), 1.0)
    assert np.allclose(v[0], e.embed(["multiple regression"])[0])  # deterministic


# --------------------------------------------------------------- vector store
def test_insert_retrieve_and_metadata_filter(services, indexed):
    r = services.retriever
    hits = r.dense("multiple regression coefficients predictors", 3, None)
    assert hits and hits[0].page == 1 and hits[0].filename == "lecture.pdf"
    assert hits[0].similarity > hits[-1].similarity or len(hits) == 1
    where = VectorStore.build_where(doc_ids=["nonexistent"])
    assert r.dense("regression", 3, where) == []
    assert all(h.doc_id == indexed.doc_id for h in r.dense("regression", 3, VectorStore.build_where([indexed.doc_id])))


def test_reingest_same_chunks_is_idempotent(services, indexed):
    n = services.store.count()
    services.ingestion.index(indexed.doc_id)  # index twice
    assert services.store.count() == n


def test_get_by_ids_round_trip(services, indexed):
    chunk = services.store.list_chunks()[0]
    got = services.store.get_by_ids([chunk.chunk_id])
    assert got[0].text == chunk.text and got[0].metadata["page"] == chunk.metadata["page"]


def test_delete_document_removes_vectors_parents_and_sparse_hits(services, indexed):
    services.retriever.sparse("regression", 3, None, None)  # builds the BM25 index
    assert services.ingestion.delete(indexed.doc_id)
    assert services.store.count() == 0
    assert services.parents.load(indexed.doc_id) == {}
    assert services.retriever.sparse("regression", 3, None, None) == []  # no stale sparse results
    assert services.registry.get(indexed.doc_id) is None


def test_embedding_model_mismatch_is_refused(settings, services, indexed):
    with pytest.raises(EmbeddingModelMismatch):
        VectorStore(settings.chroma_path, "some-other-model")


def test_persistence_across_restart(settings, services, indexed):
    reopened = VectorStore(settings.chroma_path, "hashing-test-embedder")
    assert reopened.count() == services.store.count() > 0


# ------------------------------------------------------------------------ MMR
def test_mmr_reduces_redundancy():
    q = np.array([1.0, 0.0])
    a = np.array([1.0, 0.0])
    a_dup = np.array([0.999, 0.045]); a_dup /= np.linalg.norm(a_dup)
    other = np.array([0.7, 0.714]); other /= np.linalg.norm(other)
    vecs = np.vstack([a, a_dup, other])
    assert mmr_select(q, vecs, 2, lambda_mult=1.0) == [0, 1]  # pure relevance picks the near-duplicate
    assert mmr_select(q, vecs, 2, lambda_mult=0.3) == [0, 2]  # diversity prefers the different passage


def test_mmr_strategy_returns_top_k_unique(services, indexed):
    res = services.retriever.retrieve("regression predictors coefficient", RetrievalConfig(strategy="mmr", top_k=3, candidate_k=10))
    ids = [c.chunk_id for c in res.chunks]
    assert len(ids) == len(set(ids)) <= 3


# --------------------------------------------------------- hybrid / rank fusion
def test_rrf_rewards_agreement_and_dedupes():
    dense = [rc("a"), rc("b"), rc("c")]
    sparse = [rc("c"), rc("a"), rc("d")]
    fused = reciprocal_rank_fusion([dense, sparse])
    ids = [c.chunk_id for c in fused]
    assert len(ids) == len(set(ids)) == 4
    assert set(ids[:2]) == {"a", "c"}  # present in both lists
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_hybrid_finds_exact_identifier_and_keeps_metadata(services, indexed):
    res = services.retriever.retrieve("VIF_2024", RetrievalConfig(strategy="hybrid", top_k=3))
    top = res.chunks[0]
    assert "VIF_2024" in top.text and top.page == 3 and top.doc_id == indexed.doc_id


def test_sparse_respects_doc_filter(services, indexed):
    assert services.retriever.sparse("regression", 5, ["other-doc"], None) == []


# ------------------------------------------------------------------ rerank
def test_rerank_keeps_original_and_rerank_scores_separately(services, indexed):
    cfg = RetrievalConfig(strategy="similarity", top_k=2, candidate_k=8, rerank=True)
    res = services.retriever.retrieve("F-statistic coefficient non-zero", cfg)
    assert len(res.chunks) == 2
    for c in res.chunks:
        assert c.rerank_score is not None and c.original_score is not None
    assert res.chunks[0].rerank_score >= res.chunks[1].rerank_score


def test_rerank_disabled_leaves_scores_empty(services, indexed):
    res = services.retriever.retrieve("regression", RetrievalConfig(top_k=2))
    assert all(c.rerank_score is None for c in res.chunks)


# ---------------------------------------------------- multi-query / expansion
def test_multi_query_keeps_original_dedupes_and_fuses(services, indexed):
    services.retriever.query_llm = FakeLLM(['["Multiple regression", "how do several predictors explain a response", "regression"]'])
    res = services.retriever.retrieve("multiple regression", RetrievalConfig(strategy="multi_query", top_k=3, num_queries=3))
    queries = res.diagnostics["queries"]
    assert queries[0] == "multiple regression"
    assert len({q.lower() for q in queries}) == len(queries)  # "Multiple regression" deduped
    ids = [c.chunk_id for c in res.chunks]
    assert len(ids) == len(set(ids))


def test_multi_query_degrades_gracefully_when_llm_fails(services, indexed):
    services.retriever.query_llm = FakeLLM(["not json at all"])
    res = services.retriever.retrieve("regression predictors", RetrievalConfig(strategy="multi_query", top_k=2))
    assert res.chunks and res.diagnostics["queries"] == ["regression predictors"]
    assert any("failed" in n for n in res.diagnostics["notes"])


def test_expansion_searches_original_plus_enriched_query(services, indexed):
    services.retriever.query_llm = FakeLLM(['["centroid", "variance"]'])
    res = services.retriever.retrieve("k means", RetrievalConfig(strategy="expansion", top_k=2))
    assert res.diagnostics["queries"] == ["k means", "k means centroid variance"]
    assert res.chunks[0].page == 2


def test_expansion_without_llm_falls_back_to_original(services, indexed):
    services.retriever.query_llm = None
    res = services.retriever.retrieve("k means", RetrievalConfig(strategy="expansion", top_k=2))
    assert res.diagnostics["queries"] == ["k means"] and res.chunks


# --------------------------------------------------------------- parent docs
def test_parent_document_retrieval_expands_dedupes_and_keeps_provenance(services, indexed):
    base = services.retriever.retrieve("regression predictors", RetrievalConfig(top_k=5))
    res = services.retriever.retrieve("regression predictors", RetrievalConfig(top_k=5, parent_document=True))
    parent_ids = [c.parent_id for c in res.chunks]
    assert len(parent_ids) == len(set(parent_ids))  # deduplicated
    assert all(c.parent_expanded and c.page and c.filename and c.doc_id for c in res.chunks)
    assert all(len(c.text) >= len(c.original_text) for c in res.chunks)
    assert len(res.chunks) <= len(base.chunks)


def test_missing_parent_falls_back_to_child(services, indexed):
    services.parents.delete(indexed.doc_id)
    res = services.retriever.retrieve("regression", RetrievalConfig(top_k=2, parent_document=True))
    assert res.chunks and not any(c.parent_expanded for c in res.chunks)
    assert any("missing" in n for n in res.diagnostics["notes"])


# ---------------------------------------------------------------- compression
def test_compression_shortens_text_but_preserves_provenance(services, indexed):
    cfg_plain = RetrievalConfig(top_k=2, parent_document=True)
    cfg_comp = RetrievalConfig(top_k=2, parent_document=True, compression=True, compression_threshold=0.2)
    plain = services.retriever.retrieve("adjusted R-squared penalises predictors", cfg_plain)
    comp = services.retriever.retrieve("adjusted R-squared penalises predictors", cfg_comp)
    assert comp.chunks and all(c.compressed for c in comp.chunks)
    assert sum(len(c.text) for c in comp.chunks) <= sum(len(c.text) for c in plain.chunks)
    stored = {c.chunk_id: c for c in services.store.list_chunks()}
    for c in comp.chunks:  # provenance still points at the real stored chunk
        meta = stored[c.chunk_id].metadata
        assert (c.doc_id, c.page, c.filename) == (meta["doc_id"], meta["page"], meta["filename"])
        assert c.original_text == stored[c.chunk_id].text
    assert comp.chunks[0].text.strip() and "penalises" in comp.chunks[0].text
    assert comp.diagnostics["compression"]["chars_after"] <= comp.diagnostics["compression"]["chars_before"]


# ------------------------------------------------------------ empty retrieval
def test_empty_index_returns_nothing(services):
    for strat in ("similarity", "mmr", "hybrid"):
        assert services.retriever.retrieve("anything", RetrievalConfig(strategy=strat)).chunks == []
