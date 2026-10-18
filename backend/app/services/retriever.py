"""Retrieval strategies and post-retrieval stages.

Candidate generators (pick ONE via ``RetrievalConfig.strategy``):
  similarity   dense top-k by cosine similarity (baseline)
  mmr          Maximum Marginal Relevance over a dense candidate pool
  hybrid       dense + BM25 fused with Reciprocal Rank Fusion
  expansion    original query + LLM-added keywords/synonyms as ONE enriched query
  multi_query  several LLM paraphrases retrieved separately, fused with RRF

Sequential stages (combine freely with any generator):
  rerank           cross-encoder rescoring of a wider candidate pool
  parent_document  swap each child for its (deduplicated) parent passage
  compression      keep only the sentences relevant to the query
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from ..schemas.chat import RetrievalConfig
from .embeddings import Embedder
from .registry import ParentStore
from .vector_store import StoredChunk, VectorStore

RRF_K = 60  # standard constant from the original RRF paper (Cormack et al., 2009)
MAX_PARENT_CHARS = 2500


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    text: str
    score: float = 0.0
    similarity: Optional[float] = None  # dense cosine similarity if known
    rerank_score: Optional[float] = None
    original_score: Optional[float] = None  # score before reranking
    parent_expanded: bool = False
    compressed: bool = False
    original_text: str = ""  # child text before parent/compression (provenance)
    parent_id: str = ""


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    diagnostics: dict = field(default_factory=dict)


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """A cross-encoder reads (query, passage) TOGETHER and outputs a relevance
    score, so it models their interaction and is more accurate than comparing two
    independent embeddings - but it needs one forward pass per candidate, so it
    is only affordable on a short candidate list."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy import

            self._model = CrossEncoder(self.model_name, device="cpu")
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]


class QueryGenerator(Protocol):
    def complete(self, system: str, user: str) -> str: ...


# ------------------------------------------------------------------ helpers
def _to_chunk(sc: StoredChunk, score: float, similarity: Optional[float] = None) -> RetrievedChunk:
    m = sc.metadata
    return RetrievedChunk(
        chunk_id=sc.chunk_id,
        doc_id=m.get("doc_id", ""),
        filename=m.get("filename", ""),
        page=int(m.get("page", 0)),
        text=sc.text,
        score=score,
        similarity=similarity if similarity is not None else sc.similarity,
        original_text=sc.text,
        parent_id=m.get("parent_id", ""),
    )


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def reciprocal_rank_fusion(rankings: list[list[RetrievedChunk]], k: int = RRF_K) -> list[RetrievedChunk]:
    """score(d) = sum over rankings of 1 / (k + rank). Uses ranks only, so it needs
    no score normalisation between cosine similarity and BM25. Duplicates are
    merged by chunk_id; the dense similarity is kept when any list had it."""
    fused: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, ch in enumerate(ranking, start=1):
            scores[ch.chunk_id] = scores.get(ch.chunk_id, 0.0) + 1.0 / (k + rank)
            if ch.chunk_id not in fused:
                fused[ch.chunk_id] = ch
            elif fused[ch.chunk_id].similarity is None and ch.similarity is not None:
                fused[ch.chunk_id].similarity = ch.similarity
    out = []
    for cid, ch in fused.items():
        ch = RetrievedChunk(**{**ch.__dict__, "score": scores[cid]})
        out.append(ch)
    return sorted(out, key=lambda c: -c.score)


def mmr_select(
    query_vec: np.ndarray, cand_vecs: np.ndarray, k: int, lambda_mult: float
) -> list[int]:
    """Greedy MMR. At each step pick argmax of
        lambda * sim(query, d) - (1 - lambda) * max_{s in selected} sim(d, s)
    Vectors are assumed L2-normalised so dot product == cosine similarity."""
    if len(cand_vecs) == 0:
        return []
    rel = cand_vecs @ query_vec
    selected: list[int] = []
    remaining = list(range(len(cand_vecs)))
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: rel[i])
        else:
            sel = cand_vecs[selected]
            best = max(remaining, key=lambda i: lambda_mult * rel[i] - (1 - lambda_mult) * float(np.max(sel @ cand_vecs[i])))
        selected.append(best)
        remaining.remove(best)
    return selected


def _parse_json_list(raw: str) -> list[str]:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    m = re.search(r"\[.*\]", raw, flags=re.S)
    data = json.loads(m.group(0) if m else raw)
    return [str(x).strip() for x in data if str(x).strip()]


# ---------------------------------------------------------------- retriever
class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        parents: Optional[ParentStore] = None,
        reranker: Optional[Reranker] = None,
        query_llm: Optional[QueryGenerator] = None,
    ):
        self.store, self.embedder, self.parents = store, embedder, parents
        self.reranker, self.query_llm = reranker, query_llm
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_chunks: list[StoredChunk] = []
        self._bm25_version = -1

    # ----------------------------------------------------------- primitives
    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.embed([query])[0]

    def dense(self, query: str, k: int, where: Optional[dict]) -> list[RetrievedChunk]:
        res = self.store.query(self.embed_query(query), k, where)
        return [_to_chunk(r, score=r.similarity or 0.0) for r in res]

    def top_dense_similarity(self, query: str, where: Optional[dict]) -> Optional[float]:
        res = self.store.query(self.embed_query(query), 1, where)
        return res[0].similarity if res else None

    def _ensure_bm25(self) -> None:
        """Sparse index is rebuilt lazily whenever the vector store changed
        (add/delete bump store.version), so it can never serve deleted chunks."""
        if self._bm25 is not None and self._bm25_version == self.store.version:
            return
        self._bm25_chunks = self.store.list_chunks()
        corpus = [tokenize(c.text) for c in self._bm25_chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._bm25_version = self.store.version

    def sparse(self, query: str, k: int, doc_ids: Optional[list[str]], collection: Optional[str]) -> list[RetrievedChunk]:
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        out: list[RetrievedChunk] = []
        for i in order:
            if scores[i] <= 0:
                break
            c = self._bm25_chunks[i]
            if doc_ids and c.metadata.get("doc_id") not in doc_ids:
                continue
            if collection and c.metadata.get("collection") != collection:
                continue
            out.append(_to_chunk(c, score=float(scores[i]), similarity=None))
            if len(out) >= k:
                break
        return out

    def hybrid(self, query: str, k: int, cfg: RetrievalConfig, where: Optional[dict]) -> list[RetrievedChunk]:
        d = self.dense(query, k, where)
        s = self.sparse(query, k, cfg.doc_ids, cfg.collection)
        return reciprocal_rank_fusion([d, s])

    def mmr(self, query: str, cfg: RetrievalConfig, where: Optional[dict], k: int) -> list[RetrievedChunk]:
        """Fetch `candidate_k` nearest chunks, then greedily pick `k` of them by MMR."""
        qv = self.embed_query(query)
        pool = self.store.query(qv, max(cfg.candidate_k, k), where, with_embeddings=True)
        if not pool:
            return []
        vecs = np.vstack([p.embedding for p in pool])
        idx = mmr_select(qv, vecs, k, cfg.mmr_lambda)
        return [_to_chunk(pool[i], score=pool[i].similarity or 0.0) for i in idx]

    # --------------------------------------------------- LLM-based queries
    def _generate_queries(self, query: str, n: int, mode: str, diag: dict) -> list[str]:
        if self.query_llm is None:
            diag.setdefault("notes", []).append(f"{mode}: no LLM configured, used original query only")
            return []
        if mode == "multi_query":
            system = (
                "You rewrite a student's question as different search queries for a document index. "
                f"Return ONLY a JSON array of {n} strings. Each must ask for the SAME information as the "
                "original using different wording or a different angle. Do not introduce other topics."
            )
        else:
            system = (
                "You add search keywords to a student's question. Return ONLY a JSON array of up to "
                f"{n * 3} short keywords, synonyms or technical terms that closely relate to the question's topic. "
                "Do not include unrelated topics."
            )
        try:
            return _parse_json_list(self.query_llm.complete(system, query))
        except Exception as exc:  # LLM down / bad JSON: degrade to the original query
            diag.setdefault("notes", []).append(f"{mode} failed ({type(exc).__name__}); used original query only")
            return []

    def _base_search(self, q: str, k: int, cfg: RetrievalConfig, where: Optional[dict]) -> list[RetrievedChunk]:
        return self.hybrid(q, k, cfg, where) if cfg.hybrid_base else self.dense(q, k, where)

    def multi_query(self, query: str, cfg: RetrievalConfig, where: Optional[dict], diag: dict) -> list[RetrievedChunk]:
        extra = self._generate_queries(query, cfg.num_queries, "multi_query", diag)
        queries, seen = [], set()
        for q in [query] + extra:  # original always first and always kept
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                queries.append(q.strip())
        queries = queries[: cfg.num_queries + 1]
        diag["queries"] = queries
        return reciprocal_rank_fusion([self._base_search(q, cfg.candidate_k, cfg, where) for q in queries])

    def expansion(self, query: str, cfg: RetrievalConfig, where: Optional[dict], diag: dict) -> list[RetrievedChunk]:
        terms = self._generate_queries(query, cfg.num_queries, "expansion", diag)
        expanded = query if not terms else f"{query} {' '.join(dict.fromkeys(terms))}"
        diag["queries"] = [query] if not terms else [query, expanded]
        lists = [self._base_search(query, cfg.candidate_k, cfg, where)]
        if terms:  # the enriched query is searched in addition to, not instead of, the original
            lists.append(self._base_search(expanded, cfg.candidate_k, cfg, where))
        return reciprocal_rank_fusion(lists)

    # --------------------------------------------------------------- stages
    def _rerank(self, query: str, cands: list[RetrievedChunk], top_k: int, diag: dict) -> list[RetrievedChunk]:
        if self.reranker is None:
            diag.setdefault("notes", []).append("rerank requested but no reranker available; skipped")
            return cands[:top_k]
        scores = self.reranker.score(query, [c.text for c in cands])
        for c, s in zip(cands, scores):
            c.original_score = c.score  # keep the retrieval score separately
            c.rerank_score = s
        ranked = sorted(cands, key=lambda c: -(c.rerank_score or 0.0))
        for c in ranked:
            c.score = c.rerank_score or 0.0
        return ranked[:top_k]

    def _expand_parents(self, chunks: list[RetrievedChunk], diag: dict) -> list[RetrievedChunk]:
        if self.parents is None:
            return chunks
        out, seen, missing, cache = [], set(), 0, {}
        for c in chunks:
            if not c.parent_id:
                key, text = c.chunk_id, None
            else:
                if c.doc_id not in cache:
                    cache[c.doc_id] = self.parents.load(c.doc_id)
                p = cache[c.doc_id].get(c.parent_id)
                key, text = c.parent_id, (p or {}).get("text")
                if p is None:
                    missing += 1  # deleted/missing parent: keep the child safely
            if key in seen:
                continue  # several children of one parent -> a single parent passage
            seen.add(key)
            if text:
                c.text = text[:MAX_PARENT_CHARS]
                c.parent_expanded = True
            out.append(c)
        if missing:
            diag.setdefault("notes", []).append(f"{missing} parent passage(s) missing; used child text")
        return out

    def _compress(self, query: str, chunks: list[RetrievedChunk], threshold: float, diag: dict) -> list[RetrievedChunk]:
        """Embedding-based extractive compression: within each retrieved passage keep
        sentences whose similarity to the query >= threshold (always at least the best
        one), in original order. Drop a passage if even its best sentence is below
        threshold/2. Provenance (doc, page, chunk id) is untouched; the uncompressed
        child text stays in ``original_text`` and ``compressed`` is set to True."""
        from .chunking import split_sentences

        qv = self.embed_query(query)
        kept: list[RetrievedChunk] = []
        before = sum(len(c.text) for c in chunks)
        for c in chunks:
            sents = split_sentences(c.text) or [c.text]
            sims = self.embedder.embed(sents) @ qv
            if float(np.max(sims)) < threshold / 2:
                continue
            keep = [i for i, s in enumerate(sims) if s >= threshold] or [int(np.argmax(sims))]
            c.text = " ".join(sents[i] for i in sorted(keep))
            c.compressed = True
            kept.append(c)
        diag["compression"] = {
            "chars_before": before,
            "chars_after": sum(len(c.text) for c in kept),
            "passages_dropped": len(chunks) - len(kept),
        }
        return kept

    # ----------------------------------------------------------- entrypoint
    def retrieve(self, query: str, cfg: RetrievalConfig) -> RetrievalResult:
        diag: dict = {"strategy": cfg.strategy}
        t0 = time.perf_counter()
        where = VectorStore.build_where(cfg.doc_ids, cfg.collection)
        pool = max(cfg.candidate_k, cfg.top_k) if (cfg.rerank or cfg.strategy != "similarity") else cfg.top_k
        if cfg.strategy == "similarity":
            cands = self.dense(query, pool, where)
        elif cfg.strategy == "mmr":
            cands = self.mmr(query, cfg, where, pool if cfg.rerank else cfg.top_k)
        elif cfg.strategy == "hybrid":
            cands = self.hybrid(query, pool, cfg, where)
        elif cfg.strategy == "expansion":
            cands = self.expansion(query, cfg, where, diag)
        elif cfg.strategy == "multi_query":
            cands = self.multi_query(query, cfg, where, diag)
        else:  # pragma: no cover - guarded by pydantic Literal
            raise ValueError(cfg.strategy)
        diag["candidates"] = len(cands)
        if cfg.rerank and cands:
            chunks = self._rerank(query, cands, cfg.top_k, diag)
        else:
            chunks = cands[: cfg.top_k]
        if cfg.parent_document and chunks:
            chunks = self._expand_parents(chunks, diag)
        if cfg.compression and chunks:
            chunks = self._compress(query, chunks, cfg.compression_threshold, diag)
        diag["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return RetrievalResult(chunks=chunks, diagnostics=diag)
