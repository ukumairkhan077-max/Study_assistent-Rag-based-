"""Evaluate one retrieval configuration on the controlled dataset (no frontend needed).

Two different questions are measured separately:
  * RETRIEVAL quality  - did the right pages come back, and how early?   (no LLM needed)
        Recall@K, Hit@K, Precision@K, MRR - computed against page-level ground truth.
  * ANSWER quality     - is the generated answer faithful/relevant/correct? (needs an LLM)
        RAGAS: faithfulness, response relevancy, context precision, context recall.
A strategy can retrieve well but generate badly (or vice-versa), so never merge them.

Examples (PowerShell, from the project root, venv active):
    python evaluation/evaluate_rag.py --strategy hybrid
    python evaluation/evaluate_rag.py --strategy hybrid_rerank --generate --ragas
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
sys.path.insert(0, str(HERE / "datasets"))

from app.api.deps import Services, build_services  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.schemas.chat import ChatRequest, RetrievalConfig  # noqa: E402
from app.services.embeddings import HashingEmbedder  # noqa: E402
from app.services.llm_service import LLMError  # noqa: E402

DATASET = HERE / "datasets" / "qa_dataset.json"
CORPUS_DIR = HERE / "datasets" / "corpus"

# name -> (config, needs_llm_for_query_generation)
K = 5
STRATEGIES: dict[str, tuple[RetrievalConfig, bool]] = {
    "similarity": (RetrievalConfig(strategy="similarity", top_k=K), False),
    "mmr": (RetrievalConfig(strategy="mmr", top_k=K, candidate_k=20, mmr_lambda=0.5), False),
    "hybrid": (RetrievalConfig(strategy="hybrid", top_k=K, candidate_k=20), False),
    "multi_query": (RetrievalConfig(strategy="multi_query", top_k=K, candidate_k=20, num_queries=3), True),
    "hybrid_rerank": (RetrievalConfig(strategy="hybrid", top_k=K, candidate_k=20, rerank=True), False),
    "similarity_parent": (RetrievalConfig(strategy="similarity", top_k=K, parent_document=True), False),
    "similarity_compression": (RetrievalConfig(strategy="similarity", top_k=K, compression=True), False),
    "hybrid_rerank_parent_compression": (
        RetrievalConfig(strategy="hybrid", top_k=K, candidate_k=20, rerank=True, parent_document=True, compression=True), False),
}


class OverlapReranker:
    """Offline stand-in used only with --offline (lexical overlap, not a real cross-encoder)."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(query.lower().split())
        return [float(len(q & set(t.lower().split()))) for t in texts]


# ------------------------------------------------------------------ setup
def load_dataset() -> list[dict]:
    return json.loads(DATASET.read_text(encoding="utf-8"))["questions"]


def build_eval_services(offline: bool = False, embedding_model: str | None = None) -> tuple[Services, Settings, tempfile.TemporaryDirectory]:
    """Index the controlled corpus into a THROWAWAY ChromaDB so your real library is untouched."""
    import build_test_corpus

    if not list(CORPUS_DIR.glob("*.pdf")):
        build_test_corpus.build(CORPUS_DIR)
    tmp = tempfile.TemporaryDirectory(prefix="rag_eval_")
    base = Settings()  # picks up backend/.env for LLM + embedding settings
    settings = base.model_copy(update={"chroma_path": Path(tmp.name) / "chroma", "data_dir": Path(tmp.name) / "data"})
    embedder = None
    if offline:
        settings.embedding_model = "hashing-test-embedder"
        embedder = HashingEmbedder()
    elif embedding_model:
        settings.embedding_model = embedding_model
    services = build_services(settings, embedder=embedder, reranker=OverlapReranker() if offline else None)
    for pdf in sorted(CORPUS_DIR.glob("*.pdf")):
        doc, _, path = services.ingestion.validate_and_register(pdf.name, pdf.read_bytes(), "eval")
        if path is not None:
            services.ingestion.index(doc.doc_id)
        info = services.registry.get(doc.doc_id)
        if info.status != "indexed":
            raise RuntimeError(f"Could not index {pdf.name}: {info.error}")
    return services, settings, tmp


# ---------------------------------------------------------------- metrics
def retrieval_metrics(retrieved: list[tuple[str, int]], relevant: list[dict]) -> dict:
    """Page-level metrics for one answerable question. `retrieved` is the ordered list of
    (file, page) of the final chunks. Several chunks from one page count once for recall."""
    rel = {(r["file"], r["page"]) for r in relevant}
    found = rel & set(retrieved)
    first = next((i for i, p in enumerate(retrieved, 1) if p in rel), None)
    return {
        "recall": len(found) / len(rel),
        "hit": 1.0 if found else 0.0,
        "precision": (sum(1 for p in retrieved if p in rel) / len(retrieved)) if retrieved else 0.0,
        "rr": (1.0 / first) if first else 0.0,
    }


def evaluate_strategy(services: Services, name: str, cfg: RetrievalConfig, questions: list[dict], *,
                      generate: bool = False, delay: float = 0.0) -> dict:
    """Run every question through retrieval (and optionally generation) with identical settings."""
    per_q, errors = [], []
    for q in questions:
        rec = {"id": q["id"], "type": q["type"], "question": q["question"], "reference": q["reference"]}
        try:
            t0 = time.perf_counter()
            result, top_sim = services.pipeline.retrieve(q["question"], cfg)
            rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            rec["retrieved"] = [{"file": c.filename, "page": c.page, "chunk_id": c.chunk_id} for c in result.chunks]
            rec["contexts"] = [c.text for c in result.chunks]
            rec["context_chars"] = sum(len(t) for t in rec["contexts"])
            rec["n_chunks"] = len(result.chunks)
            rec["top_dense_similarity"] = None if top_sim is None else round(top_sim, 4)
            rec["notes"] = result.diagnostics.get("notes", [])
            if q["relevant_pages"]:
                rec.update(retrieval_metrics([(r["file"], r["page"]) for r in rec["retrieved"]], q["relevant_pages"]))
            if generate:
                rec.update(_generate(services, q, cfg))
                if delay:
                    time.sleep(delay)  # stay under free-tier rate limits
        except LLMError as exc:
            rec["error"] = f"LLM: {exc}"
            errors.append(f"{q['id']}: {rec['error']}")
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(f"{q['id']}: {rec['error']}")
        per_q.append(rec)
    return {"name": name, "config": cfg.model_dump(), "per_question": per_q, "errors": errors,
            "summary": summarise(per_q, generate)}


def _generate(services: Services, q: dict, cfg: RetrievalConfig) -> dict:
    t0 = time.perf_counter()
    resp = services.pipeline.answer(ChatRequest(question=q["question"], retrieval=cfg))
    return {"answer": resp.answer, "supported": resp.supported, "generation_ms": round((time.perf_counter() - t0) * 1000, 1),
            "abstained": (not resp.supported)}


def summarise(per_q: list[dict], generate: bool) -> dict:
    ok = [r for r in per_q if "error" not in r]
    ans = [r for r in ok if "recall" in r]
    unans = [r for r in ok if r["type"] == "unanswerable"]
    s = {
        "questions": len(per_q), "answerable_scored": len(ans), "errors": len(per_q) - len(ok),
        "recall_at_k": _m(ans, "recall"), "hit_rate": _m(ans, "hit"), "precision_at_k": _m(ans, "precision"),
        "mrr": _m(ans, "rr"), "avg_latency_ms": _m(ok, "latency_ms"), "avg_chunks": _m(ok, "n_chunks"),
        "avg_context_chars": _m(ok, "context_chars"),
        "avg_top_similarity_answerable": _m(ans, "top_dense_similarity"),
        "avg_top_similarity_unanswerable": _m(unans, "top_dense_similarity"),
    }
    if generate:
        gen = [r for r in ok if "abstained" in r]
        s["abstained_on_unanswerable"] = _rate([r for r in gen if r["type"] == "unanswerable"], "abstained")
        s["abstained_on_answerable"] = _rate([r for r in gen if "recall" in r], "abstained")
        s["avg_generation_ms"] = _m(gen, "generation_ms")
    return s


def _m(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(mean(vals), 4) if vals else None


def _rate(rows, key):
    return round(sum(1 for r in rows if r[key]) / len(rows), 4) if rows else None


# ------------------------------------------------------------------ RAGAS
def run_ragas(per_q: list[dict], settings: Settings, services: Services) -> dict:
    """Score generated answers with RAGAS (pinned: ragas==0.3.9, see requirements-eval.txt).
    Uses your configured OpenAI-compatible LLM as the judge and the local embedding model for
    response relevancy. Judge calls consume LLM quota; failures yield NaN and are reported."""
    from langchain_core.embeddings import Embeddings
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy)

    if not settings.llm_configured:
        return {"error": "LLM not configured (needed as RAGAS judge)"}

    class _Emb(Embeddings):
        def __init__(self, e):
            self.e = e

        def embed_documents(self, texts):
            return self.e.embed(list(texts)).tolist()

        def embed_query(self, text):
            return self.e.embed([text])[0].tolist()

    rows = [r for r in per_q if "answer" in r and r.get("contexts") and r["type"] != "unanswerable" and "error" not in r]
    if not rows:
        return {"error": "no generated answers to score"}
    ds = EvaluationDataset.from_list([{"user_input": r["question"], "retrieved_contexts": r["contexts"],
                                       "response": r["answer"], "reference": r["reference"]} for r in rows])
    llm = LangchainLLMWrapper(ChatOpenAI(model=settings.llm_model, api_key=settings.llm_api_key,
                                         base_url=settings.llm_base_url or None, temperature=0,
                                         timeout=settings.llm_timeout_seconds))
    emb = LangchainEmbeddingsWrapper(_Emb(services.embedder))
    metrics = [Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithReference(), LLMContextRecall()]
    result = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb, raise_exceptions=False,
                      run_config=RunConfig(timeout=180, max_retries=5, max_workers=2), show_progress=True)
    df = result.to_pandas()
    out: dict = {"n_scored": len(rows), "ragas_version": _ragas_version()}
    for col in df.columns:
        if col in ("user_input", "retrieved_contexts", "response", "reference"):
            continue
        vals = [v for v in df[col].tolist() if isinstance(v, (int, float)) and not math.isnan(v)]
        out[col] = round(mean(vals), 4) if vals else None
        out[f"{col}_failed"] = int(len(df) - len(vals))  # NaN = judge call failed (quota, parse error...)
    return out


def _ragas_version() -> str:
    import ragas
    return ragas.__version__


# -------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="similarity", choices=list(STRATEGIES))
    ap.add_argument("--generate", action="store_true", help="also generate answers with the configured LLM")
    ap.add_argument("--ragas", action="store_true", help="score answers with RAGAS (implies --generate)")
    ap.add_argument("--offline", action="store_true", help="hashing embedder + lexical reranker: pipeline smoke test only")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between LLM calls")
    args = ap.parse_args()
    services, settings, tmp = build_eval_services(offline=args.offline)
    cfg, needs_llm = STRATEGIES[args.strategy]
    if needs_llm and not services.llm.configured:
        sys.exit(f"Strategy '{args.strategy}' needs an LLM for query generation; configure backend/.env.")
    res = evaluate_strategy(services, args.strategy, cfg, load_dataset(), generate=args.generate or args.ragas, delay=args.delay)
    if args.ragas:
        res["ragas"] = run_ragas(res["per_question"], settings, services)
    for r in res["per_question"]:
        r.pop("contexts", None)
    print(json.dumps({k: res[k] for k in ("name", "summary", "errors", *(["ragas"] if args.ragas else []))}, indent=2))
    tmp.cleanup()


if __name__ == "__main__":
    main()
