"""Tests for the evaluation harness itself (metrics maths + offline end-to-end run)."""
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[2] / "evaluation"
sys.path.insert(0, str(EVAL))
import compare_retrieval as cmp  # noqa: E402
import evaluate_rag as ev  # noqa: E402


def test_retrieval_metrics_math():
    rel = [{"file": "a.pdf", "page": 1}, {"file": "a.pdf", "page": 2}]
    m = ev.retrieval_metrics([("b.pdf", 1), ("a.pdf", 2), ("a.pdf", 2), ("c.pdf", 9)], rel)
    assert m["recall"] == 0.5          # 1 of 2 relevant pages found
    assert m["hit"] == 1.0
    assert m["rr"] == 0.5              # first relevant at rank 2
    assert m["precision"] == 0.5       # 2 of 4 retrieved chunks are on relevant pages


def test_retrieval_metrics_miss():
    m = ev.retrieval_metrics([("b.pdf", 1)], [{"file": "a.pdf", "page": 1}])
    assert (m["recall"], m["hit"], m["rr"], m["precision"]) == (0.0, 0.0, 0.0, 0.0)


def test_dataset_is_well_formed_and_points_at_real_pages():
    import build_test_corpus as bc
    qs = ev.load_dataset()
    assert {q["type"] for q in qs} >= {"factual", "multi_chunk", "technical_term", "unanswerable", "ambiguous", "similar_distractor"}
    for q in qs:
        for r in q["relevant_pages"]:
            assert r["file"] in bc.CORPUS and 1 <= r["page"] <= len(bc.CORPUS[r["file"]])
    assert all(q["reference"] for q in qs)


def test_ragas_imports_resolve_and_missing_llm_is_reported(monkeypatch):
    from app.core.config import Settings
    s = Settings(llm_api_key="", llm_model="", _env_file=None)
    out = ev.run_ragas([], s, services=None)
    assert "error" in out and "not configured" in out["error"]


@pytest.mark.parametrize("name", ["similarity", "hybrid", "hybrid_rerank_parent_compression"])
def test_offline_end_to_end_strategy_run(name):
    services, settings, tmp = ev.build_eval_services(offline=True)
    try:
        cfg, _ = ev.STRATEGIES[name]
        res = ev.evaluate_strategy(services, name, cfg, ev.load_dataset())
        assert res["errors"] == [] and res["summary"]["answerable_scored"] == 14
        assert 0 < res["summary"]["recall_at_k"] <= 1
    finally:
        tmp.cleanup()


def test_report_flags_offline_runs_and_formats_numbers():
    meta = {"created_at": "t", "embedding_model": "m", "reranker": "r", "questions": 1, "top_k": 5,
            "metric_notes": "n", "warning": "SMOKE", "offline_smoke_test": True}
    rows = [{"strategy": "similarity", "status": "ok", "recall_at_k": 1.0, "mrr": 0.5, "avg_context_chars": 1833.9375, "errors": 0}]
    md = cmp.report(meta, rows)
    assert "SMOKE" in md and "1834" in md and "Observations" not in md
