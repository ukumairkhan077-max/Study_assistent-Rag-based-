"""Compare retrieval strategies on the SAME questions, corpus, embedder and top-k.

Outputs (evaluation/results/):
    comparison.json   full results incl. per-question details (read by the dashboard)
    comparison.csv    one row per strategy
    comparison.md     human-readable report

    python evaluation/compare_retrieval.py                 # retrieval metrics only, no LLM needed
    python evaluation/compare_retrieval.py --generate      # + answers, abstention stats
    python evaluation/compare_retrieval.py --generate --ragas
    python evaluation/compare_retrieval.py --offline       # smoke test, NOT a quality benchmark
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_rag as ev  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
COLUMNS = ["strategy", "recall_at_k", "hit_rate", "precision_at_k", "mrr", "avg_latency_ms", "avg_chunks",
           "avg_context_chars", "errors", "status"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--ragas", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--strategies", nargs="*", default=list(ev.STRATEGIES))
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    generate = args.generate or args.ragas

    print("Indexing controlled corpus into a temporary ChromaDB ...", flush=True)
    services, settings, tmp = ev.build_eval_services(offline=args.offline)
    questions = ev.load_dataset()
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": services.embedder.model_name,
        "reranker": "lexical-overlap (offline stand-in)" if args.offline else settings.reranker_model,
        "llm_model": settings.llm_model if (generate and settings.llm_configured) else None,
        "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap, "parent_chunk_size": settings.parent_chunk_size,
        "top_k": ev.K, "questions": len(questions),
        "offline_smoke_test": args.offline,
        "warning": ("OFFLINE SMOKE TEST: the hashing embedder is lexical, not semantic, and the reranker is a word-overlap "
                    "stand-in. These numbers validate the pipeline only and say nothing about real retrieval quality.")
        if args.offline else None,
        "metric_notes": "Page-level ground truth over a small dataset; differences of a few points are within noise.",
    }
    rows, details = [], {}
    for name in args.strategies:
        cfg, needs_llm = ev.STRATEGIES[name]
        row = {"strategy": name, "config": cfg.model_dump(), "status": "ok", "errors": 0}
        if needs_llm and not services.llm.configured:
            row.update(status="skipped: LLM not configured (query generation needs it)")
            rows.append(row)
            print(f"[skip] {name}")
            continue
        print(f"[run ] {name} ...", flush=True)
        t0 = time.perf_counter()
        try:
            res = ev.evaluate_strategy(services, name, cfg, questions, generate=generate, delay=args.delay)
        except Exception as exc:  # e.g. reranker model download failed
            row.update(status=f"failed: {type(exc).__name__}: {exc}")
            rows.append(row)
            continue
        if args.ragas:
            try:
                res["ragas"] = ev.run_ragas(res["per_question"], settings, services)
            except Exception as exc:
                res["ragas"] = {"error": f"{type(exc).__name__}: {exc}"}
        row.update(res["summary"])
        row["errors"] = len(res["errors"])
        row["error_messages"] = res["errors"][:5]
        row["wall_seconds"] = round(time.perf_counter() - t0, 1)
        if "ragas" in res:
            row["ragas"] = res["ragas"]
        if row["errors"] and row["errors"] == len(questions):
            row["status"] = "failed: every question errored (see error_messages)"
        for r in res["per_question"]:
            r.pop("contexts", None)  # keep the JSON small
        details[name] = res["per_question"]
        rows.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps({"meta": meta, "rows": rows, "per_question": details},
                                                          indent=1, ensure_ascii=False), encoding="utf-8")
    with open(args.out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS + ["ragas"], extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "ragas": json.dumps(r.get("ragas", ""))})
    (args.out / "comparison.md").write_text(report(meta, rows), encoding="utf-8")
    print(f"Wrote results to {args.out}")
    tmp.cleanup()


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"
    return str(v)


def report(meta: dict, rows: list[dict]) -> str:
    out = ["# Retrieval strategy comparison", ""]
    if meta.get("warning"):
        out += [f"> **{meta['warning']}**", ""]
    out += [f"- Created: {meta['created_at']}", f"- Embedding model: `{meta['embedding_model']}`",
            f"- Reranker: `{meta['reranker']}`", f"- Questions: {meta['questions']} (top-k = {meta['top_k']})",
            f"- {meta['metric_notes']}", ""]
    out += ["| Strategy | Recall@K | Hit@K | Precision@K | MRR | Latency ms | Chunks | Context chars | Errors | Status |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append("| " + " | ".join([r["strategy"], *(fmt(r.get(k)) for k in ("recall_at_k", "hit_rate", "precision_at_k", "mrr",
                   "avg_latency_ms", "avg_chunks", "avg_context_chars")), str(r.get("errors", 0)), r["status"]]) + " |")
    ok = [r for r in rows if r["status"] == "ok" and r.get("mrr") is not None]
    if len(ok) > 1 and not meta.get("offline_smoke_test"):
        out += ["", "## Observations (descriptive, from the table above)", ""]
        for key, label in (("recall_at_k", "Recall@K"), ("mrr", "MRR"), ("precision_at_k", "Precision@K")):
            best = max(r[key] for r in ok)
            winners = ", ".join(r["strategy"] for r in ok if r[key] == best)
            out.append(f"- Highest {label}: {winners} ({best:.3f}). Ties and small gaps are not evidence of a real difference on {meta['questions']} questions.")
        base = next((r for r in ok if r["strategy"] == "similarity"), None)
        if base:
            for r in ok:
                if r is not base and r["mrr"] < base["mrr"]:
                    out.append(f"- `{r['strategy']}` scored lower MRR than the similarity baseline ({r['mrr']:.3f} vs {base['mrr']:.3f}).")
    ragas_rows = [r for r in rows if isinstance(r.get("ragas"), dict) and "error" not in r["ragas"]]
    if ragas_rows:
        out += ["", "## RAGAS (answer-level, LLM-judged)", ""]
        metrics = sorted({k for r in ragas_rows for k in r["ragas"] if not k.endswith("_failed") and k not in ("n_scored", "ragas_version")})
        out += ["| Strategy | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
        for r in ragas_rows:
            out.append("| " + r["strategy"] + " | " + " | ".join(fmt(r["ragas"].get(m)) for m in metrics) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
