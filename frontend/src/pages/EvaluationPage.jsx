import { useCallback, useEffect, useState } from 'react'
import { Loader2, Play } from 'lucide-react'
import ErrorBanner from '../components/ErrorBanner'
import { getEvaluation, getEvaluationStatus, runEvaluation, errorMessage } from '../services/api'

const METRICS = [['recall_at_k', 'Recall@K'], ['hit_rate', 'Hit@K'], ['precision_at_k', 'Precision@K'], ['mrr', 'MRR']]
const fmt = (v) => (v == null ? '-' : typeof v === 'number' ? (Math.abs(v) < 100 ? v.toFixed(3) : Math.round(v)) : v)

export default function EvaluationPage() {
  const [data, setData] = useState(null)
  const [status, setStatus] = useState({ state: 'idle' })
  const [error, setError] = useState('')
  const [empty, setEmpty] = useState(false)

  const load = useCallback(async () => {
    try { setData(await getEvaluation()); setEmpty(false) } catch (e) {
      if (e.response?.status === 404) setEmpty(true); else setError(errorMessage(e))
    }
  }, [])
  useEffect(() => { load(); getEvaluationStatus().then(setStatus).catch(() => {}) }, [load])

  useEffect(() => {
    if (status.state !== 'running') return
    const t = setInterval(async () => {
      const s = await getEvaluationStatus().catch(() => null)
      if (s) { setStatus(s); if (s.state !== 'running') load() }
    }, 2500)
    return () => clearInterval(t)
  }, [status.state, load])

  const start = async () => {
    setError('')
    try { await runEvaluation(); setStatus({ state: 'running' }) } catch (e) { setError(errorMessage(e)) }
  }
  const rows = data?.rows || []
  const ragasCols = [...new Set(rows.flatMap((r) => Object.keys(r.ragas || {})).filter((k) => !k.endsWith('_failed') && !['n_scored', 'ragas_version', 'error'].includes(k)))]

  return (
    <div className="page">
      <h1>Evaluation</h1>
      <p className="hint">Retrieval metrics are computed against page-level ground truth on a controlled 3-PDF corpus. Everything shown here was measured by <code>evaluation/compare_retrieval.py</code>; nothing is simulated.</p>
      <div className="card row">
        <button className="btn btn--primary" onClick={start} disabled={status.state === 'running'}>
          {status.state === 'running' ? <><Loader2 size={14} className="spin" /> Running…</> : <><Play size={14} /> Run retrieval comparison</>}
        </button>
        <small>Uses your configured embedding and reranker models (first run downloads them). RAGAS answer metrics must be run from the command line (see README).</small>
      </div>
      {status.state === 'failed' && <ErrorBanner message={`Last run failed: ${status.log_tail || 'see evaluation/results/last_run.log'}`} />}
      <ErrorBanner message={error} onClose={() => setError('')} />
      {empty && <p className="empty">No results yet. Click “Run retrieval comparison”, or run the script from a terminal.</p>}
      {data && (
        <>
          {data.meta.warning && <div className="banner banner--warn" role="alert">{data.meta.warning}</div>}
          <p className="hint">
            {data.meta.created_at} · embeddings <code>{data.meta.embedding_model}</code> · {data.meta.questions} questions · top-k {data.meta.top_k}
            {data.meta.llm_model && <> · LLM <code>{data.meta.llm_model}</code></>}
          </p>
          <section className="card">
            <h2>Retrieval quality</h2>
            <div className="scroll-x">
              <table className="table">
                <thead><tr><th>Strategy</th>{METRICS.map(([, l]) => <th key={l}>{l}</th>)}<th>Latency (ms)</th><th>Chunks</th><th>Context chars</th><th>Status</th></tr></thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.strategy}>
                      <td><b>{r.strategy}</b></td>
                      {METRICS.map(([k]) => (
                        <td key={k}>
                          {fmt(r[k])}
                          {r[k] != null && <div className="bar" aria-hidden="true"><div className="bar__fill" style={{ width: `${Math.min(100, r[k] * 100)}%` }} /></div>}
                        </td>
                      ))}
                      <td>{fmt(r.avg_latency_ms)}</td><td>{fmt(r.avg_chunks)}</td><td>{fmt(r.avg_context_chars)}</td>
                      <td>{r.status === 'ok' ? (r.errors ? `${r.errors} error(s)` : 'ok') : r.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="hint">{data.meta.metric_notes}</p>
          </section>
          {ragasCols.length > 0 && (
            <section className="card">
              <h2>RAGAS (answer quality, LLM-judged)</h2>
              <div className="scroll-x"><table className="table">
                <thead><tr><th>Strategy</th>{ragasCols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
                <tbody>{rows.filter((r) => r.ragas && !r.ragas.error).map((r) => (
                  <tr key={r.strategy}><td><b>{r.strategy}</b></td>{ragasCols.map((c) => <td key={c}>{fmt(r.ragas[c])}</td>)}</tr>
                ))}</tbody>
              </table></div>
            </section>
          )}
        </>
      )}
    </div>
  )
}
