import { useState } from 'react'
import { Search } from 'lucide-react'
import { retrieve, errorMessage } from '../services/api'
import { useApp } from '../services/AppContext'
import ErrorBanner from './ErrorBanner'
import SourceCard from './SourceCard'

// Runs retrieval only (no LLM) so you can see exactly which chunks a configuration returns.
export default function Inspector({ initialQuery = '' }) {
  const { retrieval, selectedDocIds } = useApp()
  const [query, setQuery] = useState(initialQuery)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const run = async (e) => {
    e?.preventDefault()
    if (!query.trim()) return
    setLoading(true); setError('')
    try {
      setResult(await retrieve({ query, retrieval: { ...retrieval, doc_ids: selectedDocIds.length ? selectedDocIds : null } }))
    } catch (err) {
      setError(errorMessage(err)); setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="inspector">
      <form onSubmit={run} className="inspector__form">
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Query to inspect…" aria-label="Query to inspect" />
        <button className="btn btn--primary" disabled={loading}><Search size={14} /> {loading ? 'Searching…' : 'Retrieve'}</button>
      </form>
      <ErrorBanner message={error} onClose={() => setError('')} />
      {result && (
        <>
          <p className="hint">
            Strategy <b>{result.retrieval.strategy}</b> · {result.sources.length} passage(s) · best dense similarity{' '}
            <b>{result.top_dense_similarity ?? 'n/a'}</b>
            {result.diagnostics.retrieval_ms != null && <> · {result.diagnostics.retrieval_ms} ms</>}
          </p>
          {result.diagnostics.queries && <p className="hint">Queries used: {result.diagnostics.queries.map((q, i) => <code key={i}>{q}</code>)}</p>}
          {result.diagnostics.notes?.map((n, i) => <p key={i} className="hint hint--warn">{n}</p>)}
          {result.diagnostics.compression && (
            <p className="hint">Compression: {result.diagnostics.compression.chars_before} → {result.diagnostics.compression.chars_after} characters, {result.diagnostics.compression.passages_dropped} passage(s) dropped.</p>
          )}
          {result.sources.length === 0 && <p className="empty">Nothing retrieved.</p>}
          {result.sources.map((s) => <SourceCard key={s.chunk_id} source={s} defaultOpen />)}
        </>
      )}
    </div>
  )
}
