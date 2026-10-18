import { useState } from 'react'
import { Loader2 } from 'lucide-react'
import ErrorBanner from '../components/ErrorBanner'
import SourceCard from '../components/SourceCard'
import { study, errorMessage } from '../services/api'
import { useApp } from '../services/AppContext'

const TOOLS = [
  { id: 'summarize', label: 'Summarise document', needsDoc: true, needsTopic: false },
  { id: 'explain', label: 'Explain simply', needsTopic: true },
  { id: 'quiz', label: 'MCQ quiz', needsTopic: true, count: true },
  { id: 'short-answer', label: 'Short-answer questions', needsTopic: true, count: true },
  { id: 'flashcards', label: 'Flashcards', needsTopic: true, count: true },
]

function Unsupported({ ok }) {
  return ok ? null : <span className="badge badge--warn">no valid source - treat as unsupported</span>
}

function Quiz({ items, retrieval }) {
  const [picked, setPicked] = useState({})
  const [explain, setExplain] = useState({})
  const ask = async (i, it) => {
    setExplain((e) => ({ ...e, [i]: { loading: true } }))
    try {
      const r = await study('explain-answer', {
        question: it.question, selected_answer: it.options[picked[i]], correct_answer: it.options[it.answer_index], retrieval,
      })
      setExplain((e) => ({ ...e, [i]: { text: r.text, warnings: r.warnings } }))
    } catch (err) {
      setExplain((e) => ({ ...e, [i]: { error: errorMessage(err) } }))
    }
  }
  return items.map((it, i) => (
    <div key={i} className="card card--tight">
      <p><b>{i + 1}. {it.question}</b> <Unsupported ok={it.source_supported} /></p>
      {it.options.map((o, j) => {
        const done = picked[i] !== undefined
        const cls = done ? (j === it.answer_index ? 'opt opt--correct' : j === picked[i] ? 'opt opt--wrong' : 'opt') : 'opt'
        return <button key={j} className={cls} disabled={done} onClick={() => setPicked({ ...picked, [i]: j })}>{String.fromCharCode(65 + j)}. {o}</button>
      })}
      {picked[i] !== undefined && (
        <div className="reveal">
          <p><b>{picked[i] === it.answer_index ? 'Correct.' : `Answer: ${it.options[it.answer_index]}`}</b> {it.explanation} {it.source_ids?.map((s) => <span key={s} className="chip">{s}</span>)}</p>
          {!explain[i] && <button className="btn" onClick={() => ask(i, it)}>Explain with sources</button>}
          {explain[i]?.loading && <Loader2 className="spin" size={16} />}
          {explain[i]?.error && <ErrorBanner message={explain[i].error} />}
          {explain[i]?.text && <p className="explain">{explain[i].text}</p>}
        </div>
      )}
    </div>
  ))
}

function Flashcard({ item }) {
  const [flipped, setFlipped] = useState(false)
  return (
    <button className={`flashcard ${flipped ? 'flashcard--flipped' : ''}`} onClick={() => setFlipped(!flipped)} aria-label="Flip card">
      <span className="flashcard__text">{flipped ? item.back : item.front}</span>
      <small>{item.source_ids?.join(', ')} <Unsupported ok={item.source_supported} /></small>
    </button>
  )
}

export default function StudyPage() {
  const { documents, retrieval } = useApp()
  const indexed = documents.filter((d) => d.status === 'indexed')
  const [tool, setTool] = useState(TOOLS[0])
  const [docId, setDocId] = useState('')
  const [topic, setTopic] = useState('')
  const [count, setCount] = useState(5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const run = async (e) => {
    e.preventDefault()
    setLoading(true); setError(''); setResult(null)
    try {
      setResult(await study(tool.id, { topic, doc_id: docId || null, count, retrieval }))
    } catch (err) { setError(errorMessage(err)) } finally { setLoading(false) }
  }
  const disabled = loading || (tool.needsDoc && !docId) || (tool.needsTopic && !topic.trim())

  return (
    <div className="page">
      <h1>Study tools</h1>
      <div className="tabs" role="tablist">
        {TOOLS.map((t) => (
          <button key={t.id} role="tab" aria-selected={t.id === tool.id} className={`tab ${t.id === tool.id ? 'is-active' : ''}`} onClick={() => { setTool(t); setResult(null) }}>{t.label}</button>
        ))}
      </div>
      <form className="card row row--wrap" onSubmit={run}>
        <label className="field field--inline"><span>Document {tool.needsDoc ? '' : '(optional)'}</span>
          <select value={docId} onChange={(e) => setDocId(e.target.value)}>
            <option value="">{tool.needsDoc ? 'Select…' : 'All documents'}</option>
            {indexed.map((d) => <option key={d.doc_id} value={d.doc_id}>{d.filename}</option>)}
          </select>
        </label>
        {tool.needsTopic && (
          <label className="field field--grow"><span>Topic</span>
            <input value={topic} maxLength={500} onChange={(e) => setTopic(e.target.value)} placeholder="e.g. multiple regression" />
          </label>
        )}
        {tool.count && (
          <label className="field field--inline"><span>Items</span>
            <input type="number" min={1} max={15} value={count} onChange={(e) => setCount(+e.target.value)} />
          </label>
        )}
        <button className="btn btn--primary" disabled={disabled}>{loading ? 'Generating…' : 'Generate'}</button>
      </form>
      <ErrorBanner message={error} onClose={() => setError('')} />
      {loading && <p className="hint"><Loader2 className="spin" size={14} /> Retrieving from your documents and generating…</p>}
      {result && (
        <>
          {result.warnings?.map((w, i) => <p key={i} className="hint hint--warn">{w}</p>)}
          {result.text && <div className="card prose">{result.text}</div>}
          {result.kind === 'mcq' && <Quiz items={result.items} retrieval={retrieval} />}
          {result.kind === 'short_answer' && result.items.map((it, i) => (
            <details key={i} className="card card--tight"><summary>{it.question} <Unsupported ok={it.source_supported} /></summary><p>{it.model_answer} {it.source_ids?.map((s) => <span key={s} className="chip">{s}</span>)}</p></details>
          ))}
          {result.kind === 'flashcards' && <div className="flashgrid">{result.items.map((it, i) => <Flashcard key={i} item={it} />)}</div>}
          {result.sources?.length > 0 && (
            <div className="sources"><h4>Sources used</h4>{result.sources.map((s) => <SourceCard key={s.chunk_id} source={s} highlight={s.cited} />)}</div>
          )}
        </>
      )}
    </div>
  )
}
