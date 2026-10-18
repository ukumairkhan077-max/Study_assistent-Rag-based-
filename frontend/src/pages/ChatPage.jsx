import { useEffect, useRef, useState } from 'react'
import { Loader2, SendHorizonal, SlidersHorizontal, Sparkles } from 'lucide-react'
import Message from '../components/Message'
import RetrievalSettings from '../components/RetrievalSettings'
import Inspector from '../components/Inspector'
import { chat, errorMessage } from '../services/api'
import { useApp } from '../services/AppContext'

const SUGGESTIONS = ['Explain multiple regression in simple terms', 'Summarise the key definitions in my notes', 'What are the main differences between the methods covered?']

export default function ChatPage() {
  const { active, updateActive, retrieval, selectedDocIds, documents } = useApp()
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [detailed, setDetailed] = useState(false)
  const [panel, setPanel] = useState(null) // 'settings' | 'inspector' | null
  const [inspectQuery, setInspectQuery] = useState('')
  const endRef = useRef(null)
  const indexedCount = documents.filter((d) => d.status === 'indexed').length

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [active?.messages.length, loading])

  const send = async (text) => {
    const question = text.trim()
    if (!question || loading) return
    const history = active.messages
      .filter((m) => !m.error)
      .slice(-8)
      .map((m) => ({ role: m.role, content: m.content }))
    updateActive((c) => ({
      ...c,
      title: c.messages.length ? c.title : question.slice(0, 40),
      messages: [...c.messages, { role: 'user', content: question }],
    }))
    setInput(''); setLoading(true)
    try {
      const res = await chat({
        question, history, detailed,
        retrieval: { ...retrieval, doc_ids: selectedDocIds.length ? selectedDocIds : null },
      })
      updateActive((c) => ({ ...c, messages: [...c.messages, { role: 'assistant', question, content: res.answer, supported: res.supported, sources: res.sources, warnings: res.warnings }] }))
    } catch (e) {
      updateActive((c) => ({ ...c, messages: [...c.messages, { role: 'assistant', error: true, content: errorMessage(e) }] }))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="chat-layout">
      <section className="chat" aria-label="Conversation">
        <header className="chat__header">
          <h1>{active.title}</h1>
          <div className="chat__tools">
            <button className={`btn ${panel === 'inspector' ? 'btn--active' : ''}`} onClick={() => setPanel(panel === 'inspector' ? null : 'inspector')}>Inspector</button>
            <button className={`btn ${panel === 'settings' ? 'btn--active' : ''}`} onClick={() => setPanel(panel === 'settings' ? null : 'settings')}><SlidersHorizontal size={14} /> Settings</button>
          </div>
        </header>
        <div className="chat__scroll" aria-live="polite">
          {active.messages.length === 0 && (
            <div className="welcome">
              <Sparkles size={32} aria-hidden="true" />
              <h2>Ask your documents anything</h2>
              {indexedCount === 0
                ? <p>Upload a PDF on the Documents page first. Answers will cite the document and page.</p>
                : <p>{indexedCount} document(s) indexed. Answers are grounded in your material and cite their sources.</p>}
              <div className="suggestions">
                {SUGGESTIONS.map((s) => <button key={s} className="btn" onClick={() => send(s)} disabled={indexedCount === 0}>{s}</button>)}
              </div>
            </div>
          )}
          {active.messages.map((m, i) => (
            <Message key={i} message={m} onInspect={(q) => { setInspectQuery(q); setPanel('inspector') }} />
          ))}
          {loading && <div className="msg msg--assistant"><div className="msg__bubble msg__bubble--loading"><Loader2 className="spin" size={16} /> Retrieving and answering…</div></div>}
          <div ref={endRef} />
        </div>
        <form className="composer" onSubmit={(e) => { e.preventDefault(); send(input) }}>
          <textarea
            value={input} rows={1} maxLength={2000} placeholder="Ask a question about your documents…" aria-label="Question"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input) } }}
          />
          <label className="check composer__detail"><input type="checkbox" checked={detailed} onChange={(e) => setDetailed(e.target.checked)} /> Detailed</label>
          <button className="btn btn--primary" disabled={loading || !input.trim()} aria-label="Send"><SendHorizonal size={16} /></button>
        </form>
      </section>
      {panel && (
        <aside className="side-panel" aria-label={panel === 'settings' ? 'Retrieval settings' : 'Retrieval inspector'}>
          {panel === 'settings' ? <RetrievalSettings /> : <Inspector key={inspectQuery} initialQuery={inspectQuery} />}
        </aside>
      )}
    </div>
  )
}
