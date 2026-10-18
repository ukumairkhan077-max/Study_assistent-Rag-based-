import { AlertCircle, CheckCircle2, User, Bot, Search } from 'lucide-react'
import SourceCard from './SourceCard'

// Turn "[S1]" markers into clickable chips that scroll to the matching source card.
function renderAnswer(text, sources) {
  const known = new Set(sources.map((s) => s.source_id))
  return text.split(/(\[S\d+\])/g).map((part, i) => {
    const m = part.match(/^\[(S\d+)\]$/)
    if (!m) return <span key={i}>{part}</span>
    return known.has(m[1]) ? (
      <a key={i} className="chip chip--link" href={`#src-${m[1]}`} onClick={(e) => { e.preventDefault(); document.getElementById(`src-${m[1]}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }) }}>
        {m[1]}
      </a>
    ) : null
  })
}

export default function Message({ message, onInspect }) {
  if (message.role === 'user') {
    return (
      <div className="msg msg--user">
        <User size={18} aria-hidden="true" className="msg__avatar" />
        <div className="msg__bubble">{message.content}</div>
      </div>
    )
  }
  const sources = message.sources || []
  return (
    <div className="msg msg--assistant">
      <Bot size={18} aria-hidden="true" className="msg__avatar" />
      <div className="msg__main">
        {message.error ? (
          <div className="banner banner--error" role="alert"><AlertCircle size={16} /> {message.content}</div>
        ) : (
          <>
            <div className="msg__bubble">{renderAnswer(message.content, sources)}</div>
            <div className="msg__status">
              {message.supported ? (
                <span className="badge badge--ok"><CheckCircle2 size={12} /> Supported by your documents</span>
              ) : (
                <span className="badge badge--warn"><AlertCircle size={12} /> Not supported by your documents</span>
              )}
              <button className="link-btn" onClick={() => onInspect(message.question)}><Search size={12} /> Inspect retrieval</button>
            </div>
            {message.warnings?.map((w, i) => <p key={i} className="hint hint--warn">{w}</p>)}
            {sources.length > 0 && (
              <div className="sources">
                <h4>Sources ({sources.length})</h4>
                {sources.map((s) => <SourceCard key={s.chunk_id} source={s} highlight={s.cited} />)}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
