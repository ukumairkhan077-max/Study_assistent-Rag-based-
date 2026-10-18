import { useState } from 'react'
import { ChevronDown, ChevronRight, FileText } from 'lucide-react'

// One retrieved passage. Scores are shown for debugging; a cosine similarity is a
// relative ranking signal, NOT a probability that the passage answers the question.
export default function SourceCard({ source, highlight = false, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div id={`src-${source.source_id}`} className={`source ${highlight ? 'source--cited' : ''}`}>
      <button className="source__head" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        <span className="chip">{source.source_id}</span>
        <FileText size={14} aria-hidden="true" />
        <span className="source__file">{source.filename}</span>
        <span className="source__page">p. {source.page}</span>
        {source.cited && <span className="badge badge--ok">cited</span>}
      </button>
      {open && (
        <div className="source__body">
          <p className="source__excerpt">{source.excerpt}</p>
          <dl className="source__meta">
            {source.similarity != null && (<><dt>cosine similarity</dt><dd>{source.similarity}</dd></>)}
            {source.rerank_score != null && (<><dt>rerank score</dt><dd>{source.rerank_score}</dd></>)}
            <dt>stage score</dt><dd>{source.score}</dd>
            {source.parent_expanded && (<><dt>context</dt><dd>parent passage</dd></>)}
            {source.compressed && (<><dt>context</dt><dd>compressed</dd></>)}
            <dt>chunk id</dt><dd className="mono">{source.chunk_id}</dd>
          </dl>
        </div>
      )}
    </div>
  )
}
