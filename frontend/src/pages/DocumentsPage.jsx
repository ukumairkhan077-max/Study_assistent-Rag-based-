import { useState } from 'react'
import { AlertCircle, CheckCircle2, Eye, FileText, Loader2, Trash2, X } from 'lucide-react'
import UploadPanel from '../components/UploadPanel'
import ErrorBanner from '../components/ErrorBanner'
import { deleteDocument, getChunks, errorMessage } from '../services/api'
import { useApp } from '../services/AppContext'

function StatusBadge({ doc }) {
  if (doc.status === 'processing') return <span className="badge badge--info"><Loader2 size={12} className="spin" /> Indexing…</span>
  if (doc.status === 'failed') return <span className="badge badge--bad"><AlertCircle size={12} /> Failed</span>
  return <span className="badge badge--ok"><CheckCircle2 size={12} /> Indexed</span>
}

export default function DocumentsPage() {
  const { documents, docError, refreshDocuments, refreshHealth } = useApp()
  const [error, setError] = useState('')
  const [viewing, setViewing] = useState(null)
  const [chunks, setChunks] = useState([])

  const remove = async (doc) => {
    if (!window.confirm(`Delete "${doc.filename}" and all its indexed content?`)) return
    try {
      await deleteDocument(doc.doc_id)
      await refreshDocuments(); refreshHealth()
    } catch (e) { setError(errorMessage(e)) }
  }
  const view = async (doc) => {
    try { setChunks(await getChunks(doc.doc_id)); setViewing(doc) } catch (e) { setError(errorMessage(e)) }
  }

  return (
    <div className="page">
      <h1>Documents</h1>
      <UploadPanel />
      <ErrorBanner message={error || docError} onClose={() => setError('')} />
      <section className="card">
        <h2>Library ({documents.length})</h2>
        {documents.length === 0 ? (
          <p className="empty">No documents yet. Upload a lecture PDF to get started.</p>
        ) : (
          <table className="table">
            <thead><tr><th>Document</th><th>Collection</th><th>Status</th><th>Pages</th><th>Chunks</th><th>Chunking</th><th><span className="sr-only">Actions</span></th></tr></thead>
            <tbody>
              {documents.map((d) => (
                <tr key={d.doc_id}>
                  <td><FileText size={14} aria-hidden="true" /> {d.filename}{d.error && <div className="hint hint--warn">{d.error}</div>}</td>
                  <td>{d.collection}</td>
                  <td><StatusBadge doc={d} /></td>
                  <td>{d.pages || '-'}</td>
                  <td>{d.chunk_count || '-'}{d.parent_count ? <small> ({d.parent_count} parents)</small> : null}</td>
                  <td>{d.chunk_strategy}</td>
                  <td className="actions">
                    {d.status === 'indexed' && <button className="icon-btn" onClick={() => view(d)} aria-label={`Inspect chunks of ${d.filename}`} title="Inspect chunks"><Eye size={16} /></button>}
                    <button className="icon-btn icon-btn--danger" onClick={() => remove(d)} aria-label={`Delete ${d.filename}`} title="Delete"><Trash2 size={16} /></button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
      {viewing && (
        <div className="modal" role="dialog" aria-modal="true" aria-label={`Chunks of ${viewing.filename}`}>
          <div className="modal__panel">
            <div className="modal__head">
              <h2>{viewing.filename} - {chunks.length} chunks</h2>
              <button className="icon-btn" onClick={() => setViewing(null)} aria-label="Close"><X size={18} /></button>
            </div>
            <div className="modal__body">
              {chunks.map((c) => (
                <div key={c.chunk_id} className="chunk"><div className="chunk__meta">page {c.page} · <span className="mono">{c.chunk_id}</span></div><p>{c.text}</p></div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
