import { useRef, useState } from 'react'
import { Loader2, UploadCloud } from 'lucide-react'
import { uploadDocument, errorMessage } from '../services/api'
import { useApp } from '../services/AppContext'
import ErrorBanner from './ErrorBanner'

export default function UploadPanel() {
  const { refreshDocuments, refreshHealth } = useApp()
  const inputRef = useRef(null)
  const [drag, setDrag] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [collection, setCollection] = useState('General')
  const [strategy, setStrategy] = useState('recursive')

  const send = async (files) => {
    setError(''); setNotice(''); setBusy(true)
    for (const file of files) {
      try {
        const res = await uploadDocument(file, collection, strategy)
        if (res.duplicate) setNotice(`"${file.name}" is already in your library - not indexed again.`)
      } catch (e) {
        setError(`${file.name}: ${errorMessage(e)}`)
      }
    }
    setBusy(false)
    await refreshDocuments(); refreshHealth()
  }

  return (
    <section className="card">
      <h2>Upload documents</h2>
      <div
        className={`dropzone ${drag ? 'dropzone--active' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); send([...e.dataTransfer.files]) }}
        role="button" tabIndex={0} aria-label="Upload PDF files"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
      >
        {busy ? <Loader2 className="spin" size={28} /> : <UploadCloud size={28} />}
        <p>{busy ? 'Uploading…' : 'Drop PDF files here or click to browse'}</p>
        <small>Text-based PDFs only (scanned PDFs need OCR first). Max 25 MB.</small>
        <input ref={inputRef} type="file" accept="application/pdf,.pdf" multiple hidden onChange={(e) => { send([...e.target.files]); e.target.value = '' }} />
      </div>
      <div className="row">
        <label className="field field--inline"><span>Collection</span>
          <input value={collection} maxLength={60} onChange={(e) => setCollection(e.target.value)} />
        </label>
        <label className="field field--inline"><span>Chunking</span>
          <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
            <option value="recursive">Recursive (default)</option>
            <option value="fixed">Fixed-size (baseline)</option>
            <option value="semantic">Semantic (slower)</option>
          </select>
        </label>
      </div>
      <ErrorBanner message={error} onClose={() => setError('')} />
      {notice && <p className="hint">{notice}</p>}
    </section>
  )
}
