import { RotateCcw } from 'lucide-react'
import { DEFAULT_RETRIEVAL, useApp } from '../services/AppContext'

const STRATEGIES = [
  ['similarity', 'Similarity (baseline)', 'Top-k by cosine similarity.'],
  ['mmr', 'MMR', 'Relevant but diverse: fewer near-duplicate passages.'],
  ['hybrid', 'Hybrid (dense + BM25)', 'Adds exact-keyword matching; fused with RRF.'],
  ['expansion', 'Query expansion', 'Adds LLM keywords to the query. Needs the LLM.'],
  ['multi_query', 'Multi-query', 'Several rephrasings, fused with RRF. Needs the LLM.'],
]

export default function RetrievalSettings() {
  const { retrieval: r, setRetrieval, documents, selectedDocIds, setSelectedDocIds, health } = useApp()
  const set = (patch) => setRetrieval({ ...r, ...patch })
  const needsLlm = r.strategy === 'expansion' || r.strategy === 'multi_query'
  const indexed = documents.filter((d) => d.status === 'indexed')
  const toggleDoc = (id) =>
    setSelectedDocIds(selectedDocIds.includes(id) ? selectedDocIds.filter((x) => x !== id) : [...selectedDocIds, id])

  return (
    <div className="settings">
      <div className="settings__title">
        <h3>Retrieval settings</h3>
        <button className="icon-btn" onClick={() => setRetrieval(DEFAULT_RETRIEVAL)} aria-label="Reset settings" title="Reset"><RotateCcw size={14} /></button>
      </div>

      <label className="field">
        <span>Strategy</span>
        <select value={r.strategy} onChange={(e) => set({ strategy: e.target.value })}>
          {STRATEGIES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
        <small>{STRATEGIES.find(([v]) => v === r.strategy)[2]}</small>
      </label>
      {needsLlm && health && !health.llm_configured && (
        <p className="hint hint--warn">LLM not configured: this strategy will fall back to the original query.</p>
      )}

      <label className="field">
        <span>Top-k: {r.top_k}</span>
        <input type="range" min="1" max="15" value={r.top_k} onChange={(e) => set({ top_k: +e.target.value })} />
      </label>
      {(r.strategy !== 'similarity' || r.rerank) && (
        <label className="field">
          <span>Candidate pool: {r.candidate_k}</span>
          <input type="range" min="5" max="60" value={r.candidate_k} onChange={(e) => set({ candidate_k: +e.target.value })} />
        </label>
      )}
      {r.strategy === 'mmr' && (
        <label className="field">
          <span>MMR lambda: {r.mmr_lambda} <small>(1 = relevance, 0 = diversity)</small></span>
          <input type="range" min="0" max="1" step="0.05" value={r.mmr_lambda} onChange={(e) => set({ mmr_lambda: +e.target.value })} />
        </label>
      )}
      {needsLlm && (
        <>
          <label className="field">
            <span>Generated queries: {r.num_queries}</span>
            <input type="range" min="1" max="6" value={r.num_queries} onChange={(e) => set({ num_queries: +e.target.value })} />
          </label>
          <label className="check"><input type="checkbox" checked={r.hybrid_base} onChange={(e) => set({ hybrid_base: e.target.checked })} /> Use hybrid search for each query</label>
        </>
      )}

      <fieldset className="stages">
        <legend>Pipeline stages</legend>
        <label className="check"><input type="checkbox" checked={r.rerank} onChange={(e) => set({ rerank: e.target.checked })} /> Cross-encoder reranking <small>(downloads a model on first use; slower)</small></label>
        <label className="check"><input type="checkbox" checked={r.parent_document} onChange={(e) => set({ parent_document: e.target.checked })} /> Parent-document context</label>
        <label className="check"><input type="checkbox" checked={r.compression} onChange={(e) => set({ compression: e.target.checked })} /> Contextual compression</label>
        {r.compression && (
          <label className="field">
            <span>Keep sentences with similarity ≥ {r.compression_threshold}</span>
            <input type="range" min="0.05" max="0.7" step="0.05" value={r.compression_threshold} onChange={(e) => set({ compression_threshold: +e.target.value })} />
            <small>Aggressive compression can remove evidence.</small>
          </label>
        )}
      </fieldset>

      <fieldset className="stages">
        <legend>Search in</legend>
        {indexed.length === 0 && <p className="hint">No indexed documents yet.</p>}
        {indexed.map((d) => (
          <label key={d.doc_id} className="check">
            <input type="checkbox" checked={selectedDocIds.includes(d.doc_id)} onChange={() => toggleDoc(d.doc_id)} /> {d.filename}
          </label>
        ))}
        {indexed.length > 0 && <small>{selectedDocIds.length === 0 ? 'All documents are searched.' : `${selectedDocIds.length} selected.`}</small>}
      </fieldset>
    </div>
  )
}
