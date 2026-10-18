import axios from 'axios'

// All backend access goes through this one module. The API key never reaches the
// browser: the backend holds it and only returns safe diagnostics.
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
  timeout: 120000,
})

export function errorMessage(err) {
  if (err.code === 'ERR_NETWORK') return 'Cannot reach the backend. Is it running on http://localhost:8000?'
  const d = err.response?.data?.detail
  if (Array.isArray(d)) return d.map((x) => x.msg).join('; ')
  return d || err.message || 'Unexpected error'
}

export const getHealth = () => api.get('/health').then((r) => r.data)
export const listDocuments = () => api.get('/documents').then((r) => r.data)
export const uploadDocument = (file, collection, chunkStrategy) => {
  const form = new FormData()
  form.append('file', file)
  form.append('collection', collection || 'General')
  if (chunkStrategy) form.append('chunk_strategy', chunkStrategy)
  return api.post('/documents', form).then((r) => r.data)
}
export const deleteDocument = (id) => api.delete(`/documents/${id}`)
export const getChunks = (id) => api.get(`/documents/${id}/chunks`).then((r) => r.data)

export const chat = (body) => api.post('/chat', body).then((r) => r.data)
export const retrieve = (body) => api.post('/retrieve', body).then((r) => r.data)

export const study = (kind, body) => api.post(`/study/${kind}`, body).then((r) => r.data)

export const getEvaluation = () => api.get('/evaluation/results').then((r) => r.data)
export const runEvaluation = () => api.post('/evaluation/run').then((r) => r.data)
export const getEvaluationStatus = () => api.get('/evaluation/status').then((r) => r.data)
