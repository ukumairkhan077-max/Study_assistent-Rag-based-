import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { getHealth, listDocuments, errorMessage } from './api'

const AppContext = createContext(null)
export const useApp = () => useContext(AppContext)

export const DEFAULT_RETRIEVAL = {
  strategy: 'similarity',
  top_k: 5,
  candidate_k: 20,
  mmr_lambda: 0.5,
  num_queries: 3,
  hybrid_base: false,
  rerank: false,
  parent_document: false,
  compression: false,
  compression_threshold: 0.25,
}

const load = (key, fallback) => {
  try {
    return JSON.parse(localStorage.getItem(key)) ?? fallback
  } catch {
    return fallback
  }
}

const newConversation = () => ({ id: crypto.randomUUID(), title: 'New conversation', messages: [] })

export function AppProvider({ children }) {
  const [health, setHealth] = useState(null)
  const [documents, setDocuments] = useState([])
  const [docError, setDocError] = useState('')
  const [retrieval, setRetrieval] = useState(() => load('retrieval', DEFAULT_RETRIEVAL))
  const [selectedDocIds, setSelectedDocIds] = useState(() => load('selectedDocIds', []))
  const [conversations, setConversations] = useState(() => {
    const saved = load('conversations', [])
    return saved.length ? saved : [newConversation()]
  })
  const [activeId, setActiveId] = useState(() => load('activeId', null))
  const pollRef = useRef(null)

  // Conversations live in the browser; the backend is stateless. History sent to the
  // LLM is context only - the backend never treats it as evidence.
  useEffect(() => localStorage.setItem('retrieval', JSON.stringify(retrieval)), [retrieval])
  useEffect(() => localStorage.setItem('selectedDocIds', JSON.stringify(selectedDocIds)), [selectedDocIds])
  useEffect(() => localStorage.setItem('conversations', JSON.stringify(conversations)), [conversations])
  useEffect(() => localStorage.setItem('activeId', JSON.stringify(activeId)), [activeId])

  const active = conversations.find((c) => c.id === activeId) || conversations[0]

  const refreshDocuments = useCallback(async () => {
    try {
      const docs = await listDocuments()
      setDocuments(docs)
      setDocError('')
      // drop selections that no longer exist (e.g. deleted documents)
      setSelectedDocIds((ids) => ids.filter((id) => docs.some((d) => d.doc_id === id)))
      return docs
    } catch (e) {
      setDocError(errorMessage(e))
      return []
    }
  }, [])

  const refreshHealth = useCallback(() => {
    getHealth().then(setHealth).catch(() => setHealth(null))
  }, [])

  useEffect(() => {
    refreshDocuments()
    refreshHealth()
  }, [refreshDocuments, refreshHealth])

  // poll while anything is still being indexed
  useEffect(() => {
    clearInterval(pollRef.current)
    if (documents.some((d) => d.status === 'processing')) {
      pollRef.current = setInterval(() => {
        refreshDocuments().then(refreshHealth)
      }, 1500)
    }
    return () => clearInterval(pollRef.current)
  }, [documents, refreshDocuments, refreshHealth])

  const updateActive = useCallback(
    (fn) => setConversations((cs) => cs.map((c) => (c.id === (active?.id) ? fn(c) : c))),
    [active?.id]
  )
  const startConversation = () => {
    const c = newConversation()
    setConversations((cs) => [c, ...cs])
    setActiveId(c.id)
  }
  const deleteConversation = (id) => {
    setConversations((cs) => {
      const rest = cs.filter((c) => c.id !== id)
      return rest.length ? rest : [newConversation()]
    })
    setActiveId(null)
  }

  const value = useMemo(
    () => ({
      health, documents, docError, refreshDocuments, refreshHealth,
      retrieval, setRetrieval, selectedDocIds, setSelectedDocIds,
      conversations, active, setActiveId, updateActive, startConversation, deleteConversation,
    }),
    [health, documents, docError, retrieval, selectedDocIds, conversations, active, updateActive, refreshDocuments, refreshHealth]
  )
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}
