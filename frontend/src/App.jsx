import { useState } from 'react'
import { Route, Routes } from 'react-router-dom'
import { Menu } from 'lucide-react'
import Sidebar from './components/Sidebar'
import ChatPage from './pages/ChatPage'
import DocumentsPage from './pages/DocumentsPage'
import StudyPage from './pages/StudyPage'
import EvaluationPage from './pages/EvaluationPage'

export default function App() {
  const [open, setOpen] = useState(false)
  return (
    <div className="app">
      <Sidebar open={open} onClose={() => setOpen(false)} />
      {open && <div className="scrim" onClick={() => setOpen(false)} />}
      <main className="main">
        <button className="menu-btn icon-btn" onClick={() => setOpen(true)} aria-label="Open menu"><Menu size={20} /></button>
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/documents" element={<DocumentsPage />} />
          <Route path="/study" element={<StudyPage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
        </Routes>
      </main>
    </div>
  )
}
