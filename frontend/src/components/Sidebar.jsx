import { NavLink, useNavigate } from 'react-router-dom'
import { BookOpen, FileText, FlaskConical, GraduationCap, MessageSquare, Plus, Trash2 } from 'lucide-react'
import { useApp } from '../services/AppContext'

const nav = [
  { to: '/', label: 'Chat', icon: MessageSquare },
  { to: '/documents', label: 'Documents', icon: FileText },
  { to: '/study', label: 'Study tools', icon: GraduationCap },
  { to: '/evaluation', label: 'Evaluation', icon: FlaskConical },
]

export default function Sidebar({ open, onClose }) {
  const { conversations, active, setActiveId, startConversation, deleteConversation, health } = useApp()
  const navigate = useNavigate()
  return (
    <aside className={`sidebar ${open ? 'sidebar--open' : ''}`} aria-label="Primary">
      <div className="brand">
        <BookOpen size={22} aria-hidden="true" />
        <span>Study Assistant</span>
      </div>
      <nav className="nav">
        {nav.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} end={to === '/'} onClick={onClose} className={({ isActive }) => `nav__item ${isActive ? 'is-active' : ''}`}>
            <Icon size={18} aria-hidden="true" /> {label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar__section">
        <div className="sidebar__heading">
          <span>Conversations</span>
          <button className="icon-btn icon-btn--light" onClick={() => { startConversation(); navigate('/'); onClose() }} aria-label="New conversation" title="New conversation">
            <Plus size={16} />
          </button>
        </div>
        <ul className="convo-list">
          {conversations.map((c) => (
            <li key={c.id} className={`convo ${c.id === active?.id ? 'is-active' : ''}`}>
              <button className="convo__title" onClick={() => { setActiveId(c.id); navigate('/'); onClose() }}>
                {c.title}
              </button>
              <button className="icon-btn icon-btn--light" aria-label={`Delete ${c.title}`} onClick={() => deleteConversation(c.id)}>
                <Trash2 size={14} />
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="sidebar__footer">
        <span className={`dot ${health ? 'dot--ok' : 'dot--bad'}`} aria-hidden="true" />
        {health ? (health.llm_configured ? `LLM: ${health.llm_model}` : 'LLM not configured (retrieval only)') : 'Backend offline'}
      </div>
    </aside>
  )
}
