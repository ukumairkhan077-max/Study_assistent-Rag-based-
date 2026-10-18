import { AlertTriangle, X } from 'lucide-react'

export default function ErrorBanner({ message, onClose }) {
  if (!message) return null
  return (
    <div className="banner banner--error" role="alert">
      <AlertTriangle size={16} aria-hidden="true" />
      <span>{message}</span>
      {onClose && (
        <button className="icon-btn" onClick={onClose} aria-label="Dismiss"><X size={14} /></button>
      )}
    </div>
  )
}
