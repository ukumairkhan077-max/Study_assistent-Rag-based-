import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server runs on :5173 and calls the FastAPI backend on :8000 directly
// (see src/services/api.js). The backend allows this origin through CORS.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true },
})
