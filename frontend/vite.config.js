import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy /api calls to the FastAPI backend during dev so we avoid CORS issues
    proxy: {
      '/query':        { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/health':       { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/images':       { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
