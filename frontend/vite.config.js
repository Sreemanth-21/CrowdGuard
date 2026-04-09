import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // WebSocket proxying via Vite causes ECONNABORTED on long-lived connections.
      // The frontend connects directly to ws://localhost:8000/ws instead.
    },
  },
})
