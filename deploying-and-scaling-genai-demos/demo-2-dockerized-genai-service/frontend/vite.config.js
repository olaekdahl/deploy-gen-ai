import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // Build output goes to ../static so FastAPI can serve it
  build: {
    outDir: '../static',
    emptyOutDir: true,
  },
  server: {
    // Proxy API calls to the FastAPI backend during development
    proxy: {
      '/generate': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
});
