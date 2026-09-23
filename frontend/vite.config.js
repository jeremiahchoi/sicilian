import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Node 18 compatible (Vite 5). The model runs in the browser via
// onnxruntime-web, so there is no backend to proxy to. BASE_PATH is set by
// the GitHub Pages workflow (the site lives under /<repo>/).
export default defineConfig({
  plugins: [react()],
  base: process.env.BASE_PATH || '/',
  resolve: {
    // Browser: the wasm-only build, loading the runtime from public/ort
    // (copied by scripts/copy-ort.mjs) instead of bundling a 28 MB WebGPU
    // variant. Node (scripts/parity.mjs) still resolves the root package.
    alias: [{ find: /^onnxruntime-web$/, replacement: 'onnxruntime-web/wasm' }],
    conditions: ['onnxruntime-web-use-extern-wasm'],
  },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  server: { port: 5173 },
})
