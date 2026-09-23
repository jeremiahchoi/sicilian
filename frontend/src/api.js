// The model runs in the browser (see src/engine). Game state lives here too;
// we pass the full UCI move list so the "ghost layer" input (the previous
// position) is populated exactly as it was in training.
//
// The onnxruntime wasm runtime is referenced with Vite `?url` imports so the
// dev server serves it and the production build emits it as a hashed asset.
import { analyze as analyzeLocal, initEngine } from './engine/index.js'
import ortMjs from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url'
import ortWasm from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'

export const ready = initEngine({
  model: `${import.meta.env.BASE_URL}model/sicilianzero_v2.onnx`,
  wasmPaths: { mjs: ortMjs, wasm: ortWasm },
})

export async function analyze(moves) {
  await ready
  return analyzeLocal(moves)
}
