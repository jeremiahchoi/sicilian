// In-browser inference: the ONNX export of models/v2/v2_final.pth run with
// onnxruntime-web. No server involved; this is what the GitHub Pages build
// uses. backend/app.py remains the reference implementation.
import * as ort from 'onnxruntime-web'
import { Chess } from 'chess.js'
import { boardToPlanes } from './encode.js'
import { analyzeLogits } from './analyze.js'

let sessionPromise = null

/**
 * @param model    URL (browser) or Uint8Array (node) of the ONNX file
 * @param wasmPaths directory URL the ort-wasm-simd-threaded.{wasm,mjs} files are served from
 */
export function initEngine({ model, wasmPaths }) {
  if (wasmPaths) ort.env.wasm.wasmPaths = wasmPaths
  ort.env.wasm.numThreads = 1 // GitHub Pages is not cross-origin isolated
  sessionPromise = ort.InferenceSession.create(model, { executionProviders: ['wasm'] })
  return sessionPromise
}

export async function runModel(planes) {
  if (!sessionPromise) throw new Error('initEngine() has not been called')
  const session = await sessionPromise
  const out = await session.run({ board: new ort.Tensor('float32', planes, [1, 18, 8, 8]) })
  return { logits: out.policy_logits.data, value: out.value.data[0] }
}

/** Same contract as POST /api/move: UCI move list from the start position. */
export async function analyze(moves) {
  const game = new Chess()
  for (const [i, uci] of moves.entries()) {
    const mv = game.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] })
    if (!mv) throw new Error(`moves[${i}] ${uci} is illegal`)
  }
  const planes = boardToPlanes(game)
  const { logits, value } = await runModel(planes)
  return analyzeLogits(game, logits, value)
}
