// Checks the JavaScript port (src/engine) against the Python backend on the
// positions recorded by scripts/export_onnx.py. Run: npm run parity
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { Chess } from 'chess.js'
import { boardToPlanes } from '../src/engine/encode.js'
import { analyzeLogits } from '../src/engine/analyze.js'
import { initEngine, runModel } from '../src/engine/index.js'

const here = dirname(fileURLToPath(import.meta.url))
const fixture = JSON.parse(readFileSync(join(here, 'parity_fixture.json'), 'utf8'))
await initEngine({ model: new Uint8Array(readFileSync(join(here, '..', 'public', 'model', 'sicilianzero_v2.onnx'))) })

const maxAbs = (a, b) => a.reduce((m, x, i) => Math.max(m, Math.abs(x - b[i])), 0)
let failures = 0
let worstDist = 0
let worstValue = 0
for (const c of fixture) {
  const game = new Chess()
  for (const uci of c.moves) game.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] })
  const planes = boardToPlanes(game)
  const planeDiff = maxAbs(Array.from(planes), c.planes)
  const { logits, value } = await runModel(planes)
  const r = analyzeLogits(game, logits, value)
  const problems = []
  if (planeDiff !== 0) problems.push(`input planes differ (max ${planeDiff})`)
  if (r.fen !== c.fen) problems.push(`fen ${r.fen} != ${c.fen}`)
  if ((r.move?.uci ?? null) !== c.move) problems.push(`move ${r.move?.uci} != ${c.move}`)
  if (r.n_legal !== c.n_legal) problems.push(`n_legal ${r.n_legal} != ${c.n_legal}`)
  if (r.game_over !== c.game_over) problems.push('game_over differs')
  const d = Math.max(
    maxAbs(r.raw.from, c.raw_from), maxAbs(r.raw.to, c.raw_to),
    maxAbs(r.legal.from, c.legal_from), maxAbs(r.legal.to, c.legal_to),
  )
  worstDist = Math.max(worstDist, d)
  if (d > 1e-4) problems.push(`distribution diff ${d}`)
  worstValue = Math.max(worstValue, Math.abs(r.value - c.value))
  if (Math.abs(r.value - c.value) > 1e-3) problems.push(`value ${r.value} != ${c.value}`)
  if (Math.abs(r.entropy_bits - c.entropy_bits) > 1e-3) problems.push(`entropy ${r.entropy_bits} != ${c.entropy_bits}`)
  if (Math.abs(r.legal_mass - c.legal_mass) > 1e-4) problems.push(`legal_mass ${r.legal_mass} != ${c.legal_mass}`)
  r.top_moves.forEach((m, i) => {
    const e = c.top_moves[i]
    if (!e || m.uci !== e.uci || m.san !== e.san || Math.abs(m.prob - e.prob) > 1e-4) {
      problems.push(`top[${i}] ${m.san} ${m.prob} != ${e?.san} ${e?.prob}`)
    }
  })
  if (problems.length) {
    failures++
    console.log(`FAIL after [${c.moves.join(' ')}]\n  ${problems.join('\n  ')}`)
  }
}
console.log(`${fixture.length - failures}/${fixture.length} positions match the Python backend`)
console.log(`worst distribution diff ${worstDist.toExponential(2)}, worst value diff ${worstValue.toExponential(2)}`)
process.exit(failures ? 1 : 0)
