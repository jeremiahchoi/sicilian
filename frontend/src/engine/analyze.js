// JavaScript port of backend/app.py:analyze. Same legal masking as
// src/test_model.py: gather logits at the legal move cells, softmax over
// only those. Returns the same JSON shape the FastAPI backend does.
import { encodeMove, sq } from './encode.js'

const ROUND = 6
const round = (x) => Number(x.toFixed(ROUND))

function softmax(xs) {
  let m = -Infinity
  for (const x of xs) if (x > m) m = x
  const out = new Float64Array(xs.length)
  let s = 0
  for (let i = 0; i < xs.length; i++) {
    out[i] = Math.exp(xs[i] - m)
    s += out[i]
  }
  for (let i = 0; i < xs.length; i++) out[i] /= s
  return out
}

// Relative 64x64 (flat 4096) -> absolute board coordinates. XOR 56 is its
// own inverse, so the same permutation maps both ways.
function toAbsolute(rel, black) {
  const joint = []
  for (let a = 0; a < 64; a++) {
    const ra = black ? a ^ 56 : a
    const row = new Array(64)
    for (let b = 0; b < 64; b++) {
      const rb = black ? b ^ 56 : b
      row[b] = round(rel[ra * 64 + rb])
    }
    joint.push(row)
  }
  return joint
}

function distPayload(joint) {
  const from = new Array(64).fill(0)
  const to = new Array(64).fill(0)
  for (let a = 0; a < 64; a++) {
    for (let b = 0; b < 64; b++) {
      from[a] += joint[a][b]
      to[b] += joint[a][b]
    }
  }
  return { from: from.map(round), to: to.map(round), joint }
}

function result(game) {
  if (game.isCheckmate()) return game.turn() === 'w' ? '0-1' : '1-0'
  return '1/2-1/2'
}

/**
 * @param game   chess.js instance at the position that was analysed
 * @param logits Float32Array(4096) policy logits in relative coordinates
 * @param value  number from the value head
 */
export function analyzeLogits(game, logits, value) {
  const black = game.turn() === 'b'

  // Raw distribution: softmax over all 4096 (from, to) pairs, no masking.
  const rawRel = softmax(logits)

  // Legal masking. The four promotion moves share one cell; keep the queen.
  const cellToMove = new Map()
  for (const mv of game.moves({ verbose: true })) {
    const idx = encodeMove(sq(mv.from), sq(mv.to), black)
    const prev = cellToMove.get(idx)
    if (!prev || mv.promotion === 'q') cellToMove.set(idx, mv)
  }
  const idxs = [...cellToMove.keys()].sort((a, b) => a - b)
  const legalRel = new Float64Array(4096)
  let legalMass = 0
  let entropy = 0
  let maxEntropy = 0
  if (idxs.length) {
    const p = softmax(idxs.map((i) => logits[i]))
    idxs.forEach((idx, k) => {
      legalRel[idx] = p[k]
      legalMass += rawRel[idx]
      entropy -= p[k] * Math.log2(p[k] + 1e-30)
    })
    maxEntropy = Math.log2(idxs.length)
  }

  const ranked = idxs
    .map((idx) => ({ idx, mv: cellToMove.get(idx), prob: legalRel[idx] }))
    .sort((a, b) => b.prob - a.prob)
  const moveOut = ({ mv, prob }) => ({
    uci: mv.from + mv.to + (mv.promotion ?? ''),
    san: mv.san,
    from_sq: sq(mv.from),
    to_sq: sq(mv.to),
    prob: round(prob),
  })
  const top = ranked.slice(0, 5).map(moveOut)
  const gameOver = game.isGameOver()

  return {
    fen: game.fen(),
    turn: black ? 'b' : 'w',
    game_over: gameOver,
    result: gameOver ? result(game) : null,
    ghost_layer_missing: false,
    move: top[0] ?? null,
    top_moves: top,
    value: round(value),
    n_legal: idxs.length,
    entropy_bits: Number(entropy.toFixed(4)),
    max_entropy_bits: Number(maxEntropy.toFixed(4)),
    legal_mass: round(legalMass),
    raw: distPayload(toAbsolute(rawRel, black)),
    legal: distPayload(toAbsolute(legalRel, black)),
  }
}
