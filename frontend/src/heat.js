// Single-hue amber heat scale for the dark board. Zero probability draws
// nothing; strong squares get an inner glow toward pale cream.
const AMBER = [245, 185, 66]
const CREAM = [255, 241, 201]

export const SQUARES = []
for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) SQUARES.push('abcdefgh'[f] + (r + 1))
export const sqIndex = (name) => 'abcdefgh'.indexOf(name[0]) + (Number(name[1]) - 1) * 8

// Position on the ramp, 0..1. Square-root so that the long tail of small
// probabilities is still visible next to a dominant square.
export function ramp(p, pmax) {
  if (!pmax || p <= 0) return 0
  return Math.sqrt(Math.min(p / pmax, 1))
}

export function rampColor(t) {
  if (t <= 0) return 'transparent'
  const mix = Math.max(0, (t - 0.8) / 0.2)
  const c = AMBER.map((v, i) => Math.round(v + (CREAM[i] - v) * mix))
  // Near-zero values fade out entirely instead of washing the whole board.
  const alpha = t < 0.04 ? 2 * t : Math.min(0.9, 0.08 + 0.82 * t)
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${alpha.toFixed(3)})`
}

export function glow(t) {
  if (t <= 0.5) return 'none'
  return `inset 0 0 ${Math.round(30 * t)}px rgba(255, 241, 201, ${(0.5 * t).toFixed(2)})`
}

export function cssGradient() {
  return 'linear-gradient(to right, rgba(245,185,66,0) 0%, rgba(245,185,66,0.35) 20%, rgba(245,185,66,0.7) 55%, #f5b942 85%, #fff1c9 100%)'
}

export function fmtP(p) {
  if (p == null || Number.isNaN(p)) return '–'
  if (p === 0) return '0'
  if (p >= 0.1) return `${(p * 100).toFixed(0)}%`
  if (p >= 0.01) return `${(p * 100).toFixed(1)}%`
  if (p >= 0.001) return `${(p * 100).toFixed(2)}%`
  if (p >= 0.0001) return `${(p * 100).toFixed(3)}%`
  return '<0.01%'
}
