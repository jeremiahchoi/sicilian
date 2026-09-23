// JavaScript port of src/utils.py: board_to_matrix and encode_move.
// Kept line-for-line faithful to the Python; frontend/scripts/parity.mjs
// checks it against the Python backend's output on recorded positions.
//
// Square indices are python-chess order: a1 = 0, b1 = 1, ... h8 = 63.
// The network sees the board from the side to move: when Black is to move,
// every square index is XOR 56 (rank flip), exactly as utils.orient does.

export const sq = (name) => 'abcdefgh'.indexOf(name[0]) + (Number(name[1]) - 1) * 8
const PIECE_IDX = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 }

/**
 * 18 x 8 x 8 input planes, flattened row-major (plane, rank, file), as a
 * Float32Array. `game` is a chess.js instance whose move history reproduces
 * the position (the ghost layer, plane 17, is the occupancy before the last
 * move, read by undoing it).
 */
export function boardToPlanes(game) {
  const planes = new Float32Array(18 * 64)
  const turn = game.turn()
  const black = turn === 'b'
  const orient = (s) => (black ? s ^ 56 : s)

  // Planes 0-11: my pieces then enemy pieces, P N B R Q K.
  const rows = game.board() // rank 8 first
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const pc = rows[r][f]
      if (!pc) continue
      const s = f + (7 - r) * 8
      const idx = (pc.color === turn ? 0 : 6) + PIECE_IDX[pc.type]
      planes[idx * 64 + orient(s)] = 1
    }
  }

  // Planes 12-15: castling rights, my K, my Q, enemy K, enemy Q.
  const mine = game.getCastlingRights(turn)
  const theirs = game.getCastlingRights(black ? 'w' : 'b')
  if (mine.k) planes.fill(1, 12 * 64, 13 * 64)
  if (mine.q) planes.fill(1, 13 * 64, 14 * 64)
  if (theirs.k) planes.fill(1, 14 * 64, 15 * 64)
  if (theirs.q) planes.fill(1, 15 * 64, 16 * 64)

  // Plane 16: en passant target. python-chess sets ep_square after any
  // double pawn push, whether or not a capture is possible.
  const hist = game.history({ verbose: true })
  const last = hist[hist.length - 1]
  if (last && last.flags.includes('b')) {
    const midRank = (Number(last.from[1]) + Number(last.to[1])) / 2
    planes[16 * 64 + orient(sq(last.from[0] + midRank))] = 1
  }

  // Plane 17: ghost layer, occupancy one move ago.
  if (last) {
    game.undo()
    const prev = game.board()
    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        if (prev[r][f]) planes[17 * 64 + orient(f + (7 - r) * 8)] = 1
      }
    }
    game.move({ from: last.from, to: last.to, promotion: last.promotion })
  }

  return planes
}

/** Index 0..4095 of a move in side-to-move-relative coordinates. */
export function encodeMove(from, to, black) {
  const f = black ? from ^ 56 : from
  const t = black ? to ^ 56 : to
  return f * 64 + t
}
