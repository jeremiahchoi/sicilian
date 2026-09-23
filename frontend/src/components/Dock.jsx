import { fmtP } from '../heat.js'

export default function Dock({ analysis, thinking, gameOver, pinnedSq, setPinnedSq }) {
  const a = analysis
  const chosen = a?.move?.uci
  const barMax = a?.top_moves?.length ? a.top_moves[0].prob : 1
  const headline = thinking ? 'Thinking' : a?.move ? 'Played' : gameOver ? 'Game over' : 'Waiting'
  const big = thinking ? '…' : a?.move ? a.move.san : '—'

  return (
    <footer className="dock">
      <div className="dock-played">
        <div className="card-title">{headline}</div>
        <div className="mono dock-move">{big}</div>
      </div>
      {a && a.top_moves.length === 0 ? (
        <div className="chip chip-empty">No legal moves. The raw heatmap still shows where it was looking.</div>
      ) : (
        (a?.top_moves ?? [null, null, null, null, null]).map((m, i) => (
          <button
            key={m ? m.uci : i}
            className={`chip ${m && m.uci === chosen ? 'chip-chosen' : ''} ${m ? '' : 'chip-empty'}`}
            disabled={!m}
            onMouseEnter={() => m && setPinnedSq(m.from_sq)}
            onMouseLeave={() => m && setPinnedSq((p) => (p === m.from_sq ? null : p))}
            onClick={() => m && setPinnedSq((p) => (p === m.from_sq ? null : m.from_sq))}
            title={m ? `${m.uci} — hover to see this piece's to-squares` : ''}
          >
            <span className="chip-row mono">
              <span className="chip-san">{m ? m.san : '·'}</span>
              <span className="chip-pct">{m ? fmtP(m.prob) : ''}</span>
            </span>
            <span className="chip-bar">
              <span className="chip-fill" style={{ width: m ? `${(m.prob / barMax) * 100}%` : 0 }} />
            </span>
          </button>
        ))
      )}
    </footer>
  )
}
