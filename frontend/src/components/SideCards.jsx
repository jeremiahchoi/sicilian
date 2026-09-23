import { SQUARES, cssGradient, fmtP } from '../heat.js'

// Left column ------------------------------------------------------------

export function LegendCard({ dist, pmax, source }) {
  const title = !dist
    ? 'Probability'
    : dist.kind === 'cond'
      ? `P(to | from = ${SQUARES[dist.from]}), ${source}`
      : `P(${dist.kind}), ${source}`
  // Square-root ramp: tick values are not evenly spaced (see heat.js).
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => pmax * t * t)
  return (
    <section className="card">
      <div className="card-title mono">{title}</div>
      <div className="legend-bar" style={{ background: cssGradient() }} />
      <div className="legend-ticks mono">
        {ticks.map((p, i) => (
          <span key={i}>{fmtP(p)}</span>
        ))}
      </div>
      <p className="card-note">
        {source === 'raw'
          ? 'Raw = all 4096 from–to pairs, illegal ones included. Tiny values on White’s pieces are attention leaking where it cannot move.'
          : 'Legal = renormalised over legal moves only. This is what picks the move. Square-root scale so small values stay visible.'}
      </p>
    </section>
  )
}

export function HoverCard({ readout, dist, pinnedSq, onUnpin, view }) {
  return (
    <section className="card">
      <div className="card-title">Hover</div>
      {readout ? (
        <>
          <div className="mono hover-label">{readout.label}</div>
          <div className="mono hover-value">{fmtP(readout.p)}</div>
        </>
      ) : (
        <p className="card-note">
          {!dist
            ? 'Make a move to see the model think.'
            : dist.kind === 'cond'
              ? `Showing where the piece on ${SQUARES[dist.from]} wants to go.`
              : view === 'from'
                ? 'Hover a piece to see where it wants to go. Click to pin.'
                : 'Hover a square for its probability. Click a piece to condition on it.'}
        </p>
      )}
      {pinnedSq != null && (
        <button className="link" onClick={onUnpin}>
          pinned on {SQUARES[pinnedSq]} · unpin
        </button>
      )}
    </section>
  )
}

// Right column -----------------------------------------------------------

export function CertaintyCard({ analysis }) {
  const a = analysis
  const certainty = a && a.max_entropy_bits > 0 ? 1 - a.entropy_bits / a.max_entropy_bits : null
  const C = 2 * Math.PI * 50
  const dash = certainty == null ? 0 : certainty * C
  return (
    <section className="card card-center">
      <svg width="120" height="120" viewBox="0 0 120 120" role="img" aria-label="Certainty ring">
        <circle cx="60" cy="60" r="50" fill="none" stroke="#262626" strokeWidth="10" />
        {dash > 0 && (
          <circle
            cx="60" cy="60" r="50" fill="none" stroke="#f5b942" strokeWidth="10" strokeLinecap="round"
            strokeDasharray={`${dash} ${C}`} transform="rotate(-90 60 60)"
          />
        )}
        <text x="60" y="58" textAnchor="middle" fill="#f2efe8" fontFamily="'DM Mono', monospace" fontSize="24" fontWeight="500">
          {certainty == null ? '–' : `${Math.round(certainty * 100)}%`}
        </text>
        <text x="60" y="76" textAnchor="middle" fill="#a3a096" fontFamily="Manrope, sans-serif" fontSize="10" fontWeight="600">
          CERTAIN
        </text>
      </svg>
      <p className="card-note center">
        {a ? (
          <>
            {a.entropy_bits.toFixed(2)} of {a.max_entropy_bits.toFixed(2)} bits
            <br />
            over {a.n_legal} legal moves
          </>
        ) : (
          'entropy of the legal distribution'
        )}
      </p>
    </section>
  )
}

export function StatsCard({ analysis }) {
  const a = analysis
  return (
    <section className="card">
      <div className="stat-row">
        <span className="stat-key">legal mass</span>
        <span className="mono">{a ? fmtP(a.legal_mass) : '–'}</span>
      </div>
      <div className="stat-row">
        <span className="stat-key">value head</span>
        <span className={`mono ${a ? (a.value >= 0 ? 'pos' : 'neg') : ''}`}>
          {a ? `${a.value >= 0 ? '+' : '−'}${Math.abs(a.value).toFixed(2)}` : '–'}
        </span>
      </div>
      <p className="card-note">
        Legal mass is the share of the raw softmax that lands on a legal move. Value is Black’s own guess at the result, −1 to +1. It was never trained on defence, so read it with suspicion.
      </p>
      {a?.ghost_layer_missing && (
        <p className="card-note warn">Position was sent as a bare FEN, so the model could not see the previous position.</p>
      )}
    </section>
  )
}
