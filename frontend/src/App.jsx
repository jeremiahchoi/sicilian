import { useCallback, useMemo, useRef, useState } from 'react'
import { Chess } from 'chess.js'
import Board from './components/Board.jsx'
import Dock from './components/Dock.jsx'
import { CertaintyCard, HoverCard, LegendCard, StatsCard } from './components/SideCards.jsx'
import { analyze } from './api.js'
import { SQUARES } from './heat.js'

const MODEL_MOVE_DELAY_MS = 650 // let the heatmap land on the position it analysed before the piece moves

function uciHistory(game) {
  return game.history({ verbose: true }).map((m) => m.from + m.to + (m.promotion ?? ''))
}

function statusText(game, thinking, error) {
  if (error) return error
  if (game.isCheckmate()) return game.turn() === 'w' ? 'Checkmate. The model won.' : 'Checkmate. You won.'
  if (game.isStalemate()) return 'Stalemate.'
  if (game.isDraw()) return 'Draw.'
  if (thinking) return 'Model is thinking…'
  if (game.turn() === 'w') return game.isCheck() ? 'Your move. You are in check.' : 'Your move, White.'
  return 'Model to move.'
}

function Seg({ value, onChange, options }) {
  return (
    <div className="seg">
      {options.map(([v, label, title]) => (
        <button key={v} className={v === value ? 'on' : ''} onClick={() => onChange(v)} title={title}>
          {label}
        </button>
      ))}
    </div>
  )
}

export default function App() {
  const gameRef = useRef(new Chess())
  const game = gameRef.current
  const [fen, setFen] = useState(game.fen())
  const [analyses, setAnalyses] = useState([]) // [{ ply, data }]
  const [thinking, setThinking] = useState(false)
  const [error, setError] = useState(null)
  const reqId = useRef(0)

  const [showOverlay, setShowOverlay] = useState(true)
  const [source, setSource] = useState('legal') // 'legal' | 'raw'
  const [view, setView] = useState('from') // 'from' | 'to'
  const [hoverSq, setHoverSq] = useState(null)
  const [pinnedSq, setPinnedSq] = useState(null)

  const analysis = analyses.length ? analyses[analyses.length - 1].data : null
  const focusSq = pinnedSq ?? (view === 'from' ? hoverSq : null)

  // What the overlay is showing right now.
  const dist = useMemo(() => {
    if (!analysis) return null
    const d = analysis[source]
    if (focusSq != null) {
      const row = d.joint[focusSq]
      const mass = row.reduce((a, b) => a + b, 0)
      if (mass > 0) {
        return { kind: 'cond', from: focusSq, fromMass: mass, values: row.map((v) => v / mass) }
      }
    }
    return { kind: view, values: view === 'from' ? d.from : d.to }
  }, [analysis, source, view, focusSq])

  const pmax = useMemo(() => (dist ? Math.max(...dist.values) : 0), [dist])

  const readout = useMemo(() => {
    if (!dist || hoverSq == null) return null
    const name = SQUARES[hoverSq]
    const p = dist.values[hoverSq]
    if (dist.kind === 'cond') {
      const f = SQUARES[dist.from]
      if (hoverSq === dist.from) return { label: `P(from = ${f})`, p: dist.fromMass }
      return { label: `P(to = ${name} | from = ${f})`, p }
    }
    return { label: `P(${dist.kind} = ${name})`, p }
  }, [dist, hoverSq])

  const requestModel = useCallback(async () => {
    const id = ++reqId.current
    const ply = game.history().length
    setThinking(true)
    setError(null)
    try {
      const data = await analyze(uciHistory(game))
      if (id !== reqId.current) return
      setAnalyses((a) => [...a, { ply, data }])
      setPinnedSq(null)
      if (data.move && !game.isGameOver()) {
        await new Promise((r) => setTimeout(r, MODEL_MOVE_DELAY_MS))
        if (id !== reqId.current) return
        const u = data.move.uci
        game.move({ from: u.slice(0, 2), to: u.slice(2, 4), promotion: u[4] })
        setFen(game.fen())
      }
    } catch (e) {
      if (id === reqId.current) setError(e.message)
    } finally {
      if (id === reqId.current) setThinking(false)
    }
  }, [game])

  const onDrop = useCallback(
    (from, to, piece) => {
      if (thinking || game.turn() !== 'w' || game.isGameOver()) return false
      const promotion = piece?.[1] === 'P' && to[1] === '8' ? 'q' : undefined
      try {
        game.move({ from, to, promotion })
      } catch {
        return false
      }
      setFen(game.fen())
      requestModel()
      return true
    },
    [game, thinking, requestModel],
  )

  const newGame = useCallback(() => {
    reqId.current++
    game.reset()
    setFen(game.fen())
    setAnalyses([])
    setThinking(false)
    setError(null)
    setPinnedSq(null)
    setHoverSq(null)
  }, [game])

  const undo = useCallback(() => {
    if (thinking || game.history().length === 0) return
    // Take back the model's reply (if any) and your move.
    do {
      game.undo()
    } while (game.turn() !== 'w' && game.history().length > 0)
    const len = game.history().length
    setFen(game.fen())
    setAnalyses((a) => a.filter((x) => x.ply < len))
    setPinnedSq(null)
    setError(null)
  }, [game, thinking])

  const chosenArrow = useMemo(() => {
    if (!analysis?.move) return null
    // Only draw the arrow while the board still reflects that reply.
    const last = analyses[analyses.length - 1]
    if (game.history().length !== last.ply + 1) return null
    return [SQUARES[analysis.move.from_sq], SQUARES[analysis.move.to_sq]]
  }, [analysis, analyses, fen]) // eslint-disable-line react-hooks/exhaustive-deps

  const onSquareClick = useCallback(
    (sq) => {
      const idx = SQUARES.indexOf(sq)
      const hasMass = analysis ? analysis[source].from[idx] > 0 : false
      setPinnedSq((p) => (p === idx || !hasMass ? null : idx))
    },
    [analysis, source],
  )

  const gameOver = game.isGameOver()

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="wordmark">SicilianZero</span>
          <span className={`status ${error ? 'status-error' : ''}`}>{statusText(game, thinking, error)}</span>
        </div>
        <div className="toggles">
          <Seg
            value={view}
            onChange={(v) => { setView(v); setPinnedSq(null) }}
            options={[
              ['from', 'From', 'Which piece it wants to move'],
              ['to', 'To', 'Where it wants pieces to land, summed over every piece'],
            ]}
          />
          <Seg
            value={source}
            onChange={setSource}
            options={[
              ['legal', 'Legal', 'Softmax over legal moves only (this is what picks the move)'],
              ['raw', 'Raw', 'Softmax over all 4096 from–to pairs, illegal ones included'],
            ]}
          />
          <button
            className={`icon-btn ${showOverlay ? 'on' : ''}`}
            aria-label={showOverlay ? 'Hide heatmap' : 'Show heatmap'}
            aria-pressed={showOverlay}
            title={showOverlay ? 'Hide heatmap' : 'Show heatmap'}
            onClick={() => setShowOverlay((s) => !s)}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" />
              <circle cx="12" cy="12" r="3" />
              {!showOverlay && <path d="M4 4l16 16" />}
            </svg>
          </button>
          <button className="pill" onClick={undo} disabled={thinking || game.history().length === 0}>
            Take back
          </button>
          <button className="pill" onClick={newGame}>
            New game
          </button>
        </div>
      </header>

      <main className="stage">
        <aside className="side side-left">
          <LegendCard dist={dist} pmax={pmax} source={source} />
          <HoverCard readout={readout} dist={dist} pinnedSq={pinnedSq} onUnpin={() => setPinnedSq(null)} view={view} />
        </aside>

        <div className="board-col">
          <Board
            fen={fen}
            onDrop={onDrop}
            draggable={!thinking && game.turn() === 'w' && !gameOver}
            dist={showOverlay ? dist : null}
            pmax={pmax}
            labelMin={source === 'raw' ? 0.0005 : 0.005}
            focusSq={focusSq}
            pinnedSq={pinnedSq}
            arrow={chosenArrow}
            onHover={(sq) => setHoverSq(sq == null ? null : SQUARES.indexOf(sq))}
            onSquareClick={onSquareClick}
          />
        </div>

        <aside className="side side-right">
          <CertaintyCard analysis={analysis} />
          <StatsCard analysis={analysis} />
        </aside>
      </main>

      <Dock analysis={analysis} thinking={thinking} gameOver={gameOver} pinnedSq={pinnedSq} setPinnedSq={setPinnedSq} />
    </div>
  )
}
