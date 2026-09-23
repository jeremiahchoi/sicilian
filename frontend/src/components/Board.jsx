import { createContext, forwardRef, useContext, useEffect, useState } from 'react'
import { Chessboard } from 'react-chessboard'
import { SQUARES, fmtP, glow, ramp, rampColor, sqIndex } from '../heat.js'

// The overlay is drawn *inside* each square (above the square colour, below
// the piece) so the light/dark pattern stays visible under the tint.
const HeatContext = createContext({ values: null, pmax: 0, labelMin: 1 })

const HeatSquare = forwardRef(function HeatSquare({ children, square, style }, ref) {
  const { values, pmax, labelMin } = useContext(HeatContext)
  const p = values ? values[sqIndex(square)] : 0
  const t = ramp(p, pmax)
  return (
    <div ref={ref} style={{ ...style, position: 'relative' }}>
      {t > 0 && <div className="heat" style={{ background: rampColor(t), boxShadow: glow(t) }} />}
      {children}
      {p >= labelMin && <span className={`heat-label ${t > 0.7 ? 'heat-label-dark' : ''}`}>{fmtP(p)}</span>}
    </div>
  )
})

function useBoardWidth() {
  const calc = () => {
    const w = window.innerWidth
    const h = window.innerHeight
    if (w >= 1100) return Math.max(320, Math.min(576, w - 540, h - 240))
    return Math.max(280, Math.min(576, w - 32))
  }
  const [width, setWidth] = useState(calc)
  useEffect(() => {
    const onResize = () => setWidth(calc())
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])
  return width
}

export default function Board({ fen, onDrop, draggable, dist, pmax, labelMin, focusSq, pinnedSq, arrow, onHover, onSquareClick }) {
  const width = useBoardWidth()

  const squareStyles = {}
  if (focusSq != null) {
    const pinned = focusSq === pinnedSq
    squareStyles[SQUARES[focusSq]] = {
      boxShadow: pinned ? 'inset 0 0 0 3px #f5b942' : 'inset 0 0 0 2px rgba(245, 185, 66, 0.55)',
    }
  }

  return (
    <HeatContext.Provider value={{ values: dist ? dist.values : null, pmax, labelMin }}>
      <div className="board-wrap" style={{ width }} onMouseLeave={() => onHover(null)}>
        <Chessboard
          id="sicilian"
          position={fen}
          boardWidth={width}
          boardOrientation="white"
          arePiecesDraggable={draggable}
          isDraggablePiece={({ piece }) => piece[0] === 'w'}
          onPieceDrop={onDrop}
          autoPromoteToQueen
          animationDuration={200}
          customSquare={HeatSquare}
          customSquareStyles={squareStyles}
          customArrows={arrow ? [[arrow[0], arrow[1], '#f5b942']] : []}
          customArrowColor="#f5b942"
          customLightSquareStyle={{ backgroundColor: '#4a4a4a' }}
          customDarkSquareStyle={{ backgroundColor: '#303030' }}
          customBoardStyle={{ borderRadius: 10, boxShadow: '0 0 0 1px #2a2a2a, 0 40px 80px rgba(0,0,0,0.7)' }}
          customNotationStyle={{ color: '#a3a096', fontFamily: "'DM Mono', monospace", fontSize: 10 }}
          onMouseOverSquare={(sq) => onHover(sq)}
          onMouseOutSquare={() => onHover(null)}
          onSquareClick={onSquareClick}
        />
      </div>
    </HeatContext.Provider>
  )
}
