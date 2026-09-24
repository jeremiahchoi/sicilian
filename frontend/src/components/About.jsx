function Section({ title, open, children }) {
  return (
    <details className="sec" open={open}>
      <summary>
        <span>{title}</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M6 9l6 6 6-6" />
        </svg>
      </summary>
      <div className="sec-body">{children}</div>
    </details>
  )
}

import { ME } from '../me.js'

function AboutMe() {
  const initials = ME.name.split(' ').map((w) => w[0]).join('')
  return (
    <div className="me">
      <img className="me-avatar" src={ME.avatar} alt="" width="88" height="88" onError={(e) => { e.currentTarget.style.display = 'none'; e.currentTarget.nextSibling.style.display = 'flex' }} />
      <div className="me-initials" aria-hidden="true">{initials}</div>
      <div className="me-text">
        <div className="me-name">{ME.name}</div>
        {ME.blurb && <p className="me-blurb">{ME.blurb}</p>}
        <div className="me-links">
          {ME.linkedin && (
            <a href={ME.linkedin} target="_blank" rel="noreferrer">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M20.45 20.45h-3.55v-5.57c0-1.33-.03-3.04-1.85-3.04-1.86 0-2.14 1.45-2.14 2.94v5.67H9.35V9h3.41v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28zM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12zM7.12 20.45H3.56V9h3.56v11.45zM22.22 0H1.77C.79 0 0 .77 0 1.73v20.54C0 23.23.79 24 1.77 24h20.45c.98 0 1.78-.77 1.78-1.73V1.73C24 .77 23.2 0 22.22 0z"/></svg>
              LinkedIn
            </a>
          )}
          <a href={ME.github} target="_blank" rel="noreferrer">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .3a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1.1-.7.1-.7.1-.7 1.2.1 1.9 1.2 1.9 1.2 1.1 1.9 2.8 1.3 3.5 1 .1-.8.4-1.3.8-1.6-2.7-.3-5.5-1.3-5.5-5.9 0-1.3.5-2.4 1.2-3.2-.1-.3-.5-1.5.1-3.2 0 0 1-.3 3.3 1.2a11.5 11.5 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2.7 1.7.2 2.9.1 3.2.8.8 1.2 1.9 1.2 3.2 0 4.6-2.8 5.6-5.5 5.9.4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.6A12 12 0 0 0 12 .3"/></svg>
            GitHub
          </a>
        </div>
      </div>
    </div>
  )
}

export default function About() {
  return (
    <aside className="about" id="about">
      <AboutMe />
      <div className="about-lead">
        <p>
          <b>This is not meant to be a good chess engine.</b> It is a window into how a neural network “feels” a
          position. Play it and watch the board light up with what it is paying attention to, including everything it
          misses.
        </p>
      </div>

      <Section title="What is this?" open>
        <p>
          A small neural network I trained to play chess by imitation. It studied about 5,000 grandmaster games and
          20,000 tactical puzzles and learned one skill: look at the board and guess which move a strong player makes
          next. It never calculates, never looks a move ahead, and does not know the rules. A filter discards its illegal
          guesses.
        </p>
        <p>
          The heatmap is its guess drawn on the board. Bright squares are the pieces it wants to move; hover a piece to
          see where it wants that piece to go. The chips under the board are its five favourite moves, and it always
          plays the first one. Stockfish searches millions of positions before choosing. This is the opposite: the
          intuition half of an AlphaZero-style system with the search removed.
        </p>
      </Section>

      <Section title="Why I built it">
        <p>
          I’m Jeremiah. I wanted to learn neural networks properly, not from a tutorial, so I picked a problem I care
          about and built every layer myself: streaming games from Lichess, encoding positions as tensors, training on
          my Mac’s GPU, the legal-move masking, and this page.
        </p>
        <p>
          I spent weeks trying to make it play well and planned to bolt a search on top. Then I realised the interesting
          part was never its strength. It was watching how it thinks. A network that blunders in the open is a better
          exhibit than one hidden behind a search, so the weakness became the point.
        </p>
      </Section>

      <Section title="What I found">
        <ol className="findings">
          <li>
            <h3>It learns goals before coordinates.</h3>
            <p>
              Early on it pushed the c2 pawn toward d4 instead of the d2 pawn, and tried to jump a blocked b8 knight to
              e5 because it wanted a knight on e5. Right destination, wrong piece, every time.
            </p>
            <span className="try">switch to To squares; the destinations make more sense than the piece choice</span>
          </li>
          <li>
            <h3>It is over-socialised.</h3>
            <p>
              Trained only on the winning side of grandmaster games, it has never seen a hanging piece and does not
              believe in them. I left my queen en prise with 4.Nf3?? and it replied 4...Nf6, the “normal” move.
            </p>
            <span className="try">hang a piece and watch the heatmap not even glance at it</span>
          </li>
          <li>
            <h3>Puzzles made it a glass cannon.</h3>
            <p>
              20,000 mate-in-N puzzles cured its passive play and gave it a real attacking instinct, but it learned to
              deliver mates, not to prevent them.
            </p>
            <span className="try">build a mating attack and look at the squares around its king before the end</span>
          </li>
          <li>
            <h3>Winner-only data taught it to pose.</h3>
            <p>
              Filtering out losers’ moves got me ten moves of perfect Najdorf theory, then a beautiful centralising
              knight move that lost on the spot. Style without consequences.
            </p>
          </li>
          <li>
            <h3>The value head hallucinates.</h3>
            <p>
              A second output predicts who is winning. It scores grandmaster positions well but rates blunders, which it
              never saw, as winning. That is why its readout carries a warning and never influences the move.
            </p>
          </li>
          <li>
            <h3>Momentum is part of its intuition.</h3>
            <p>
              Showing it the previous position as an extra input plane changes its answers: from a bare position after
              1.e4 it plays e5; when it can see the e-pawn just moved, it plays the Sicilian c5.
            </p>
          </li>
          <li>
            <h3>Rolling dice made it look dumber than it is.</h3>
            <p>
              My first harness sampled moves from its probabilities, so four moves in ten were not its first choice.
              Playing the top move deterministically removed most of the random blunders. Same weights.
            </p>
          </li>
        </ol>
      </Section>

      <Section title="Technical decisions & lessons">
        <h4 className="sub">Decisions</h4>
        <ul className="tech">
          <li>
            <b>Board as an image, one-hot.</b> Piece codes 1–6 in a grid would make a king “worth six pawns” to the
            network, so each piece type gets its own binary plane. Later planes add castling rights, the en passant
            square, and the previous position’s occupancy: 18 × 8 × 8 in total.
          </li>
          <li>
            <b>Side-to-move encoding.</b> The board is flipped so the mover is always “at the bottom”, so one network
            plays both colours and every pattern it learns is shared between them.
          </li>
          <li>
            <b>One 4096-way policy, not two 64-way heads.</b> My first design predicted the from-square and the
            to-square separately, which throws away the dependence between them. A joint softmax over (from, to) keeps
            it; the heatmaps here are its marginals and conditionals.
          </li>
          <li>
            <b>Legality at inference, not in training.</b> Gather the logits at the legal moves and softmax over those.
            Illegal moves went from a constant nuisance to zero without touching the model.
          </li>
          <li>
            <b>Smaller residual net over a bigger plain one.</b> v1 was three conv layers feeding a 8192→1024 dense
            layer: 12.9M parameters, 8.4M of them in that one layer. v2 is four residual blocks with a 1×1 policy conv:
            1.7M parameters, and better.
          </li>
          <li>
            <b>Multi-task loss.</b> Cross-entropy on the move index plus mean squared error on the game result, summed
            1:1, so the shared trunk learns features useful for both. Adam at 1e-3 with a scheduler over 20 epochs took
            the policy loss from 3.37 to 1.01.
          </li>
        </ul>
        <h4 className="sub">Lessons</h4>
        <ul className="tech">
          <li>
            <b>Distribution shift explains nearly every failure.</b> Winner-only games contain no hanging pieces, so the
            net cannot punish one. The value head never saw a blunder, so it rates blunders as winning. Coverage of the
            data mattered more than any architecture change.
          </li>
          <li>
            <b>Imitation learns correlation.</b> It knows where grandmaster pieces usually go, not why. Mimicking a
            style without the consequences produces confident, well-shaped nonsense.
          </li>
          <li>
            <b>The decision rule is part of the model.</b> Sampling at temperature 0.8 versus taking the argmax changed
            the apparent strength more than an extra training run would have.
          </li>
          <li>
            <b>Every input feature must be reproducible at serve time.</b> The previous-position plane is filled by
            replaying the game. Feed a bare FEN and it is silently empty, and the net’s answer changes.
          </li>
          <li>
            <b>Loss is not evaluation.</b> A loss of 1.01 looked great. Playing it, and later drawing its distributions,
            surfaced failure modes no metric showed. The heatmap became the eval tool.
          </li>
          <li>
            <b>It learned the rules it was never told.</b> After 1.e4, 99.96% of the raw softmax lands on legal moves.
            Legality was absorbed from the data even though the training loss never mentioned it.
          </li>
        </ul>
      </Section>

      <Section title="How it works">
        <p>
          A convolutional network, the kind used for image recognition, because a chess board is an 8×8 picture with 18
          channels: one per piece type per side, castling rights, the en passant square, and where the pieces were one
          move ago. Four residual blocks, 1.7 million parameters, trained in PyTorch on Apple Metal.
        </p>
        <p>
          For this page the model is exported to ONNX and runs entirely in your browser. A test checks the browser
          version against the PyTorch original on 61 positions before every deploy.
        </p>
      </Section>

      <p className="about-foot">
        Code, training pipeline and the day-by-day journal:{' '}
        <a href="https://github.com/jeremiahchoi/sicilian">github.com/jeremiahchoi/sicilian</a>
      </p>
    </aside>
  )
}
