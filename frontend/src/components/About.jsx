export default function About() {
  return (
    <aside className="about" id="about">
      <div className="about-lead">
        <p>
          <b>This is not meant to be a good chess engine.</b> It is a window into how a neural network “feels” a
          position. Play it, and watch the board light up with what it is paying attention to, including everything it
          is missing.
        </p>
      </div>
      <div>
        <h2>What is this?</h2>
        <p>
          SicilianZero is a small neural network that I trained to play chess by imitation. It looked at about 5,000
          grandmaster games and 20,000 tactical puzzles and learned one thing: given a picture of the board, guess which
          move a strong player would make next. That is all it does. It never calculates and never looks a single move
          ahead. It does not even know the rules; a filter throws away its illegal guesses before they reach the board.
        </p>
        <p>
          When you move, it looks at the position once and produces a probability for every possible move. The heatmap
          is that guess drawn on the board. Bright squares are the pieces it wants to move. Hover a piece and the overlay
          switches to where it wants that piece to go. The chips below the board are its five favourite moves, and it
          always plays the first one. A real engine like Stockfish searches millions of positions before choosing. This
          is the opposite: the intuition half of an AlphaZero-style system with the search half removed, so you can see
          what pure pattern-matching looks like, mistakes included.
        </p>
        <p>
          Under the hood it is a convolutional network, the same kind used for image recognition, because a chess board
          is an 8×8 picture with 18 channels: one per piece type for each side, castling rights, the en passant square,
          and where the pieces were one move ago. It has 1.7 million parameters and four residual blocks, and it runs in
          your browser.
        </p>
      </div>

      <div>
        <h2>Why I built it</h2>
        <p>
          A learning project. I wanted hands-on experience with neural networks, computer vision and ML pipelines,
          applied to a game I care about. I built the whole thing myself: streaming games from Lichess, encoding
          positions as tensors, training on my Mac’s GPU, the inference engine with legal-move masking, and this page.
        </p>
        <p>
          I spent the first weeks trying to make it play well, and for a while I planned to bolt a search on top. Then I
          realised the interesting part was never its strength. It was watching how it thinks. A policy network that
          blunders in the open is a better exhibit than one hidden behind a search, so the weakness became the point of
          the project.
        </p>
      </div>

      <div>
        <h2>What I found</h2>
        <ol className="findings">
          <li>
            <h3>It learns goals before it learns coordinates.</h3>
            <p>
              Early on it wanted to play the Open Sicilian and tried to push the c2 pawn to d4 instead of the d2 pawn.
              Later it tried to jump a blocked knight from b8 to e5, because it wanted a knight on e5. The destination
              was right and the piece was wrong, every time. The from-square is the weak half of its intuition.
            </p>
            <span className="try">switch to To squares; the destinations are often more sensible than the choice of piece</span>
          </li>
          <li>
            <h3>It is over-socialised.</h3>
            <p>
              It was trained only on the winning side of grandmaster games, so it has never seen a hanging piece and does
              not believe in them. In a Sicilian I deliberately left my queen to be taken with 4.Nf3?? and it replied
              4...Nf6, the “normal” developing move. It wants to play a proper game so badly that it ignores a free
              queen. I called this the polite blunder.
            </p>
            <span className="try">hang a piece and watch the heatmap not even glance at it</span>
          </li>
          <li>
            <h3>Puzzles made it a glass cannon.</h3>
            <p>
              Adding 20,000 mate-in-N puzzles fixed its passive play and gave it a real attacking instinct: it now plays
              sharp openings and goes for the king. But it learned to deliver mates, not to prevent them, and it gets
              mated by simple attacks.
            </p>
            <span className="try">build a mating attack and look at the squares around its king right before the end</span>
          </li>
          <li>
            <h3>Winner-only training taught it to pose.</h3>
            <p>
              Filtering out every move by the losing side made it play ten moves of perfect Najdorf theory and then play
              a beautiful centralising knight move that lost on the spot. It learned where pieces usually go, not why.
              Style without consequences.
            </p>
          </li>
          <li>
            <h3>The value head hallucinates.</h3>
            <p>
              I added a second output that predicts who is winning. It learned to score grandmaster positions well, but
              when shown a blunder, something outside its training data, it often rated it as winning, because it never
              learned what bad looks like. That is why the value readout on this page comes with a warning and never
              influences the move.
            </p>
          </li>
          <li>
            <h3>Momentum is part of its intuition.</h3>
            <p>
              Giving the network the previous position as an extra input plane changes its answers. From a bare position
              after 1.e4 it plays e5; when it can also see that the e-pawn just moved, it plays the Sicilian c5. The
              page always feeds it the full game for that reason.
            </p>
          </li>
          <li>
            <h3>Rolling dice made it look dumber than it was.</h3>
            <p>
              My first test harness picked moves by sampling from the network’s probabilities. With around 60%
              confidence on a typical move, four moves in ten were not its first choice, and the tail of that
              distribution is garbage. Playing the top move deterministically, as this page does, removed most of the
              random blunders without touching the weights.
            </p>
          </li>
        </ol>
      </div>

      <p className="about-foot">
        Source, training code and the full day-by-day journal are on{' '}
        <a href="https://github.com/jeremiahchoi/sicilian">GitHub</a>. Built with PyTorch, python-chess, React and
        onnxruntime-web. No search, no engine, no retraining for the demo.
      </p>
    </aside>
  )
}
