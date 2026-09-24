# SicilianZero ♟️

**Play it here: <https://jeremiahchoi.github.io/sicilian/>**

A chess neural network you can watch think. You play White against a 1.7M-parameter policy network I trained on grandmaster games and puzzles, and the board shows what it was looking at: which piece it wants to move, where it wants that piece to go, and how sure it is. There is no search of any kind. It blunders, and that is the point.

## What you're looking at

- **From squares.** A heatmap of P(from-square): how much of the model's probability wants to move each piece. Hover any square for the exact number.
- **To squares.** Hover or click a piece and the overlay switches to P(to-square | that piece). The chips under the board are its five favourite moves; it always plays the first.
- **Legal vs raw.** The network outputs one softmax over all 4096 (from, to) pairs. *Legal* renormalises over legal moves (what picks the move). *Raw* shows the unmasked distribution, illegal squares included.
- **Certainty, legal mass, value head.** Entropy of the legal distribution, the share of raw probability that lands on any legal move, and the value head's guess at the result (which it gets badly wrong off-distribution, so it never influences the move).

## What I found

The day-by-day log is in [journal.md](journal.md). The short version:

1. **It learns goals before coordinates.** Early models pushed the c2 pawn toward d4 instead of d2, or tried to jump a blocked b8 knight to e5 because they wanted a knight on e5. Right destination, wrong piece.
2. **It is over-socialised.** Trained only on the winning side of GM games, it has never seen a hanging piece. I left my queen en prise with 4.Nf3?? and it replied 4...Nf6, the "normal" move.
3. **Puzzles made it a glass cannon.** 20,000 mate-in-N puzzles fixed passive play and gave it attacking instinct, but it learned to deliver mates, not prevent them.
4. **Winner-only data taught it to pose.** Ten moves of perfect Najdorf theory, then a beautiful centralising knight move that lost on the spot.
5. **The value head hallucinates.** It scores GM positions well but rates blunders, which it never saw, as winning.
6. **Momentum is part of its intuition.** Given the previous position as an extra input plane, it answers 1.e4 with c5; from a bare FEN it plays e5.
7. **Sampling hid how good it was.** The old harness sampled moves at temperature 0.8, so four in ten were not its first choice. Argmax with the same weights removed most of the random blunders.

## Technical decisions and lessons

**Decisions**

- **Board as an image, one-hot.** Piece codes 1–6 in a grid would make a king "worth six pawns", so each piece type gets its own binary plane. Later planes add castling rights, the en passant square and the previous position's occupancy: 18×8×8 in total.
- **Side-to-move encoding.** The board is flipped so the mover is always at the bottom (`square ^ 56`), so one network plays both colours and every learned pattern is shared.
- **One 4096-way policy head, not two 64-way heads.** The first design predicted from- and to-squares separately, which discards the dependence between them. A joint softmax over (from, to) keeps it; the demo's heatmaps are its marginals and conditionals.
- **Legality at inference, not in training.** Gather logits at the legal moves and softmax over those. Illegal moves went to zero without touching the model.
- **A smaller residual net over a bigger plain one.** v1 was three conv layers into an 8192→1024 dense layer: 12.9M parameters, 8.4M in that one layer. v2 is four residual blocks with a 1×1 policy conv: 1.7M parameters, and better.
- **Multi-task loss.** Cross-entropy on the move index plus MSE on the game result, summed 1:1. Adam at 1e-3 with a scheduler over 20 epochs took policy loss from 3.37 to 1.01.

**Lessons**

- **Distribution shift explains nearly every failure.** Winner-only games contain no hanging pieces, so it cannot punish one; the value head never saw a blunder, so it rates blunders as winning. Data coverage mattered more than any architecture change.
- **Imitation learns correlation.** It knows where grandmaster pieces usually go, not why.
- **The decision rule is part of the model.** Sampling at temperature 0.8 versus argmax changed apparent strength more than another training run would have.
- **Every input feature must be reproducible at serve time.** The previous-position plane is filled by replaying the game; from a bare FEN it is silently empty and the answer changes.
- **Loss is not evaluation.** A loss of 1.01 looked great. Playing it, and later drawing its distributions, surfaced failure modes no metric showed.
- **It learned rules it was never told.** After 1.e4, 99.96% of the raw softmax lands on legal moves, although the loss never mentioned legality.

## How it works

- **Data.** 5,000+ high-Elo games streamed from Lichess (winner's moves only) plus 20,000 Lichess tactical puzzles, encoded as 18×8×8 tensors: 12 piece planes from the side to move's perspective, 4 castling planes, the en passant square, and a "ghost layer" of the previous position's occupancy (`src/utils.py`).
- **Model.** A residual CNN with four blocks and separate policy (4096-way) and value heads (`src/model.py`), trained with Adam on Apple Metal (`src/train.py`).
- **Demo.** `frontend/` is a React + Vite page. The checkpoint is exported to ONNX and runs in the browser with onnxruntime-web; `frontend/src/engine/` is a line-for-line JavaScript port of the encoding and the legal masking. `npm run parity` checks it against the PyTorch reference (`backend/app.py`) on 61 recorded positions, and CI runs that before every deploy.

## Run it locally

Node 18+:

```bash
cd frontend && npm install && npm run dev
```

Then open <http://localhost:5173>. To regenerate the ONNX file and the parity fixture after changing the weights (Python 3.10+):

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt fastapi uvicorn onnx onnxscript onnxruntime && ./venv/bin/python scripts/export_onnx.py
```

The FastAPI reference backend, if you want the PyTorch path: `./venv/bin/uvicorn backend.app:app --port 8000`.

## Layout

```
src/            model, encoding, training, data pipeline, CLI engine
models/v2/      the checkpoint the demo uses (v2_final.pth = epoch 20)
backend/        FastAPI reference implementation of the analysis
frontend/       the demo (React, onnxruntime-web); public/model holds the ONNX export
scripts/        export_onnx.py
journal.md      training log, January 2026 onward
```

Built by Jeremiah Choi. Python, PyTorch, python-chess, React, ONNX Runtime.
