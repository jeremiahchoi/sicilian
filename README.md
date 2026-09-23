# SicilianZero ♟️

## 🔍 The demo: watching a policy network think

**What this is.** A visualization of a policy network's intuition. You play White in the browser. After each of your moves the network replies with its single highest-probability legal move, and the board shows the probability distributions behind that choice: which piece it wanted to move, where it wanted that piece to go, and how sure it was.

**What this is not.** A competitive engine. The network (`models/v2/v2_final.pth`, a 1.7M-parameter residual CNN) never looks a single move ahead; there is no search, minimax, or MCTS anywhere in the demo. It blunders, and it will get mated. That is the exhibit: when it is about to be checkmated, look at the heatmap and notice that none of its attention is on the squares around its own king.

**What you see**

- **From squares.** A colour overlay of P(from-square): how much of the network's probability mass wants to move the piece on each square. Hover any square for the exact number.
- **To squares, conditioned on a piece.** Hover (or click to pin) a piece and the overlay switches to P(to-square | from = that piece). Hovering a candidate in the side panel does the same.
- **Candidate moves.** The five highest-probability legal moves as bars, with the played move highlighted.
- **Legal vs raw.** The network outputs one softmax over all 4096 (from, to) pairs. *Legal* renormalises over the legal moves only, which is what picks the move. *Raw* shows the unmasked distribution, illegal squares included; it is the more honest picture of where the net is looking.
- **How sure was it.** Entropy of the legal distribution (as a certainty percentage), the share of raw probability that lands on legal moves at all, and the value head's guess at the outcome.
- **The eye button** toggles the overlay off so you can see the board plainly. The colour scale is square-root scaled so the long tail of small probabilities stays visible next to a dominant square; the legend ticks show the actual values.

**Run it.** The model runs in your browser (an ONNX export of the checkpoint via onnxruntime-web), so the demo is a static site. One command, Node 18+:

```bash
cd frontend && npm install && npm run dev
```

Then open <http://localhost:5173>. Every push to `main` also publishes it to GitHub Pages via `.github/workflows/pages.yml`; enable Pages once in the repo settings with **Source: GitHub Actions** and the site lives at `https://<owner>.github.io/<repo>/`.

**How it is wired.** `frontend/src/engine/` is a line-for-line JavaScript port of the board encoding (`src/utils.py`) and the legal masking (`src/test_model.py`), driving `frontend/public/model/sicilianzero_v2.onnx`. It takes the game's UCI move list rather than a FEN because the network's 18th input plane encodes the previous position; a bare FEN would leave that plane empty, something the model never saw in training. The from-square distribution is the row sum of the 64×64 joint, and the conditional to-square distribution is a renormalised row. `backend/app.py` is the same thing as a FastAPI service on top of the PyTorch checkpoint and remains the reference implementation: `npm run parity` checks the browser engine against it on 61 recorded positions (distributions agree to 5e-6), and CI runs that check before every deploy. To regenerate the ONNX file and the fixture after touching the weights:

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt fastapi uvicorn onnx onnxscript onnxruntime && ./venv/bin/python scripts/export_onnx.py
```

---

### An End-to-End Deep Learning Chess Engine

SicilianZero is a computer vision-based chess AI I built from scratch using PyTorch.

**Why I built this:**
This is a personal learning project, not an attempt to dethrone Stockfish. My goal is to get hands-on experience with Neural Networks, Computer Vision, and ML Pipelines by applying them to a problem I’m passionate about: Chess. I wanted to see if I could build a "brain" that plays intuitively rather than just calculating millions of variations.

Unlike traditional engines that rely heavily on brute-force calculation, SicilianZero uses a **Convolutional Neural Network (CNN)** to "look" at the board geometry and predict moves based on patterns it learned from data.

Currently, I'm training it on a hybrid dataset of **5,000+ high-Elo Grandmaster games** and **20,000+ tactical puzzles**. This "Mixed Training" approach has taught the engine to play sharp theoretical openings (specifically the **Sicilian Najdorf**) while developing the "killer instinct" needed to find checkmates.

---

## ✅ Current Features

* **Deep Learning Pipeline:** End-to-end ETL system that streams matches from the Lichess API, parses PGNs, and converts board states into 12-channel binary tensors.
* **CNN Architecture:** A custom 3-Layer Policy Network (PyTorch) with separate heads for predicting "From" and "To" squares.
* **"Winner-Only" Learning:** The training pipeline strictly filters for moves made by the winning side of Grandmaster games, preventing the AI from learning losing patterns.
* **Tactical Bootcamp:** Integrated a dataset of 20,000 Lichess puzzles (Mates in 1-5, Forks, Pins) to fix the AI's passive play style.
* **Inference Engine:** A CLI-based interface for human-vs-AI play, featuring a Legal Move Masker to ensure the Neural Net never attempts invalid moves.
* **Mac Optimization:** Full support for Apple Metal (MPS) acceleration for faster training on M-series chips.

## 🚧 To-Do List

* **Defensive Awareness:** The bot is currently a "Glass Cannon"—it attacks well but struggles to see when *it* is about to be mated. I need to implement "Anti-Blunder" heuristics.
* **Hybrid Search:** Implement a lightweight search algorithm (like MiniMax or MCTS) that uses the CNN to prune the tree, adding "calculation" to its "intuition."
* **Web Interface:** Build a simple web-based GUI (React/Flask) so I can drag-and-drop pieces instead of typing coordinates.
* **Self-Play RL:** Allow the bot to play against itself to discover novel strategies (AlphaZero style).
* **Docker Support:** Containerize the application for easier deployment and reproducibility.

---

## 🛠️ Tech Stack
* **Core:** Python 3.10+, PyTorch
* **Data:** Lichess API, `python-chess`, NumPy, Pandas
* **Hardware Acceleration:** Apple MPS (Metal Performance Shaders)

## 🚀 Getting Started
Todo
