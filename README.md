# SicilianZero ♟️

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
