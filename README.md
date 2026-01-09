# SicilianZero ♟️

**An End-to-End Deep Learning Chess Engine**

SicilianZero is a computer vision-based chess AI built from scratch using PyTorch. Unlike traditional engines that rely purely on brute-force calculation, SicilianZero uses a **Convolutional Neural Network (CNN)** to evaluate board geometry and "intuitively" predict Grandmaster-level moves.

Currently trained on 5,000+ high-Elo games from the Lichess API, the engine has successfully learned complex opening theory—specifically mastering the **Sicilian Najdorf** variation. It utilizes a **Legal Move Masking** inference engine to ensure robustness and valid play.

### 🗺️ Roadmap & To-Do

**Phase 1: The "Mimic" (Completed) ✅**
- [x] Build ETL pipeline (Lichess API $\to$ PGN $\to$ Binary Tensors)
- [x] Design 3-Layer CNN Architecture (Policy Network)
- [x] Implement Multi-Task Learning (From/To Heads)
- [x] Create CLI for human-vs-AI play
- [x] Implement Legal Move Masking (Solver-assisted Inference)

**Phase 2: The "Calculator" (Current Focus) 🚧**
- [ ] **Add Value Head:** Upgrade architecture to predict Win/Loss probability (eval bar).
- [ ] **Tactical Search:** Implement a hybrid search algorithm (e.g., MiniMax or MCTS) that uses the Neural Net to prune the tree.
- [ ] **Tactical Fine-Tuning:** Retrain on puzzle datasets to fix "Blunder Blindness" (The Grandmaster Bias).

**Phase 3: The "Product" (Future) 🚀**
- [ ] **Visual Interface:** Build a web-based GUI (React/Flask) so users can drag-and-drop pieces instead of using CLI notation.
- [ ] **Self-Play Reinforcement Learning:** Allow the bot to play against itself to discover novel strategies (AlphaZero style).
- [ ] **Dockerize:** Containerize the application for easy deployment.
