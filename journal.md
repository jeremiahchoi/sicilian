# Project: Behavioral Cloning Chess Engine

## Journal #11

**Date**: Jan 13, 2026 (11:20pm)

**Focus**: Architecture Overhaul (ResNet) & The Hybrid Engine

**The Strategy: Going Deeper**
- After implementing the Value head, I realized the standard CNN was hitting a "learning ceiling." It struggled to connect complex board states to long-term outcomes.
- I upgraded the model architecture to a ResNet (Residual Network). By adding "skip connections," the model can now be trained to be much deeper (4 blocks) without the gradient signal vanishing. This is the same architectural shift that allowed AlphaZero to dominate.
- I also solved the "static board" problem. By adding a "Ghost Layer" (Channel 18), the model now sees where pieces were one move ago, giving it a sense of momentum and motion.
- Multi-Task Training: The training loop was updated to balance two competing goals simultaneously: predicting the move (Policy Loss) and predicting the winner (Value Loss).
- The Hybrid Engine: I built the Engine class to be a true system, not just a network. It now combines three brains:
- Opening Book: Plays instant GM moves (Polyglot) in the opening.
- Neural Intuition: Uses the ResNet to sort moves and spot "human-like" candidates.
- Tactical Search: Uses Alpha-Beta pruning (Minimax) with PeSTO evaluation to verify the network's suggestions.

**Current State: Self-Play Capable**
- The system is now capable of full autonomous gameplay.
- I created src/play_self.py, which allows the engine to play against itself.
- I added a "Scenario Mode" that allows me to force specific openings (like the Sicilian) to test how the engine handles specific middlegame structures without waiting for them to happen randomly.

**Conclusion**

We have moved from a simple predictor to a robust chess engine. The code is modular (utils, engine, model), the architecture is modern (ResNet), and the engine can now play complete, valid games with a mix of memorized theory (Book) and calculated intuition (Search). The next step is to let it play thousands of games to see if the ResNet actually learns strategy or just tactics.

## Journal #10

**Date**: Jan 11, 2026 (12:15am)

**Focus:** Building an Evaluation System

**The Strategy: Adding a Judge**
 - After testing the previous model, it became clear that just copying moves from Grandmaster games isn't enough. The engine was playing moves that "looked" right but didn't actually understand if the position was good or bad. It had a move-predictor (Policy), but it lacked a judge (Value).
 - I decided to refactor the project into a modular system and created a separate Value Network. The goal is for this second model to look at any board state and provide a score between -1 and 1 to determine who is actually winning.

**The Development: Refactoring and Data**
 - This session involved restructuring the entire codebase to be cleaner and more organized.
 - Modular Design: I split the code into separate files for the model, the dataset, and general utilities. This makes it easier to manage the two different networks.
 - Data Perspective: I set up a new training pipeline that looks at the final result of games. Instead of just learning which move was played, the network now learns whether the current player's position eventually led to a win, loss, or draw.
 - Broadening the Training: I prepared the system to use a wider variety of high-level games. This helps the model learn that even if a position looks standard, it might be objectively worse if it leads to a loss.

**Current State: The Untested Value Model**
 - The refactor is done, and the Value Network has been trained. However, the two systems are still working in isolation.
 - The Player (Policy): Suggests moves based on common patterns and theory.
 - The Critic (Value): Provides a score for how good the board looks.
 - The Missing Link: These two models haven't been combined yet. I have the "instinct" and the "judgment" ready, but they aren't talking to each other.

**Conclusion**

The project is now much more professional and follows the same logic used by advanced engines like AlphaZero. The code is modular and the evaluation model is ready. The next step is to write a search algorithm that uses the Value Network to double-check the moves suggested by the Policy Network before they are played.

## Journal #9

**Date:** Jan 9, 2026 (11:40pm)

**Focus:** Failure Analysis (The "Polite" Blunder)

**The Strategy: Reality Check**

I took the "Strategy + Violence" model out for a live test to see if the puzzle training improved its tactical awareness.
- **Expectation:** The AI should punish blunders and take free pieces.
- **Reality:** The AI ignored a free Queen because it was too focused on "looking proper."

**Gameplay Analysis: Sicilian Defense vs. AI**

I played **White** against the AI (Black).
- **The Moves:** 1. e4 c5 2. d4 cxd4 3. Qxd4 Nc6 (The AI attacks my Queen).
- **The Test:** I played 4. Nf3??, deliberately leaving my Queen on d4 to be captured by the Knight on c6.
- **The AI's Response:** Instead of taking the free Queen (...Nxd4), the AI played **4... Nf6??**.
- **Diagnosis:** This is **Context Blindness** at its peak. The move ...Nf6 is a standard, "high-probability" Sicilian move. The model saw the pattern "Sicilian Defense" and output the most common developing move, completely failing to check if it could win the game on the spot.
- **Verdict:** The AI has learned *manners* (standard development) but lacks *killer instinct* (material greed). It is hallucinating a normal game instead of reacting to the board in front of it.

**Conclusion**

The model is "over-socialized." It wants to play a "normal" game of chess so badly that it refuses to deviate from standard theory, even when the opponent hangs a Queen. I need to implement a **Greedy Check** (can I take a piece for free?) alongside the Safety Check.

## Journal #8

**Date:** Jan 9, 2026 (6:53pm)

**Focus:** Puzzle Training 

**The Strategy: "Strategy + Violence"**

To fix the AI's passive play, we moved to a Mixed Training Strategy.

- Dataset A: 5,000 Grandmaster Games (Winner-Only) $\rightarrow$ Teaches Structure.
- Dataset B: 20,000 Lichess Puzzles (Mate in 1-5, Forks) $\rightarrow$ Teaches Tactics.
- Training: We updated train.py to merge these datasets, forcing the AI to balance "Book Moves" with "Killer Moves."

**Gameplay Analysis: Sicilian Defense vs. Human**

I played White against the new model.
- Opening: The AI played a sharp Sicilian (1. e4 c5 2. Nf3 Nf6). It correctly challenged the center with 5... f5 and developed active pieces (...Bb4, ...Qc7). This is lightyears ahead of the random shuffling we saw earlier.
- Middlegame: It castled into safety, showing it understands King Safety concepts.The Blunder: It succumbed to a mating attack (Nf6+ $\to$ Qxh7#).
- Verdict: The AI has learned to be aggressive, but it lacks Defensive Calculation. It didn't "see" the mate coming because it was trained primarily on delivering mates, not preventing them.

**Conclusion**

The "Bootcamp" worked. The bot is now a dangerous, albeit glass-cannon, opponent. It plays principled chess but needs to learn defensive resilience.

## Journal #7

**Date:** Jan 9, 2026 (5:11pm)

**Focus:** The "Value Head" Experiment & Winner-Only Filtering

**The Experiment: Adding a Critic (Value Head)**

I attempted to upgrade SicilianZero by adding a Value Head—a secondary output in the Neural Network to predict the game winner (-1.0 to 1.0).
- The Goal: To let the AI "calculate" by evaluating future board states, solving the "Blunder Blindness" problem.
- The Result: The Value Head successfully learned to evaluate Grandmaster games (Loss: 0.13), but failed in practice due to Distribution Shift.
- The Failure Mode: When faced with a "stupid" move (like a blunder), the Value Head—having never seen it in training—often "hallucinated" that the blunder was actually a winning move. This caused the AI to prefer chaos over safe play.

**"Winner-Only" Policy Learning**
I made the  decisin to revert the architecture to a pure Policy Network (Level 1) but drastically improve the Data Quality.

 - New Strategy: "Winner-Only Filtering."
 - Implementation: We updated make_dataset.py to discard all moves made by the losing side or in drawn games.
 - The Theory: If the AI only mimics winners, it naturally filters out the "losing patterns" that crept into the previous model.

**Gameplay Analysis: The "Poser" Problem**

I tested the new "Winner-Only" model against the English Attack.

 - The Good: It played the Sicilian Najdorf opening theory perfectly (...a6, ...e6, ...Be7). It "looks" like a Grandmaster.
 - The Bad: In the middlegame, it played ...Nd4. While visually appealing (centralizing a Knight), it was a blunder.
 - Verdict: The AI has learned Strategic Posture (where pieces usually go) but still lacks Tactical Awareness (why pieces go there). It is mimicking style without understanding consequences.

## Journal #6

**Date:** Jan 8, 2026 (11:12pm)

**Implementation: Legal Move Masking**

I successfully upgraded the Inference Engine 
 - Old Logic: "Pick the highest score. If illegal -> Random Move."
 - New Logic: "Filter scores through board.legal_moves. Pick the highest legal score."
 - Result: The "Fallback" error logs have completely vanished. The AI is now 100% robust.

**Gameplay Analysis: The Sicilian Najdorf**

I tested the engine against the Open Sicilian.
 - Moves 1-9: The AI played the Najdorf Variation (5... a6) perfectly, including the complex "Scheveningen" setup (e6, b5, Bb7).
 - Verdict: The CNN has successfully memorized high-level opening theory.

**The Failure Mode: "Blunder Blindness"**

At Move 13, I intentionally blundered a piece (Nxe6??) to test the AI's tactical awareness.
 - The AI's Reaction: It refused to take the free piece, playing Qd7 instead.
 - The Root Cause: Distribution Shift. The model is trained on top-tier GM games where dropping a piece is statistically impossible.
 - Conclusion: The AI assumes every move I make is "Grandmaster Quality." It lacks the ability to refute bad moves because it has never seen a bad move in its training data. It has learned Strategy (Mimicry) but not Calculation (Truth).

## Journal #5

**Date:** Jan 8, 2026 (8:12pm) 

**Focus:** Optimize training with LR Scheduler

**Training Performance**

I extended training to 20 epochs with a Learning Rate Scheduler.
- Start: Loss 3.37 (Roughly knowing legal moves).
- Finish: Loss 1.01 (Knowing specific theory).
- Observation: The scheduler worked perfectly. You can see the loss dropping consistently, proving the model was refining its weights without overfitting or getting stuck.

**The "King's Indian" Mirror**

I played a game opening with the King's Indian Attack (1. Nf3, 2. g3, 3. Bg2, 4. O-O). The AI's Response: It played the King's Indian Defense (1... Nf6, 2... g6, 3... Bg7, 4... O-O).

- Theory Check: The AI played 8 moves of perfect book theory. It mirrored my strategy, fianchettoed its bishop, and castled.
- Significance: This proves the CNN has successfully learned high-level strategic concepts like "Piece Development" and "King Safety." It didn't just move random pawns; it built a fortress.

**The Remaining Glitch:** 

At Move 9 and 11, the AI attempted illegal moves with the Knight on b8 (b8e5 and b8e1).Analysis: The Knight on b8 is blocked. The AI really wanted to put a Knight on e5 (a strong outpost square).The Failure Mode: The model correctly identified the Goal ("I want a Knight on e5"), but it selected the wrong Source piece (the trapped Knight on b8 instead of a free one).

**Conclusion:** The model has mastered Strategy (Where pieces should be) but still struggles with Tactical Constraints (Which pieces are allowed to go there).

## Journal #4

**Date:** Jan 8, 2026 (5:05pm)

**Focus:** First Inference Test

_Training logs:_
```
Epoch 1/5 | Loss: 5.7999
Epoch 2/5 | Loss: 4.9511
Epoch 3/5 | Loss: 4.4847
Epoch 4/5 | Loss: 4.0666
Epoch 5/5 | Loss: 3.6915
```

**The Test Case**

I played a game against the model (trained on 5,000 GM games) using the command line interface. I chose Black and played the Sicilian Defense (1... c5), fittingly for the repo name.

**Observations**
- Move 1 (White): AI played e2e4. (Perfect).
- Move 2 (White): After I played c5, AI played g1f3. (Perfect Main Line).
- Move 3 (White): After I played e6, AI attempted c2d4 (illegal). The theoretical move is d2d4 (The Open Sicilian).

**Interpretation** 

The AI correctly identified the strategic goal (control d4, break the center), but suffered a "Coordinate Hallucination." It selected the c2 pawn instead of d2.

**Technical Analysis**
 - This error reveals the current limitation of the model: Spatial Precision vs. Pattern Recognition.
 - The CNN (Layer 3) successfully recognized the "Open Sicilian" pattern.
 - The Output Head (Linear) failed to distinguish between the c2 and d2 squares, likely because they share very similar feature maps (both are pawns in front of the King/Queen).

**Conclusion**

The model has moved past "Random Play" and entered the "Student Phase." It knows what to do (play the Open Sicilian), but it is clumsy in execution. This suggests that while 5,000 games is enough to learn concepts, I likely need more training epochs or a lower learning rate to refine the precision of the output heads.

Next Step: How to fix the "clumsiness"?
- The model "knows" the right idea but misses by one square. This is usually fixed by:
- More Training: 5 epochs might not be enough for the weights to settle on exact coordinates.
- Learning Rate Decay: Lowering the learning rate (lr=0.0001) after a few epochs so it stops "overshooting" the correct square.

## Journal #3

**Date:** Jan 8, 2026 (3:45pm)

**Focus:** The Training Loop & Optimization Theory

**Training Loop Architecture Architecture**

I implemented a standard PyTorch training loop to optimize the ChessNet weights. The loop follows the Forward -> Loss -> Backward -> Step cycle.
- Loss Function: nn.CrossEntropyLoss
- Optimizer: optim.Adam (Learning Rate: 0.001)
- Strategy: Multi-Task Learning (Summing the loss of two distinct heads).

**Key Technical Concepts**
- Logits (The Raw Scores)
  - What they are: When the model outputs numbers for the 64 squares, they are not probabilities yet. They are raw scores called Logits.
  - Example: Square E4 might have a score of 5.2, while A1 is -1.2.
  - Why use them: I shouldn't force the model to output percentages directly because the math is more stable if we keep them as raw numbers until the very last moment (the Loss function).
- CrossEntropyLoss
  - The Problem: How do I calculate the error when the answer is a category ("Square E4") rather than a number? I can't just subtract E4 - E2.
  - The Solution: Cross Entropy measures the difference between two probability distributions.
  - The Truth: A distribution where Square E4 has 100% probability and all others have 0%.
  - The Prediction: The model says E4 is 80%, D4 is 15%, etc. 
  - The Math: It takes the Softmax of the logits (converting them to percentages) and then calculates the negative log likelihood of the correct class.
  - Intuition: If the model assigns low probability to the correct square, the Loss explodes (punishing it heavily). If it assigns high probability, the Loss nears zero.

**Backpropagation**
 - What it is: loss.backward()
 - The Logic: After calculating the Loss (e.g., 2.75), I need to tell the network who is responsible for the error.
 - The Process: PyTorch uses the Chain Rule of Calculus to go backward from the output to the input. It calculates a Gradient for every single weight.
 - Gradient: "If I increase this weight by 0.001, will the error go up or down?"

**The Optimizer: Adam**
 - Why not SGD? Standard Stochastic Gradient Descent (SGD) applies the same "learning speed" (learning rate) to every parameter.
 - Why Adam? (Adaptive Moment Estimation) It gives every single parameter its own custom learning rate.
  - It has "Momentum": If a weight has been moving in the same direction for a while, Adam speeds it up (like a ball rolling down a hill).
  - Result: It converges significantly faster on complex landscapes like Chess strategy.

**Engineering Decision: Multi-Task Loss**

The model has two distinct outputs: From_Square and To_Square.
- The Implementation: Total_Loss = Loss_From + Loss_To
- The Theory: By summing the losses, I force the Backpropagation to send error signals from both tasks into the shared Convolutional Backbone.
- The Benefit: The model learns a shared understanding of the board. A feature like "This Knight is pinned" is mathematically useful for both selecting the piece (don't move it) and choosing a destination (can't go anywhere), so the shared layers learn it faster.

## Journal #2

**Date:** Jan 8, 2026 (1:36pm)

**Focus:** Deep Learning Architecture & CNN Theory

**The Core Paradigm: Chess as an Image**

I transitioned from treating chess as a discrete sequence of moves to treating it as a spatial vision problem.
- The Analogy: Just as an image is defined by (Channels, Height, Width) (e.g., RGB is $3 \times H \times W$), a chess board is (12, 8, 8).
- The Channels: Instead of Red/Green/Blue, my channels are the 12 piece types (White Pawns, Black Kings, etc.).
- The Solution: I selected a Convolutional Neural Network (CNN). Standard feed-forward networks ignore spatial relationships (e.g., that square A1 is next to A2). CNNs preserve this topology, allowing the model to detect local patterns like "pawn chains" or "king safety" regardless of where they appear on the board.

**Engineering Decision: Kernel Size ($3 \times 3$)**

I chose a standardized $3 \times 3$ kernel size rather than larger options ($5 \times 5$ or $8 \times 8$). 
- The "Flashlight" Concept: The kernel acts as a small window scanning the board for features.
- Efficiency: A $3 \times 3$ kernel has only 9 weights per filter. A $5 \times 5$ has 25. By stacking multiple $3 \times 3$ layers, I achieve the same "field of view" as larger kernels but with significantly fewer parameters and lower computational cost (FLOPs).
- Standardization: This aligns with modern architecture standards (e.g., VGGNet, ResNet), prioritizing "depth over width."

**Engineering Decision: Network Depth & Feature Hierarchy**

I implemented a 3-layer architecture. This was not arbitrary; it mimics human cognition through Feature Hierarchy:
- Layer 1 (Geometry): The model learns simple, immediate relationships (e.g., "Is there a piece on the adjacent square?").
- Layer 2 (Patterns): By combining Layer 1 outputs, it detects intermediate structures (e.g., "Fianchettoed Bishop" or "Castled Position").
- Layer 3 (Strategy): It combines patterns to understand global board states (e.g., "Weak Back Rank").

**The Math: Receptive Fields**

I calculated the Receptive Field (how much of the board a single output neuron can "see") to justify the 3-layer depth.
- Layer 1: Sees a $3 \times 3$ area.
- Layer 2: Expands to $5 \times 5$ (neighbors of neighbors).
- Layer 3: Expands to $7 \times 7$.
- Conclusion: Since the chess board is $8 \times 8$, a 3-layer network allows the final decision layer to have near-global context ($49/64$ squares directly) without the risk of Vanishing Gradients associated with deeper networks (10+ layers).

**Components: Activations & Normalization**

ReLU (Rectified Linear Unit): I used ReLU to introduce non-linearity.
- Without this, the network would just be a linear regression model. ReLU allows the network to make discrete "decisions" (activation vs. inhibition).
- Batch Normalization: I applied BatchNorm2d after each convolution. This mitigates Internal Covariate Shift, ensuring that the distribution of inputs to each layer remains stable (mean 0, variance 1). This allows for higher learning rates and faster convergence during training.

## Journal #1

**Date**: January 8, 2026 (12:30pm)

**Focus**: Infrastructure & ETL Pipeline

**The Problem:** Data Representation. I needed to convert a standard Chess Board state (64 squares with pieces) into a format a Convolutional Neural Network (CNN) can process.

- Initial thought: Use a simple 8x8 grid where Pawn=1, Knight=2, etc.
- The Flaw: This introduces ordinality. A neural network might interpret a King (6) as "worth 6 times more" than a Pawn (1) mathematically, rather than categorically.
- The Solution: I implemented a 12-Channel Tensor Representation (One-Hot Encoding).
  - Shape: (12, 8, 8)
  - Logic: Channel 0 is exclusively for White Pawns, Channel 1 for White Knights, etc. This separates categorical data spatially, allowing the CNN to learn patterns like "Pawn Structure" or "Bishop Scope" independently.

**The Problem: Scalability (The RAM Bottleneck)** 

The Lichess dataset is massive (millions of games). Loading a .pgn file into a standard Python list would crash the memory (RAM).
- The Solution: I implemented a Python Generator using the yield keyword.
- Outcome: This creates a "lazy loader" that streams one game at a time from the disk, processes it, and discards it. Memory usage remains constant O(1) regardless of file size.














