# Project: Behavioral Cloning Chess Engine 

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





