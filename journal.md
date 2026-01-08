# Project: Behavioral Cloning Chess Engine 

**Journal #2**

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

**Journal #1**

**Date**: January 8, 2026 (12:30pm)

**Focus**: Infrastructure & ETL Pipeline

**The Problem:** Data Representation I needed to convert a standard Chess Board state (64 squares with pieces) into a format a Convolutional Neural Network (CNN) can process.

- Initial thought: Use a simple 8x8 grid where Pawn=1, Knight=2, etc.
- The Flaw: This introduces ordinality. A neural network might interpret a King (6) as "worth 6 times more" than a Pawn (1) mathematically, rather than categorically.
- The Solution: I implemented a 12-Channel Tensor Representation (One-Hot Encoding).
  - Shape: (12, 8, 8)
  - Logic: Channel 0 is exclusively for White Pawns, Channel 1 for White Knights, etc. This separates categorical data spatially, allowing the CNN to learn patterns like "Pawn Structure" or "Bishop Scope" independently.

**The Problem: Scalability (The RAM Bottleneck)** The Lichess dataset is massive (millions of games). Loading a .pgn file into a standard Python list would crash the memory (RAM).
- The Solution: I implemented a Python Generator using the yield keyword.
- Outcome: This creates a "lazy loader" that streams one game at a time from the disk, processes it, and discards it. Memory usage remains constant O(1) regardless of file size.




