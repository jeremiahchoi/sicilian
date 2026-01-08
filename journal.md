# Project: Behavioral Cloning Chess Engine 

**Date**: January 8, 2026 

**Phase**: Infrastructure & ETL Pipeline

1. The Problem: Data Representation I needed to convert a standard Chess Board state (64 squares with pieces) into a format a Convolutional Neural Network (CNN) can process.

Initial thought: Use a simple 8x8 grid where Pawn=1, Knight=2, etc.

The Flaw: This introduces ordinality. A neural network might interpret a King (6) as "worth 6 times more" than a Pawn (1) mathematically, rather than categorically.

The Solution: I implemented a 12-Channel Tensor Representation (One-Hot Encoding).

Shape: (12, 8, 8)

Logic: Channel 0 is exclusively for White Pawns, Channel 1 for White Knights, etc. This separates categorical data spatially, allowing the CNN to learn patterns like "Pawn Structure" or "Bishop Scope" independently.

2. The Problem: Scalability (The RAM Bottleneck) The Lichess dataset is massive (millions of games). Loading a .pgn file into a standard Python list would crash the memory (RAM).

The Solution: I implemented a Python Generator using the yield keyword.

Outcome: This creates a "lazy loader" that streams one game at a time from the disk, processes it, and discards it. Memory usage remains constant O(1) regardless of file size.

3. Engineering Standards: Packaging & TDD

packaging: Instead of hacking sys.path, I structured the project as an installable package using setuptools (pip install -e .). This mimics production library standards.

TDD: I wrote tests for the Tensor shapes and Generator stream before implementing the logic. This caught an off-by-one error in the Rank/Row mapping immediately.
