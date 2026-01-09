# SicilianZero

# To-do

## 1. Implement Legal Move Masking (Inference)
**Goal:** Prevent the AI from suggesting illegal moves so we don't need the "Random Fallback."

**Current Behavior:**
1. Model outputs raw scores for all 64 "From" squares and 64 "To" squares.
2. We pick the `argmax` (highest score).
3. If illegal, we play a random move.

**Proposed Solution:**
1. Get the list of all `legal_moves` from the `python-chess` board.
2. Create a "Mask" (a list of zeros) for the 64x64 possibilities.
3. Set the Mask to `1` only for moves that are currently legal.
4. Multiply `Model_Output * Mask`.
5. Select the highest score from the *remaining* options.

**Benefit:**
The AI will never make an illegal move, and its "second choice" (e.g., the correct Knight) will automatically win.

## 2. Upgrade Model Architecture (ResNet)
**Goal:** Improve strategic depth.
- Replace the simple 3-layer CNN with a Residual Network (ResNet) block.
- This allows for deeper networks (10+ layers) without vanishing gradients.

## 3. Self-Play Reinforcement Learning
**Goal:** Let the AI train itself.
- Instead of learning from GM games, let the AI play against itself.
- If it wins, reinforce the moves it made.
