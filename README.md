# Project Roadmap
## Phase 1: The Translator (Data Pipeline)
**Goal:** Ensure we can reliably turn a chess move into numbers
- Test Case 1:
  - _Logic_: Given a standard board, the output must be exactly `(12,8,8)`
- Test Case 2:
  - Place a White Knight on `e4`. The tensor at `channel=1` (white knights), `row=4`,`col=4` must be `1.0`. All other squares in that channel must be `0.0`

## Phase 2: The Model Architecture
**Goal:** Ensure the neural network accepts the input and produces a legal move format.
- Test Case 3:
  - _Logic:_ Pass a random tensor of shape `(1,12,8,8)` into the model. Does it return two vectors `64` (To/From squares)? (Catches dimension mismatch errors immediately)
 
## Phase 3: The "Sanity Check"
**Goal:** Prove the model can learn
- Test Case 4:
  - _Logic:_ Take **one** game position. Train the model on only that position for 100 epochs.
  - _Pass Condition:_ Loss must go to nearly 0
