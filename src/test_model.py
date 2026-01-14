import torch
import torch.nn.functional as F
import chess
import numpy as np
import time
import sys
import os

# Setup path to import src
sys.path.append(os.getcwd())

from src.model import ChessNet
from src.utils import board_to_matrix, encode_move

# --- CONFIG ---
# Update this path to point to your trained model
MODEL_PATH = "models/v2/v2_final.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_MOVES = 200 # Prevent infinite games

def predict_move(model, board, temperature=1.0):
    """
    Uses the Neural Network to predict the best move (Policy Head)
    and evaluation (Value Head).
    
    Args:
        temperature (float): 
            1.0 = Standard Sampling
            0.1 = Almost deterministic (Play the 'best' move)
            2.0 = Creative / Random
    """
    # 1. Prepare Input
    tensor = board_to_matrix(board).unsqueeze(0).to(DEVICE) # Add batch dim
    
    # 2. Inference
    with torch.no_grad():
        policy_logits, value = model(tensor)
        
    # 3. Filter Illegal Moves
    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        return None, 0 # Checkmate or Stalemate
        
    move_candidates = []
    move_probs = []
    
    for move in legal_moves:
        # Get the index for this specific move
        idx = encode_move(move, board.turn)
        
        # Grab the raw logit (score)
        logit = policy_logits[0, idx].item()
        move_candidates.append(move)
        move_probs.append(logit)
        
    # 4. Softmax (Convert scores to probabilities)
    move_probs = np.array(move_probs)
    
    # Apply Temperature
    move_probs = move_probs / temperature
    
    # Numerical stability
    move_probs = move_probs - np.max(move_probs) 
    probs = np.exp(move_probs)
    probs = probs / np.sum(probs)
    
    # 5. Select Move (Weighted Random Choice)
    choice_idx = np.random.choice(len(move_candidates), p=probs)
    
    # Also return the evaluation (Value Head)
    eval_score = value.item()
    
    return move_candidates[choice_idx], eval_score

def play_self_game():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}. Train a bit first!")
        return

    print(f"🧠 Loading model: {MODEL_PATH}...")
    model = ChessNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Freeze BatchNorm/Dropout
    
    board = chess.Board()
    print("\nStarting Self-Play Test...\n")
    print(board)
    print("-" * 30)
    
    move_count = 0
    
    while not board.is_game_over() and move_count < MAX_MOVES:
        move_count += 1
        
        # Predict move
        move, eval_score = predict_move(model, board, temperature=0.8)
        
        if move is None: break
        
        # Print Stats
        turn_name = "White" if board.turn == chess.WHITE else "Black"
        print(f"\nMove {move_count}: {turn_name} plays {move.uci()}")
        
        # Interpretation: +1.0 means Current Player thinks they are winning
        print(f"Confidence: {eval_score:.3f} ({'Optimistic' if eval_score > 0 else 'Pessimistic'})")
        
        board.push(move)
        print(board)
        
        time.sleep(0.5) 
        
    print("\nGame Over!")
    print(f"Result: {board.result()}")

if __name__ == "__main__":
    play_self_game()