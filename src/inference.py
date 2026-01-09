import torch
import chess
import numpy as np
import sys
import os

# Fix path to import modules if running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import ChessNet
from src.data_processor import board_to_tensor

def get_ai_move(model, board, device):
    """
    Selects the best LEGAL move based on the model's confidence scores.
    """
    model.eval()
    
    # 1. Prepare Data
    numpy_board = board_to_tensor(board)
    tensor = torch.from_numpy(numpy_board).unsqueeze(0).to(device)
    
    # 2. Get Raw Logits (Scores) from the Brain
    with torch.no_grad():
        out_from, out_to = model(tensor)
        
    # Remove batch dimension: (1, 64) -> (64)
    out_from = out_from.squeeze()
    out_to = out_to.squeeze()
    
    # 3. Get All Legal Moves (The "Mask")
    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        return None, None 
        
    # 4. Score Only the Legal Moves
    best_score = -float('inf')
    best_move = None
    
    # We loop through legal moves and see which one the Brain likes best
    for move in legal_moves:
        f = move.from_square
        t = move.to_square
        
        # Score = Brain's desire to move FROM here + Brain's desire to move TO there
        score = out_from[f].item() + out_to[t].item()
        
        if score > best_score:
            best_score = score
            best_move = move
            
    # 5. Return the winner
    from_sq_name = chess.square_name(best_move.from_square)
    to_sq_name = chess.square_name(best_move.to_square)
    
    return from_sq_name, to_sq_name

def play():
    # 1. Setup
    print("Initializing Neural Network...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    model = ChessNet().to(device)
    try:
        model.load_state_dict(torch.load("models/chess_model.pth", map_location=device))
        print("Brain loaded successfully!")
    except FileNotFoundError:
        print("No trained model found! (Run src/train.py first)")
        return

    # 2. Game Setup
    board = chess.Board()
    user_color = input("Do you want to play White or Black? (w/b): ").lower().strip()
    
    # Python-Chess uses True for White, False for Black
    player_is_white = (user_color == 'w')
    
    print("\n--- GAME START ---")
    print("Enter moves like 'e2e4' or 'g8f6'. Type 'quit' to exit.")

    while not board.is_game_over():
        # Display Board
        print("\n" + str(board))
        print("-" * 20)
        
        # Check whose turn it is
        is_user_turn = (board.turn == chess.WHITE and player_is_white) or \
                       (board.turn == chess.BLACK and not player_is_white)
        
        if is_user_turn:
            # --- HUMAN TURN ---
            move_str = input("Your Move: ").strip()
            if move_str == 'quit':
                break
                
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("❌ Illegal move. Try again.")
            except ValueError:
                print("❌ Invalid format. Use e2e4.")
                
        else:
            # --- AI TURN ---
            print("AI is thinking...")
            from_sq, to_sq = get_ai_move(model, board, device)
            ai_move_str = f"{from_sq}{to_sq}"
            
            try:
                move = chess.Move.from_uci(ai_move_str)
                
                if move in board.legal_moves:
                    print(f"🤖 AI plays: {ai_move_str}")
                    board.push(move)
                else:
                    # FALLBACK LOGIC
                    import random
                    fallback_move = random.choice(list(board.legal_moves))
                    print(f"🤖 AI tried illegal move '{ai_move_str}'. Playing random fallback: {fallback_move}")
                    board.push(fallback_move)
                    
            except ValueError:
                 # If AI outputs garbage (rare)
                import random
                fallback_move = random.choice(list(board.legal_moves))
                print(f"🤖 AI panicked. Playing random fallback: {fallback_move}")
                board.push(fallback_move)
                
    print("Game Over!")
    print(board.outcome())

if __name__ == "__main__":
    play()