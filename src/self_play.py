import sys
import os
import datetime
import chess
import chess.pgn

sys.path.append(os.getcwd())

from src.engine import Engine

# --- CONFIGURATION ---
MODEL_PATH = "models/v2/v2_final.pth"
OUTPUT_PGN = "data/games/sicilian_test.pgn"
MAX_MOVES = 50
SEARCH_DEPTH = 4

# 🔥 FORCE AN OPENING HERE
# Example: Sicilian Defense (1. e4 c5)
# Use UCI format: "e2e4", "e7e5", "g1f3", etc.
OPENING_MOVES = ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"] 

# Example: Ruy Lopez
# OPENING_MOVES = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]

# Example: Standard Start (Engine decides everything)
# OPENING_MOVES = [] 

def play_self_game(model_path, output_pgn, max_moves, depth, forced_opening):
    print(f"🚀 Initializing Engine...")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return

    # Initialize Engine (with optional book if you implemented it)
    engine = Engine(model_path=model_path)
    
    board = chess.Board()
    game = chess.pgn.Game()
    
    # Headers
    game.headers["Event"] = "Engine Opening Test"
    game.headers["White"] = f"MyChessEngine"
    game.headers["Black"] = f"MyChessEngine"
    
    node = game 

    print(f"\n⚔️ Starting Match!")
    if forced_opening:
        print(f"📝 Forcing Opening Line: {forced_opening}")
    print("-" * 50)

    # --- PHASE 1: PLAY FORCED MOVES (Instant) ---
    for move_uci in forced_opening:
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            board.push(move)
            node = node.add_variation(move) # Add to PGN
            print(f"Force Move: {move.uci()}")
        else:
            print(f"❌ Illegal opening move {move_uci} in position {board.fen()}")
            break

    # --- PHASE 2: ENGINE TAKES OVER ---
    while not board.is_game_over() and board.fullmove_number <= max_moves:
        
        # Visuals
        turn_color = "White" if board.turn == chess.WHITE else "Black"
        print(f"\nMove {board.fullmove_number} ({turn_color}): Thinking...")
        
        # ENGINE SEARCH
        # The engine only starts thinking NOW, from the current position
        best_move = engine.get_best_move(board, depth=depth)
        
        if best_move is None:
            print("⚠️ Engine resigned.")
            break
            
        node = node.add_variation(best_move)
        board.push(best_move)
        print(board)
    
    # Save Result
    print("-" * 50)
    game.headers["Result"] = board.result()
    
    os.makedirs(os.path.dirname(output_pgn), exist_ok=True)
    with open(output_pgn, "w") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)
    
    print(f"\n💾 Game saved to: {output_pgn}")

if __name__ == "__main__":
    play_self_game(MODEL_PATH, OUTPUT_PGN, MAX_MOVES, SEARCH_DEPTH, OPENING_MOVES)