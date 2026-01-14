import argparse
import os
import glob
import numpy as np
import chess
import chess.pgn
import sys

sys.path.append(os.getcwd()) 

from src.utils import board_to_matrix, encode_move 

def get_game_result_value(game):
    """
    Parses the PGN header to get the game result.
    Returns:
         1.0 if White Won
        -1.0 if Black Won
         0.0 if Draw
         None if unknown/unfinished (*)
    """
    res = game.headers.get("Result", "*")
    if res == "1-0": return 1.0
    if res == "0-1": return -1.0
    if res == "1/2-1/2": return 0.0
    return None

def process_pgn_files(input_dir, output_file, max_games=None):
    # Buffers to hold our dataset
    inputs = []  # The Board State (18x8x8)
    policy = []  # The Move Played (Index 0-4095)
    value = []   # The Game Outcome (-1 to 1)
    
    # Find all .pgn files
    pgn_files = glob.glob(os.path.join(input_dir, "*.pgn"))
    print(f"📂 Found {len(pgn_files)} PGN files in {input_dir}")
    
    if not pgn_files:
        print("❌ No PGN files found! Please put your .pgn files in 'data/raw/'")
        return

    games_processed = 0
    
    for pgn_path in pgn_files:
        if max_games and games_processed >= max_games: break
        
        print(f"Reading {pgn_path}...")
        
        with open(pgn_path) as f:
            while True:
                # Break if we hit max games
                if max_games and games_processed >= max_games: break
                
                try:
                    game = chess.pgn.read_game(f)
                except Exception:
                    break
                
                if game is None: break # End of file
                
                # 1. GET GAME RESULT
                # We need this to train the Value Head
                global_result = get_game_result_value(game)
                if global_result is None: 
                    continue # Skip games with no result
                
                board = game.board()
                
                # 2. ITERATE THROUGH MOVES
                for move in game.mainline_moves():
                    # A. Save Input (Current Board)
                    # Use the logic from src/utils.py
                    tensor = board_to_matrix(board)
                    inputs.append(tensor.numpy()) 
                    
                    # B. Save Policy (The Move Played)
                    # Use the logic from src/utils.py
                    action_idx = encode_move(move, board.turn)
                    policy.append(action_idx)
                    
                    # C. Save Value (The Outcome)
                    # The Value Head is trained on "Canonical" results.
                    # If it's White's turn and White won, value is +1.
                    # If it's Black's turn and White won, value is -1 (Black lost).
                    turn_multiplier = 1.0 if board.turn == chess.WHITE else -1.0
                    value.append(global_result * turn_multiplier)
                    
                    # Advance board
                    board.push(move)
                
                games_processed += 1
                print(f"   Processed {games_processed} games...", end='\r')

    print(f"\n✅ Processing complete. Saving {len(inputs)} positions...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save as compressed .npz
    np.savez_compressed(
        output_file,
        inputs=np.array(inputs, dtype=np.float32),
        policy=np.array(policy, dtype=np.int64),
        value=np.array(value, dtype=np.float32)
    )
    print(f"💾 Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PGN files into training data.")
    
    parser.add_argument("--input_dir", type=str, default="data/raw", 
                        help="Folder containing .pgn files")
    
    parser.add_argument("--output_file", type=str, default="data/processed/training_data.npz", 
                        help="Output .npz file path")
    
    parser.add_argument("--max_games", type=int, default=5000, 
                        help="Maximum number of games to process")
    
    args = parser.parse_args()
    
    process_pgn_files(args.input_dir, args.output_file, args.max_games)