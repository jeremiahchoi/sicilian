import numpy as np
import chess
import chess.pgn
import os
import sys

# Import from your package
from src.data_processor import stream_games, board_to_tensor, encode_move

def process_pgn_to_dataset(pgn_path, output_dir, max_games=1000):
    """
    Reads a PGN file, converts games to tensors/labels, and saves them.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    X = [] 
    y = [] 
    
    game_count = 0
    move_count = 0
    
    print(f"Starting processing for: {pgn_path}")
    
    # --- LOOP STARTS HERE ---
    for game in stream_games(pgn_path):
        if game_count >= max_games:
            break
            
        board = game.board()
        
        for move in game.mainline_moves():
            tensor = board_to_tensor(board)
            X.append(tensor)
            
            from_sq, to_sq = encode_move(move)
            y.append([from_sq, to_sq])
            
            board.push(move)
            move_count += 1
            
        game_count += 1
        if game_count % 10 == 0:
            print(f"Processed {game_count} games...")
    # --- LOOP ENDS HERE --- 
    # (Make sure the code below is aligned with the 'for' loop, NOT inside it)

    # Convert lists to numpy arrays
    X_array = np.array(X, dtype=np.float32) 
    y_array = np.array(y, dtype=np.int64)   
    
    # Save to disk
    output_filename = os.path.join(output_dir, "chess_dataset.npz")
    print(f"Saving dataset: {X_array.shape} samples to {output_filename}...")
    
    np.savez_compressed(
        output_filename, 
        inputs=X_array, 
        labels=y_array
    )
    print("Done!")
    
    return output_filename # <--- CRITICAL: This must be here!

# --- MAIN BLOCK ---
if __name__ == "__main__":
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(raw_dir, exist_ok=True)
    
    # Create Dummy PGN
    dummy_pgn_path = os.path.join(raw_dir, "test_games.pgn")
    dummy_data = """
[Event "Test Game 1"]
[Result "0-1"]
1. f3 e5 2. g4 Qh4# 0-1

[Event "Test Game 2"]
[Result "*"]
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *
"""
    with open(dummy_pgn_path, "w") as f:
        f.write(dummy_data)
        
    # Run pipeline
    output_file = process_pgn_to_dataset(dummy_pgn_path, processed_dir)
    
    # Verify
    if output_file: # Check if it's not None
        print("\n--- VERIFICATION ---")
        data = np.load(output_file)
        inputs = data['inputs']
        print(f"Input Shape: {inputs.shape}") 
        
        if inputs.shape[0] == 10:
            print("SUCCESS: Count matches expected moves!")
        else:
            print(f"WARNING: Expected 10 moves, got {inputs.shape[0]}")
    else:
        print("ERROR: Function returned None!")