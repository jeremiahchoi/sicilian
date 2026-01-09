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
    
    # 1. Point to the Real Data (The file created by download_data.py)
    pgn_path = os.path.join(raw_dir, "grandmaster_games.pgn")
    
    if os.path.exists(pgn_path):
        print(f"Found real data at {pgn_path}!")
        
        # 2. Run the pipeline on the real data
        # We process up to 5000 games now
        output_file = process_pgn_to_dataset(pgn_path, processed_dir, max_games=5000)
        
        # 3. Verification
        if output_file:
            data = np.load(output_file)
            inputs = data['inputs']
            print("\n--- VERIFICATION ---")
            print(f"Input Shape: {inputs.shape}") 
            print(f"Label Shape: {data['labels'].shape}")
            print(f"SUCCESS: Dataset created with {inputs.shape[0]} positions!")
    else:
        print(f"ERROR: Could not find {pgn_path}")
        print("Did you run 'python src/download_data.py' first?")