import numpy as np
import chess
import chess.pgn
import os
import sys

# Import from your package
from src.data_processor import stream_games, board_to_tensor, encode_move

def process_pgn_to_dataset(pgn_path, output_dir, max_games=5000):
    os.makedirs(output_dir, exist_ok=True)
    
    X = [] 
    y = [] 
    
    game_count = 0
    
    print(f"Starting processing for: {pgn_path}")
    print("Strategy: Winner-Only Filtering (Ignoring losers and draws)")
    
    for game in stream_games(pgn_path):
        if game_count >= max_games:
            break
            
        # 1. Determine the Winner
        result = game.headers.get("Result", "*")
        if result == "1-0":
            winning_color = chess.WHITE
        elif result == "0-1":
            winning_color = chess.BLACK
        else:
            continue # Skip draws and unknown results
            
        board = game.board()
        
        for move in game.mainline_moves():
            # 2. FILTER: Only record the move if it was made by the Winner
            if board.turn == winning_color:
                tensor = board_to_tensor(board)
                X.append(tensor)
                
                from_sq, to_sq = encode_move(move)
                y.append([from_sq, to_sq])
            
            # Always push the move to advance the board, even if we didn't save it
            board.push(move)
            
        game_count += 1
        if game_count % 100 == 0:
            print(f"Processed {game_count} games...")

    # Convert and Save
    X_array = np.array(X, dtype=np.float32) 
    y_array = np.array(y, dtype=np.int64)   
    
    output_filename = os.path.join(output_dir, "chess_dataset.npz")
    print(f"Saving dataset: {X_array.shape} samples to {output_filename}...")
    
    np.savez_compressed(
        output_filename, 
        inputs=X_array, 
        labels=y_array
    )
    print("Done!")
    return output_filename

if __name__ == "__main__":
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    pgn_path = os.path.join(raw_dir, "grandmaster_games.pgn")
    
    if os.path.exists(pgn_path):
        process_pgn_to_dataset(pgn_path, processed_dir, max_games=5000)
    else:
        print(f"ERROR: Could not find {pgn_path}")