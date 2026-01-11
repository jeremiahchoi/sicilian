import chess
import chess.pgn
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from utilities import board_to_matrix

class ChessDataset(Dataset):
    def __init__(self, pgn_file, max_games=None):
        # We will store the data as big tensors in memory
        self.inputs = None
        self.targets = None
        
        # 1. SMART CACHING LOGIC
        base_name = os.path.splitext(pgn_file)[0]
        self.processed_path = f"{base_name}.pt"

        if os.path.exists(self.processed_path):
            print(f"Loading cached data from {self.processed_path}...")
            try:
                data = torch.load(self.processed_path)
                self.inputs = data['inputs']
                self.targets = data['targets']
                print(f"Loaded {len(self.inputs)} samples.")
                return 
            except Exception as e:
                print(f"Error loading cache: {e}. Re-processing...")

        # 2. PARSE PGN (If no cache)
        print(f"Parsing games from {pgn_file}...")
        
        # Temporary lists to hold data
        temp_inputs = []
        temp_targets = []

        with open(pgn_file) as f:
            count = 0
            while True:
                if max_games and count >= max_games: break
                
                try:
                    game = chess.pgn.read_game(f)
                except Exception:
                    break # specific pgn errors
                    
                if game is None: break
                
                count += 1
                if count % 100 == 0: print(f"Processing game {count}...")
                
                board = game.board()
                
                for move in game.mainline_moves():
                    # Input (X)
                    matrix = board_to_matrix(board)
                    
                    # Target (Y)
                    u_from = move.from_square
                    u_to = move.to_square
                    
                    if board.turn == chess.BLACK:
                        r_from, c_from = divmod(u_from, 8)
                        r_to, c_to = divmod(u_to, 8)
                        u_from = chess.square(c_from, 7 - r_from)
                        u_to = chess.square(c_to, 7 - r_to)

                    move_index = (u_from * 64) + u_to
                    
                    # Append to temp lists
                    temp_inputs.append(matrix)
                    temp_targets.append(move_index)
                    
                    board.push(move)
        
        print(f"Parsing complete. Converting {len(temp_inputs)} positions to Tensors...")

        # 3. CONVERT TO TENSORS (Crucial Step for Speed/Size)
        # Instead of a list of arrays, we make one giant array
        self.inputs = torch.tensor(np.array(temp_inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.array(temp_targets), dtype=torch.long)
        
        # Clear memory
        del temp_inputs
        del temp_targets

        # 4. SAVE COMPACT TENSORS
        print(f"Saving to {self.processed_path}...")
        torch.save({'inputs': self.inputs, 'targets': self.targets}, self.processed_path)
        print("Save complete.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Data is already tensor, just return it
        return self.inputs[idx], self.targets[idx]

if __name__ == "__main__":
    PGN_FILE = "data/Carlsen.pgn" 
    
    if not os.path.exists(PGN_FILE):
        print(f"Error: Could not find {PGN_FILE}")
    else:
        # Lower max_games slightly if you still crash (e.g., 5000)
        # But this optimized version should handle 10k fine on 16GB RAM
        dataset = ChessDataset(PGN_FILE, max_games=5000) 
        print("Done!")