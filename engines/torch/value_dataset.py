import chess
import chess.pgn
import torch
import numpy as np
import os
import io
import zstandard as zstd  # <--- NEW IMPORT
from torch.utils.data import Dataset
from utilities import board_to_matrix

class ValueDataset(Dataset):
    def __init__(self, pgn_file, max_games=None):
        self.inputs = None
        self.targets = None
        
        # 1. SMART CACHING
        base_name = os.path.splitext(pgn_file)[0]
        # Handle the double extension .pgn.zst
        if base_name.endswith('.pgn'):
            base_name = os.path.splitext(base_name)[0]
            
        self.processed_path = f"{base_name}_value.pt"

        if os.path.exists(self.processed_path):
            print(f"Loading cached value data from {self.processed_path}...")
            data = torch.load(self.processed_path)
            self.inputs = data['inputs']
            self.targets = data['targets']
            print(f"Loaded {len(self.inputs)} value samples.")
            return

        # 2. PARSE PGN (Handle .zst or plain .pgn)
        print(f"Parsing games for Value Training from {pgn_file}...")
        
        # Helper to open file depending on extension
        def open_pgn(filepath):
            if filepath.endswith(".zst"):
                dctx = zstd.ZstdDecompressor()
                fh = open(filepath, 'rb')
                stream_reader = dctx.stream_reader(fh)
                return io.TextIOWrapper(stream_reader, encoding='utf-8')
            else:
                return open(filepath, 'r', encoding='utf-8')

        temp_inputs = []
        temp_targets = []

        # Open the file using our helper
        f = open_pgn(pgn_file)

        count = 0
        while True:
            if max_games and count >= max_games: break
            
            try:
                game = chess.pgn.read_game(f)
            except Exception:
                break
                
            if game is None: break
            
            count += 1
            if count % 500 == 0: print(f"Processing game {count}...")

            # === DETERMINE GAME RESULT ===
            result = game.headers.get("Result", "*")
            
            if result == "1-0":
                game_value = 1.0
            elif result == "0-1":
                game_value = -1.0
            elif result == "1/2-1/2":
                game_value = 0.0
            else:
                continue

            board = game.board()
            
            for move in game.mainline_moves():
                # 1. Input
                matrix = board_to_matrix(board)
                
                # 2. Target (Relative to turn)
                if board.turn == chess.WHITE:
                    label = game_value
                else:
                    label = -game_value 
                
                temp_inputs.append(matrix)
                temp_targets.append(label)
                
                board.push(move)
        
        # Close the file handle (important for zst streams)
        f.close()
        
        print(f"Parsing complete. Converting {len(temp_inputs)} positions to Tensors...")

        # 3. CONVERT TO TENSORS
        self.inputs = torch.tensor(np.array(temp_inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.array(temp_targets), dtype=torch.float32)
        
        # 4. SAVE
        print(f"Saving to {self.processed_path}...")
        torch.save({'inputs': self.inputs, 'targets': self.targets}, self.processed_path)
        print("Save complete.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx].unsqueeze(0)

if __name__ == "__main__":
    # Point this to your new compressed file
    PGN_FILE = "data/lichess_db_standard_rated_2017-04.pgn.zst"  # Check your actual filename!
    
    if os.path.exists(PGN_FILE):
        print("Processing data...")
        # Reduce from 10,000 to 5,000 to save disk space
        dataset = ValueDataset(PGN_FILE, max_games=5000) 
        print("Done!")
    else:
        print(f"File not found: {PGN_FILE}")