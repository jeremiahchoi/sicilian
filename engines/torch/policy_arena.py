import chess
import chess.pgn
import torch
import numpy as np
import datetime
from model import ChessNet
from utilities import board_to_matrix, get_legal_move_mask

# === CONFIGURATION ===
# Path to the White player's brain
WHITE_MODEL_PATH = "engines/models/chess_model.pth" 

# Path to the Black player's brain (Can be the same file or a different version)
BLACK_MODEL_PATH = "engines/models/chess_model_epoch_5.pth" 

DEVICE = 'cpu' # CPU is fast enough for inference

class ChessAI:
    def __init__(self, model_path, name="AI"):
        self.name = name
        self.model = ChessNet()
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            self.model.to(DEVICE)
            self.model.eval()
            print(f"Loaded {self.name} from {model_path}")
        except FileNotFoundError:
            print(f"ERROR: Could not find model at {model_path}")
            exit()

    def get_move(self, board):
        # 1. Prepare Input
        matrix = board_to_matrix(board)
        input_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 2. Forward Pass
        with torch.no_grad():
            output = self.model(input_tensor)

        # 3. Mask Illegal Moves
        legal_move_indices = get_legal_move_mask(board)
        filtered_output = torch.full(output.shape, float('-inf')).to(DEVICE)
        filtered_output[0, legal_move_indices] = output[0, legal_move_indices]

        # 4. Argmax
        best_move_idx = torch.argmax(filtered_output).item()

        # 5. Decode Move
        from_sq, to_sq = divmod(best_move_idx, 64)

        # Un-flip if Black
        if board.turn == chess.BLACK:
            f_r, f_c = divmod(from_sq, 8)
            t_r, t_c = divmod(to_sq, 8)
            from_sq = chess.square(f_c, 7 - f_r)
            to_sq = chess.square(t_c, 7 - t_r)

        move = chess.Move(from_sq, to_sq)
        
        # Auto-promote to Queen
        if board.piece_type_at(move.from_square) == chess.PAWN:
            if (board.turn == chess.WHITE and chess.square_rank(move.to_square) == 7) or \
               (board.turn == chess.BLACK and chess.square_rank(move.to_square) == 0):
                move.promotion = chess.QUEEN
        
        return move

def run_match():
    # 1. Initialize Players
    white_ai = ChessAI(WHITE_MODEL_PATH, name="White_Bot")
    black_ai = ChessAI(BLACK_MODEL_PATH, name="Black_Bot")

    # 2. Setup Board & PGN
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Neural Network Arena"
    game.headers["White"] = white_ai.name
    game.headers["Black"] = black_ai.name
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    
    node = game

    print(f"\nStarting Match: {white_ai.name} vs {black_ai.name}")
    print("-" * 30)

    # 3. Game Loop
    move_count = 1
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = white_ai.get_move(board)
            # Print cleanly
            print(f"{move_count}. {move.uci()} ", end="")
        else:
            move = black_ai.get_move(board)
            print(f"{move.uci()}")
            move_count += 1
            
        board.push(move)
        node = node.add_variation(move)
        
        # Optional: Stop if game is too long (random loops)
        if move_count > 150:
            print("\nGame aborted (Too long).")
            break

    # 4. Result
    print("\n" + "-" * 30)
    print(f"Game Over. Result: {board.result()}")
    game.headers["Result"] = board.result()

    # 5. Output PGN
    print("\n=== PGN (Copy this to Lichess) ===")
    print(game)
    
    # Save to file
    with open("arena_game.pgn", "w") as f:
        print(game, file=f)
    print("\nSaved to arena_game.pgn")

if __name__ == "__main__":
    run_match()