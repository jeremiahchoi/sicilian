import chess
import torch
import numpy as np
from value_model import ValueNet
from utilities import board_to_matrix

# Load the model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ValueNet().to(DEVICE)
model.load_state_dict(torch.load("value_model.pth", map_location=DEVICE))
model.eval()

def evaluate(fen):
    board = chess.Board(fen)
    matrix = board_to_matrix(board)
    # Convert to tensor and add batch dimension [1, 13, 8, 8]
    input_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        score = model(input_tensor).item()
    
    # Perspective correction: 
    # If it's Black's turn, the model's '1.0' means 'Black is winning' 
    # relative to the current player. Let's flip it for White-absolute score.
    absolute_score = score if board.turn == chess.WHITE else -score
    
    print(f"FEN: {fen}")
    print(f"Board View:\n{board}\n")
    print(f"Network Score (Absolute): {absolute_score:.4f}")
    print("-" * 30)

# --- TEST POSITIONS ---
test_fens = {
    "Starting Position (Equal)": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "White up a Queen (Winning)": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 3",
    "Black up a Queen (Losing)": "rnbqkbnr/ppppp2p/5p2/6pP/8/8/PPPPPPP1/RNBQKBNR b KQkq - 0 3",
    "Simple Ladder Mate (Winning)": "8/8/8/8/8/2k5/1r6/2K5 b - - 0 1",
    "K+K Draw (Drawn)": "8/8/8/4k3/8/4K3/8/8 w - - 0 1"
}

for desc, fen in test_fens.items():
    print(f"Testing: {desc}")
    evaluate(fen)