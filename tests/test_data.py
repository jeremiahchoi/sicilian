import pytest
import chess
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processor import board_to_tensor, stream_games

# -- Test 1: Tensor Conversion Logic ---
def test_tensor_shape():
    """Ensure output is strictly (12, 8, 8)"""
    board = chess.Board()
    tensor = board_to_tensor(board)
    assert tensor.shape == (12, 8, 8)
    assert tensor.dtype == np.float32

def test_knight_placement():
    """Test that a piece goes to the correct channel and coordinate"""
    board = chess.Board(None)
    # Place White Knight on E4
    board.set_piece_at(chess.E4, chess.Piece(chess.KNIGHT, chess.WHITE))

    tensor = board_to_tensor(board)

    # White Knight is channel 1, E4 is rank 3, file 4
    assert tensor[1, 3, 4] == 1.0

    # Ensure no 'ghost' pieces 
    assert np.sum(tensor) == 1.0

def test_stream_games():
    """Test that the generator yields games correctly from a temporary file."""

    # Create a fake PGN file content
    dummy_pgn_data = """[Event "Game 1"]
1. e4 e5 *

[Event "Game 2"]
1. d4 d5 *
"""
    # Write this dummy data to a real temp file
    temp_filename = "temp_test.pgn"
    with open(temp_filename, "w") as f:
        f.write(dummy_pgn_data)

    try:
        # Run the generator
        games_generator = stream_games(temp_filename)

        # Get first game
        game1 = next(games_generator)
        assert game1.headers["Event"] == "Game 1"

        # Get second game
        game2 = next(games_generator)
        assert game2.headers["Event"] == "Game 2"

    finally: 
        # Cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)