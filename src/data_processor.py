import chess
import chess.pgn
import numpy as np

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Converts a chess.Board object into a 12x8x8 Numpy array

    Structure: (1-hot encoding)
    - 12 Channels: 6 for White (P, N, B, R, Q, K), 6 for Black
    - 8x8 Grid: Represents the board

    Return: np.ndarray: Shape (12, 8, 8) with dtype float32
    """

    # Initialize empty tensor
    # dtype=float32 is crucial for ML
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    # Map piece types to channel offsets (0-5)
    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    # Iterate over all 64 squares
    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece:
            # determine base channel (0-5)
            channel = piece_map[piece.piece_type]

            # if Black, shift channel by 6
            if piece.color == chess.BLACK:
                channel += 6

            row = chess.square_rank(square)
            col = chess.square_file(square)

            tensor[channel, row, col] = 1.0

    return tensor

def stream_games(file_path):
    """
    A Generator that reads a PGN file and yields games one by one (to save RAM by not loading whole file at once)

    Args:
        file_path (str): Path to .pgn file.
    
    Yields:
        chess.pgn.Game: A game object
    """
    # Open the file in 'read' mode
    with open(file_path, 'r') as pgn_file:
        while True:
            try:
                # chess.pgn.read_game reads one game and moves the file pointer
                # to the start of the next game automatically
                game = chess.pgn.read_game(pgn_file)
            except Exception:
                # If a game is corrupted
                break

            if game is None:
                break

            # Pause here and give game to caller
            yield game

def encode_move(move: chess.Move) -> tuple: 
    """
    Converts a chess.Move object into two integers representing the
    start and end squares (0-63)

    Args: 
        move (chess.Move): The move to encode.
    
    Returns:
        tuple: (from_square_index, to_square_index)
    """

    return move.from_square, move.to_square