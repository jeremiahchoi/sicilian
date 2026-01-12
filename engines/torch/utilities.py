import chess
import numpy as np

def board_to_matrix(board: chess.Board):
    """
    Converts a chess.Board into a 13x8x8 matrix for the Neural Network.
    
    Structure:
    - Channels 0-5: My Pieces (P, N, B, R, Q, K)
    - Channels 6-11: Enemy Pieces (P, N, B, R, Q, K)
    - Channel 12: Legal Move Hints (All squares I can move to)
    
    Perspective:
    - Always "Relative". The network thinks it is White (playing from bottom to top).
    - If it is actually Black's turn, the board is flipped vertically.
    """
    # Initialize 13x8x8 matrix
    # We use (13, 64) first for easier bitboard filling, then reshape.
    matrix = np.zeros((13, 64), dtype=np.float32)
    
    # 1. Determine "Us" (Current Turn) vs "Them" (Opponent)
    us = board.turn
    them = not board.turn

    # 2. Define Piece Order
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    # === CHANNELS 0-5: MY PIECES ===
    for i, p_type in enumerate(piece_types):
        bb = board.pieces(p_type, us) # Get bitboard (integer)
        # Fill the row in the matrix where bits are 1
        # (We iterate the set bits for speed, or you could use np.unpackbits if you want)
        if bb:
            for sq in list(board.pieces(p_type, us)):
                matrix[i, sq] = 1.0

    # === CHANNELS 6-11: ENEMY PIECES ===
    for i, p_type in enumerate(piece_types):
        bb = board.pieces(p_type, them)
        if bb:
            for sq in list(board.pieces(p_type, them)):
                matrix[6 + i, sq] = 1.0

    # === CHANNEL 12: LEGAL MOVE HINTS ===
    # We mark every square that "Us" can legally move to.
    # This guides the network on "active" areas of the board.
    for move in board.legal_moves:
        to_sq = move.to_square
        matrix[12, to_sq] = 1.0

    # 3. Reshape to (13, 8, 8)
    # Shape becomes: [Channels, Rows, Columns]
    matrix = matrix.reshape(13, 8, 8)

    # 4. RELATIVE PERSPECTIVE FLIP
    # If it is Black's turn, we must flip the board vertically so the network
    # always sees "My Pieces" starting at the bottom (Rows 0-1).
    if us == chess.BLACK:
        # Flip along axis 1 (Rows). 
        # Axis 0 is Channels, Axis 1 is Rows, Axis 2 is Cols.
        matrix = np.flip(matrix, axis=1)

    return matrix.copy()

def get_legal_move_mask(board: chess.Board):
    """
    Returns a list of legal move indices (0-4095) for the current board state.
    
    CRITICAL: This handles 'Relative Encoding'.
    If it is Black's turn, we must flip the legal moves vertically so they match 
    the 'White Perspective' that the Neural Network was trained on.
    """
    mask_indices = []
    
    # Check if we need to flip (Is it Black's turn?)
    is_white = (board.turn == chess.WHITE)
    
    for move in board.legal_moves:
        f_sq = move.from_square
        t_sq = move.to_square
        
        # === RELATIVE PERSPECTIVE FLIP ===
        # The network thinks it's always White.
        # If we are Black, we must flip the board vertically (Rank 0 <-> Rank 7)
        if not is_white:
            # Flip 'From' Square
            f_r, f_c = divmod(f_sq, 8)
            f_sq = chess.square(f_c, 7 - f_r) # Keep file, flip rank
            
            # Flip 'To' Square
            t_r, t_c = divmod(t_sq, 8)
            t_sq = chess.square(t_c, 7 - t_r)

        # Calculate the flat index (0 to 4095)
        # This matches the output vector of your Neural Network
        idx = (f_sq * 64) + t_sq
        mask_indices.append(idx)
        
    return mask_indices

def move_to_index(move):
    """Converts a chess.Move to an index between 0 and 4095."""
    return move.from_square * 64 + move.to_square

def index_to_move(index):
    """Converts an index back into a chess.Move (useful for debugging)."""
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)