import chess
import numpy as np
import torch
from torch.utils.data import Dataset

def board_to_matrix(board):
    """
    Converts a chess.Board into an 18x8x8 Tensor with Canonical (Relative) Encoding.
    
    The board is ALWAYS oriented so the current player's pieces are at the 'bottom' (ranks 0-1)
    and moving 'up' (towards ranks 6-7). This allows the model to play as both White and Black.
    
    Dimensions: (18, 8, 8)
    
    --- CHANNEL BREAKDOWN ---
    [0]:  My Pawns
    [1]:  My Knights
    [2]:  My Bishops
    [3]:  My Rooks
    [4]:  My Queens
    [5]:  My King
    [6]:  Enemy Pawns
    [7]:  Enemy Knights
    [8]:  Enemy Bishops
    [9]:  Enemy Rooks
    [10]: Enemy Queens
    [11]: Enemy King
    [12]: My King-side Castling Right (All 1s if true, else 0s)
    [13]: My Queen-side Castling Right
    [14]: Enemy King-side Castling Right
    [15]: Enemy Queen-side Castling Right
    [16]: En Passant Target (Marked with 1.0 at the target square)
    [17]: Previous Board Occupancy (Ghost Layer: Where pieces were 1 move ago)
    """
    # Initialize 18x8x8 matrix
    state = np.zeros((18, 8, 8), dtype=np.float32)
    
    # 1. Perspective Flipping
    # If Black to move, we flip the board indices (0->63, 1->62) so it looks like White.
    is_black = (board.turn == chess.BLACK)
    
    def orient(square):
        return square ^ 56 if is_black else square

    # 2. PIECES (Channels 0-11)
    # We iterate over all squares to fill piece planes
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Flip square index if necessary
            relative_sq = orient(square)
            rank, col = divmod(relative_sq, 8)
            
            # Determine Channel: 0-5 (My), 6-11 (Enemy)
            is_my_piece = (piece.color == board.turn)
            offset = 0 if is_my_piece else 6
            idx = offset + (piece.piece_type - 1)
            
            state[idx, rank, col] = 1.0

    # 3. CASTLING RIGHTS (Channels 12-15)
    # [My K, My Q, Op K, Op Q]
    my_color = board.turn
    op_color = not my_color
    
    if board.has_kingside_castling_rights(my_color): 
        state[12, :, :] = 1.0
    if board.has_queenside_castling_rights(my_color): 
        state[13, :, :] = 1.0
    if board.has_kingside_castling_rights(op_color): 
        state[14, :, :] = 1.0
    if board.has_queenside_castling_rights(op_color): 
        state[15, :, :] = 1.0
        
    # 4. EN PASSANT TARGET (Channel 16)
    if board.ep_square is not None:
        relative_ep = orient(board.ep_square)
        r, c = divmod(relative_ep, 8)
        state[16, r, c] = 1.0
        
    # 5. PREVIOUS BOARD OCCUPANCY (Channel 17)
    # We peek into the past by popping the last move
    if len(board.move_stack) > 0:
        last_move = board.pop() # Undo last move temporarily
        
        for square in chess.SQUARES:
            # If ANY piece was here 1 move ago
            if board.piece_at(square):
                relative_sq = orient(square)
                r, c = divmod(relative_sq, 8)
                state[17, r, c] = 1.0
                
        board.push(last_move) # Restore board state
    
    # Return as PyTorch Tensor
    return torch.tensor(state).float()

def encode_move(move, turn):
    """
    Converts a chess.Move to an integer index (0-4095).
    Handles perspective flipping (Relative Encoding).
    """
    if turn == chess.BLACK:
        # Flip the coordinates so the move looks like it was made by White
        from_sq = move.from_square ^ 56
        to_sq = move.to_square ^ 56
    else:
        from_sq = move.from_square
        to_sq = move.to_square
        
    return from_sq * 64 + to_sq

def decode_action(action_idx, board):
    """
    Converts an integer index (0-4095) back into a chess.Move.
    Handles perspective flipping based on board.turn.
    Returns: A chess.Move object (may be illegal, needs validation).
    """
    from_sq = action_idx // 64
    to_sq = action_idx % 64
    
    if board.turn == chess.BLACK:
        # Flip back to absolute coordinates
        from_sq ^= 56
        to_sq ^= 56
        
    # Note: This doesn't handle promotion logic perfectly (defaults to None)
    # A robust engine iterates legal moves to find the matching from/to squares.
    return chess.Move(from_sq, to_sq)
