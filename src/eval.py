import chess

# --- 1. PIECE VALUES (Midgame, Endgame) ---
# Format: [None, Pawn, Knight, Bishop, Rook, Queen, King]
MG_VALUES = [0, 82, 337, 365, 477, 1025, 0]
EG_VALUES = [0, 94, 281, 297, 512, 936, 0]

# --- 2. PIECE-SQUARE TABLES  ---
# Values are from the famous "PeSTO" evaluation tuning.

# PAWN
MG_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
]
EG_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
]

# KNIGHT
MG_KNIGHT = [
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
]
EG_KNIGHT = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
]

# BISHOP
MG_BISHOP = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
]
EG_BISHOP = [
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   5,  10, 10,   5,   6,   7,
    -15, -18, -12, -10, -8, -15, -15, -14,
    -14, -18, -13,  -4, -9, -14, -27, -24,
]

# ROOK
MG_ROOK = [
     32,  42,  32,  51, 63,   9,  31,  43,
     27,  32,  58,  62, 80,  67,  26,  44,
     -5,  19,  26,  36, 17,  45,  61,  16,
    -24, -11,   7,  26, 24,  35,  -8, -20,
    -36, -26, -12,  -1,  9,  -7,   6, -23,
    -45, -25, -16, -17,  3,   0,  -5, -33,
    -44, -16, -20,  -9, -1,  11,  -6, -71,
    -19, -13,   1,  17, 16,   7, -37, -26,
]
EG_ROOK = [
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
]

# QUEEN
MG_QUEEN = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9, -19, -30, -15, -56, -22,
]
EG_QUEEN = [
    -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
]

# KING
MG_KING = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
]
EG_KING = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
]

TABLES_MG = [None, MG_PAWN, MG_KNIGHT, MG_BISHOP, MG_ROOK, MG_QUEEN, MG_KING]
TABLES_EG = [None, EG_PAWN, EG_KNIGHT, EG_BISHOP, EG_ROOK, EG_QUEEN, EG_KING]

# Game Phase Calculation (Total non-pawn material)
# P=0, N=1, B=1, R=2, Q=4
GAME_PHASE_INC = [0, 0, 1, 1, 2, 4, 0] 

def evaluate_board(board):
    """
    Returns the evaluation of the board from White's perspective.
    Score is in 'centipawns' (100 = 1 pawn).
    """
    mg_score = 0
    eg_score = 0
    game_phase = 0
    
    # 1. Sum up scores for every piece
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue
            
        ptype = piece.piece_type
        
        # Calculate Phase (How "Endgame" are we?)
        game_phase += GAME_PHASE_INC[ptype]
        
        # Get Piece-Square Value
        # If White: Index matches board (0 is a1)
        # If Black: We need to mirror the index (flip rank)
        if piece.color == chess.WHITE:
            sq_idx = square ^ 56 # Flip vertically to match table layout above (Top=0, Bottom=63)
            # Actually, PeSTO tables are usually defined A1=0 to H8=63 OR A8=0...
            # The definition above is Rank 8 (Top) to Rank 1 (Bottom).
            # So square 0 (A1) needs to map to the LAST row of the table.
            table_idx = square ^ 56 # Maps A1(0) -> A8(56) which is the first row of table??
            # Wait, let's stick to standard: The table above is [A8...H8, ..., A1...H1]
            # So index 0 in table is A8. Index 63 is H1.
            # Chess.SQUARES: A1=0, H8=63.
            # So for WHITE, A1 (0) should hit the *bottom* of the table (indices 56-63).
            # square ^ 56 flips the ranks.
            table_idx = square ^ 56
            
            mg_val = MG_VALUES[ptype] + TABLES_MG[ptype][table_idx]
            eg_val = EG_VALUES[ptype] + TABLES_EG[ptype][table_idx]
            
            mg_score += mg_val
            eg_score += eg_val
            
        else:
            # For Black, A8(56) is their "back rank".
            # If A1=0 is bottom left.
            # Black piece on A8 (56) should map to the "bottom row" of the table (indices 56-63)
            # because the table is defined relative to the player.
            # So for BLACK, square 56 should become 56. 
            # Wait, no. The table defines "Good squares for ME".
            # For White, A1 is back rank. For Black, A8 is back rank.
            # If White on A1 uses table index 56.
            # Black on A8 should use table index 56.
            # square 56 ^ 0 = 56.
            table_idx = square 
            
            mg_val = MG_VALUES[ptype] + TABLES_MG[ptype][table_idx]
            eg_val = EG_VALUES[ptype] + TABLES_EG[ptype][table_idx]
            
            mg_score -= mg_val
            eg_score -= eg_val

    # 2. Tapered Evaluation Formula
    # Max phase is roughly 24 (4*N + 4*B + 4*R + 2*Q)
    # If phase > 24, clamp it.
    phase = min(game_phase, 24)
    
    # Interpolate
    # If phase = 24 (Opening), use MG score mostly.
    # If phase = 0 (Endgame), use EG score fully.
    
    score = (mg_score * phase + eg_score * (24 - phase)) // 24
    
    # 3. Return from perspective of side to move?
    # Usually eval functions return "White advantage".
    # Engine logic usually expects "Score relative to current player".
    
    if board.turn == chess.BLACK:
        return -score
        
    return score