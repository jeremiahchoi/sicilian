import zstandard as zstd
import requests
import chess
import chess.pgn
import io
import os

# Configuration
LICHESS_DB_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
OUTPUT_PGN = "data/raw/tactics.pgn"
MAX_PUZZLES = 20000  # Increased to 20k to capture more variety
THEMES = [
    # Checkmates (Calculation training)
    "mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5",
    # Winning Material (Greed training)
    "advantage",      # General winning position
    "crushing",       # Decisive advantage
    "hangingPiece",   # Free material
    "fork",           # Double attack
    "pin",            # Immobilizing pieces
    "skewer"          # X-ray attack
]

def download_and_process_puzzles():
    print(f"🔌 Connecting to Lichess Database Stream...")
    print(f"🎯 Target: {MAX_PUZZLES} puzzles")
    print(f"🧠 Themes: {', '.join(THEMES)}")
    
    response = requests.get(LICHESS_DB_URL, stream=True)
    dctx = zstd.ZstdDecompressor()
    
    stream_reader = dctx.stream_reader(response.raw)
    text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
    
    os.makedirs(os.path.dirname(OUTPUT_PGN), exist_ok=True)
    
    count = 0
    with open(OUTPUT_PGN, "w") as out_f:
        next(text_stream) # Skip header
        
        for line in text_stream:
            parts = line.split(',')
            
            # Data Parsing
            puzzle_id = parts[0]
            fen = parts[1]
            moves = parts[2].split()
            themes = parts[7]
            
            # FILTER: Must match at least one theme
            if not any(t in themes for t in THEMES):
                continue
            
            # Setup Board
            board = chess.Board(fen)
            
            try:
                # Apply Opponent's Blunder (Move 0)
                opponent_move = chess.Move.from_uci(moves[0])
                board.push(opponent_move)
                
                # The Solution (Move 1) - This is what we want to learn!
                my_solution = chess.Move.from_uci(moves[1])
            except:
                continue 
            
            # Create PGN Game
            game = chess.pgn.Game()
            game.setup(board)
            
            # "Winner-Only" Logic:
            # If it is White's turn to play the winning move, White wins.
            if board.turn == chess.WHITE:
                game.headers["Result"] = "1-0"
            else:
                game.headers["Result"] = "0-1"
                
            game.headers["Event"] = f"Lichess Puzzle {puzzle_id}"
            game.headers["Annotator"] = themes # Store themes for debugging
            
            # Save the winning move
            game.add_variation(my_solution)
            
            print(game, file=out_f, end="\n\n")
            
            count += 1
            if count % 100 == 0:
                print(f"Collected {count} tactics...", end='\r')
            
            if count >= MAX_PUZZLES:
                break
                
    print(f"\n✅ Success! Saved {count} tactical puzzles to {OUTPUT_PGN}")

if __name__ == "__main__":
    download_and_process_puzzles()