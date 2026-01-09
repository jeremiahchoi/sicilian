import berserk
import os
from datetime import datetime

def download_gm_games(output_file="data/raw/grandmaster_games.pgn", max_games_per_player=500):
    # 1. Setup Client (No token needed for public data, but slower)
    # If you have a token: session = berserk.TokenSession("YOUR_TOKEN")
    # client = berserk.Client(session)
    client = berserk.Client()
    
    grandmasters = [
        'DrNykterstein',  # Magnus Carlsen
        'alireza2003',    # Alireza Firouzja
        'RebeccaHarris',  # Daniel Naroditsky (Huge volume)
        'penguingim1',    # Andrew Tang (Huge volume, plays fast)
        'Zhigalko_Sergei',# Sergey Zhigalko (Lichess regular)
        'Night-King96',   # Olexandr Bortnyk
        'lachesisQ',      # Ian Nepomniachtchi
        'Krucke',         # A solid GM account
        'Mutdpro',        # Another active GM
        'SAValchev'       # Active GM
    ]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"--- Starting Download: {len(grandmasters)} GMs, {max_games_per_player} games each ---")
    
    total_games = 0
    
    # Open file in 'append' mode so we save progress as we go
    with open(output_file, "w") as f:
        for player in grandmasters:
            print(f"Fetching games for: {player}...")
            try:
                # Fetch games (this is a generator)
                games = client.games.export_by_player(
                    player, 
                    max=max_games_per_player, 
                    perf_type='blitz', # Blitz is good: high quality but lots of data
                    as_pgn=True        # We want the raw PGN text
                )
                
                count = 0
                for game_pgn in games:
                    f.write(game_pgn)
                    f.write("\n\n") # Separate games clearly
                    count += 1
                
                print(f"  -> Saved {count} games.")
                total_games += count
                
            except Exception as e:
                print(f"  -> Error fetching {player}: {e}")
                
    print(f"\nTotal Downloaded: {total_games} games.")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    download_gm_games()