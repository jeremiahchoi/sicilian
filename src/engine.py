import chess
import torch
import numpy as np
import time
import sys
import os

sys.path.append(os.getcwd())

from src.model import ChessNet
from src.utils import board_to_matrix, encode_move

from eval import evaluate_board 

class Engine:
    def __init__(self, model_path=None, device=None):
        # 1. Device Setup
        if device is None:
            if torch.cuda.is_available(): self.device = torch.device("cuda")
            elif torch.backends.mps.is_available(): self.device = torch.device("mps")
            else: self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"🤖 Engine initializing on {self.device}...")

        # 2. Load Model (Intuition)
        self.use_policy = False
        if model_path and os.path.exists(model_path):
            self.model = ChessNet().to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.use_policy = True
            print(f"🧠 Policy Network loaded from {model_path}")
        else:
            print("⚠️ No model found. Engine will use pure calculation + heuristics.")

    def get_best_move(self, board, depth=3):
        start_time = time.time()
        
        best_val = -float('inf')
        best_move = None
        
        # Get moves sorted by Intuition + Tactics
        moves = self._get_ordered_moves(board)
        
        alpha = -float('inf')
        beta = float('inf')

        # Root Search
        for move in moves:
            board.push(move)
            val = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            if val > best_val:
                best_val = val
                best_move = move
            
            alpha = max(alpha, val)

        elapsed = time.time() - start_time
        print(f"🚀 Best: {best_move} | Eval: {best_val} | Depth: {depth} | Time: {elapsed:.2f}s")
        return best_move

    def _negamax(self, board, depth, alpha, beta):
        # 1. Leaf Node / Game Over
        if depth == 0 or board.is_game_over():
            if board.is_checkmate(): return -99999
            if board.is_stalemate() or board.is_insufficient_material(): return 0
            return evaluate_board(board)

        # 2. Move Ordering (Hybrid)
        # At high depths, use the expensive NN + Logic sort.
        # At low depths (depth 1), just use capture heuristics to save time.
        if depth > 1 and self.use_policy:
            moves = self._get_ordered_moves(board)
        else:
            # Fast Fallback: Just prioritize captures for speed at depth 1
            moves = sorted(board.legal_moves, key=lambda m: board.is_capture(m), reverse=True)

        max_val = -float('inf')

        for move in moves:
            board.push(move)
            val = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            if val > max_val:
                max_val = val
            
            alpha = max(alpha, val)
            if alpha >= beta:
                break

        return max_val

    def _get_ordered_moves(self, board):
        """
        Sorts moves by a weighted score of:
        1. Neural Network Probability (Intuition)
        2. Captures (Tactics)
        3. Checks (Forcing Moves)
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves: return []
        
        probs = None
        if self.use_policy:
            # Batch inference
            tensor = board_to_matrix(board).unsqueeze(0).to(self.device)
            with torch.no_grad():
                policy_logits, _ = self.model(tensor)
                # Convert logits to probabilities (0.0 to 1.0)
                probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        
        move_scores = []
        for move in legal_moves:
            score = 0.0
            
            # A. Neural Network Score (0.0 to 10.0 points)
            if probs is not None:
                idx = encode_move(move, board.turn)
                # Scale probability up so it competes with integer bonuses
                score += probs[idx] * 10.0 
            
            # B. Tactical Bonuses
            # Capture Bonus (+2.0)
            # This ensures that if the NN is "unsure" (e.g. 15% vs 10%), 
            # we check the capture first.
            if board.is_capture(move):
                score += 2.0
                
            # Check Bonus (+1.0)
            if board.gives_check(move):
                score += 1.0
                
            # Promotion Bonus (+3.0)
            if move.promotion:
                score += 3.0
                
            move_scores.append((score, move))
            
        # Sort Highest Score First
        move_scores.sort(key=lambda x: x[0], reverse=True)
        
        return [m for s, m in move_scores]