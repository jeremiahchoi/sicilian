"""
SicilianZero demo API.

Wraps the existing policy network (src/model.py) and board encoding
(src/utils.py) behind one endpoint. No search, no database, no sessions.
Game state lives in the browser; every request is self-contained.

Square indexing everywhere in the response is python-chess order:
a1 = 0, b1 = 1, ... h8 = 63 (absolute board coordinates, not the
side-to-move-relative coordinates the network sees internally).
"""
import math
import os
import sys
from pathlib import Path
from typing import Optional

import chess
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import ChessNet            # noqa: E402
from src.utils import board_to_matrix, encode_move  # noqa: E402

MODEL_PATH = ROOT / "models" / "v2" / "v2_final.pth"
# A 1.7M-parameter net runs a single position in a few ms on CPU; MPS/CUDA
# only add warm-up latency. Override with SICILIAN_DEVICE if you want.
DEVICE = torch.device(os.environ.get("SICILIAN_DEVICE", "cpu"))
ROUND = 6  # decimals kept in the JSON payload


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def load_model() -> ChessNet:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"checkpoint not found: {MODEL_PATH}")
    model = ChessNet().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


MODEL = load_model()
print(f"SicilianZero: loaded {MODEL_PATH.relative_to(ROOT)} on {DEVICE}")


# --------------------------------------------------------------------------
# Request / response
# --------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    # Preferred: the UCI move list from the standard start position. The
    # network's channel 17 ("ghost layer") encodes the previous position,
    # which board_to_matrix reads off board.move_stack, so replaying the
    # game reproduces what the model saw in training.
    moves: list[str] = []
    # Fallback: a bare FEN. Works, but the ghost layer is empty (all zeros),
    # which the model never saw during training. The response flags this.
    fen: Optional[str] = None


class MoveOut(BaseModel):
    uci: str
    san: str
    from_sq: int
    to_sq: int
    prob: float


class AnalyzeResponse(BaseModel):
    fen: str
    turn: str                    # "w" or "b": side the model analysed for
    game_over: bool
    result: Optional[str]
    ghost_layer_missing: bool
    move: Optional[MoveOut]      # argmax over legal moves; None if game over
    top_moves: list[MoveOut]
    value: float                 # value head, -1..1 from side-to-move's view
    n_legal: int
    entropy_bits: float          # entropy of the legal-masked distribution
    max_entropy_bits: float      # log2(n_legal): "uniform over legal moves"
    legal_mass: float            # share of raw softmax mass on legal cells
    raw: dict
    legal: dict


# --------------------------------------------------------------------------
# Board construction
# --------------------------------------------------------------------------
def build_board(req: AnalyzeRequest) -> tuple[chess.Board, bool]:
    """Returns (board, ghost_layer_missing)."""
    if req.moves:
        board = chess.Board()
        for i, uci in enumerate(req.moves):
            try:
                mv = chess.Move.from_uci(uci)
            except ValueError:
                raise HTTPException(400, f"moves[{i}] is not UCI: {uci!r}")
            if mv not in board.legal_moves:
                raise HTTPException(400, f"moves[{i}] {uci} is illegal in {board.fen()}")
            board.push(mv)
        if req.fen and req.fen.split(" ")[0] != board.fen().split(" ")[0]:
            raise HTTPException(400, "fen does not match the replayed move list")
        return board, False

    if req.fen:
        try:
            board = chess.Board(req.fen)
        except ValueError as e:
            raise HTTPException(400, f"bad FEN: {e}")
        if not board.is_valid():
            raise HTTPException(400, f"invalid position: {board.status()!r}")
        return board, True

    return chess.Board(), False


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def _to_absolute(rel: np.ndarray, turn: chess.Color) -> np.ndarray:
    """
    The network works in side-to-move-relative coordinates (utils.orient:
    square ^ 56 when Black is to move). Convert a 64x64 relative matrix to
    absolute board coordinates. XOR 56 is its own inverse, so the same
    permutation maps both ways.
    """
    if turn == chess.WHITE:
        return rel
    perm = np.arange(64) ^ 56
    return rel[perm][:, perm]


def _dist_payload(joint_abs: np.ndarray) -> dict:
    return {
        "from": np.round(joint_abs.sum(axis=1), ROUND).tolist(),
        "to": np.round(joint_abs.sum(axis=0), ROUND).tolist(),
        "joint": np.round(joint_abs, ROUND).tolist(),
    }


def analyze(board: chess.Board, ghost_missing: bool) -> AnalyzeResponse:
    tensor = board_to_matrix(board).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, value = MODEL(tensor)
    logits = logits[0].float().cpu().numpy()          # (4096,), relative coords

    # Raw distribution: softmax over all 4096 (from, to) pairs, no masking.
    raw_rel = torch.softmax(torch.from_numpy(logits), dim=0).numpy().reshape(64, 64)

    # Legal masking, exactly as src/test_model.py does it: gather the logits
    # at the legal move indices and softmax over only those.
    legal_moves = list(board.legal_moves)
    cell_to_move: dict[int, chess.Move] = {}
    for mv in legal_moves:
        idx = encode_move(mv, board.turn)
        # The four promotion moves share one (from, to) cell; keep the queen.
        prev = cell_to_move.get(idx)
        if prev is None or (mv.promotion == chess.QUEEN):
            cell_to_move[idx] = mv

    legal_rel = np.zeros(4096, dtype=np.float64)
    if cell_to_move:
        idxs = np.array(sorted(cell_to_move))
        z = logits[idxs].astype(np.float64)
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        legal_rel[idxs] = p
        legal_mass = float(raw_rel.reshape(-1)[idxs].sum())
        entropy = float(-(p * np.log2(p + 1e-30)).sum())
        max_entropy = math.log2(len(idxs))
    else:
        legal_mass = 0.0
        entropy = 0.0
        max_entropy = 0.0
    legal_rel = legal_rel.reshape(64, 64)

    # Rank legal moves.
    ranked = sorted(cell_to_move.items(), key=lambda kv: legal_rel.reshape(-1)[kv[0]], reverse=True)

    def move_out(idx: int, mv: chess.Move) -> MoveOut:
        return MoveOut(
            uci=mv.uci(),
            san=board.san(mv),
            from_sq=mv.from_square,
            to_sq=mv.to_square,
            prob=round(float(legal_rel.reshape(-1)[idx]), ROUND),
        )

    top = [move_out(i, m) for i, m in ranked[:5]]
    chosen = top[0] if top else None

    return AnalyzeResponse(
        fen=board.fen(),
        turn="w" if board.turn == chess.WHITE else "b",
        game_over=board.is_game_over(),
        result=board.result() if board.is_game_over() else None,
        ghost_layer_missing=ghost_missing,
        move=chosen,
        top_moves=top,
        value=round(float(value.item()), ROUND),
        n_legal=len(cell_to_move),
        entropy_bits=round(entropy, 4),
        max_entropy_bits=round(max_entropy, 4),
        legal_mass=round(legal_mass, ROUND),
        raw=_dist_payload(_to_absolute(raw_rel, board.turn)),
        legal=_dist_payload(_to_absolute(legal_rel, board.turn)),
    )


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
app = FastAPI(title="SicilianZero demo", docs_url="/api/docs", openapi_url="/api/openapi.json")


@app.get("/api/health")
def health():
    return {"ok": True, "model": str(MODEL_PATH.relative_to(ROOT)), "device": str(DEVICE)}


@app.post("/api/move", response_model=AnalyzeResponse)
def move(req: AnalyzeRequest):
    board, ghost_missing = build_board(req)
    return analyze(board, ghost_missing)
