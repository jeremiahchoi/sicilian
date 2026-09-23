"""
Export models/v2/v2_final.pth to ONNX for in-browser inference, verify the
export against PyTorch, and write a parity fixture that scripts/parity.mjs
uses to check the JavaScript port of the board encoding + masking against
the Python backend.

    ./venv/bin/pip install onnx onnxscript onnxruntime   # export-time only
    ./venv/bin/python scripts/export_onnx.py

Outputs:
    frontend/public/model/sicilianzero_v2.onnx
    frontend/scripts/parity_fixture.json
"""
import json
import random
import sys
from pathlib import Path

import chess
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import ChessNet  # noqa: E402
from src.utils import board_to_matrix  # noqa: E402
from backend.app import analyze, build_board, AnalyzeRequest  # noqa: E402

CKPT = ROOT / "models" / "v2" / "v2_final.pth"
ONNX_OUT = ROOT / "frontend" / "public" / "model" / "sicilianzero_v2.onnx"
FIXTURE_OUT = ROOT / "frontend" / "scripts" / "parity_fixture.json"


def export():
    model = ChessNet()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=True)
    model.eval()
    dummy = torch.zeros(1, 18, 8, 8)
    ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(ONNX_OUT),
        input_names=["board"], output_names=["policy_logits", "value"],
        opset_version=17, dynamo=False,
    )
    print(f"wrote {ONNX_OUT.relative_to(ROOT)} ({ONNX_OUT.stat().st_size / 1e6:.1f} MB)")
    return model


def verify(model):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX_OUT), providers=["CPUExecutionProvider"])
    worst = 0.0
    board = chess.Board()
    rng = random.Random(0)
    for _ in range(50):
        if board.is_game_over():
            board = chess.Board()
        board.push(rng.choice(list(board.legal_moves)))
        x = board_to_matrix(board).unsqueeze(0)
        with torch.no_grad():
            pl, v = model(x)
        opl, ov = sess.run(None, {"board": x.numpy()})
        worst = max(worst, float(np.abs(pl.numpy() - opl).max()), float(np.abs(v.numpy() - ov).max()))
    print(f"onnx vs torch max abs diff over 50 positions: {worst:.2e}")
    assert worst < 1e-4, "ONNX export does not match PyTorch"


def fixture():
    """Random games, mixed with a few scripted lines, recorded with the backend."""
    rng = random.Random(42)
    cases = []
    lines = [
        [],
        ["e2e4"],
        ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"],
        ["f2f3", "e7e5", "g2g4", "d8h4"],  # fool's mate: no legal moves
        ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"],  # scholar's mate
        ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "d2d4", "c7c6", "g1f3", "c8g4", "h2h3", "g4f3", "d1f3"],
        ["d2d4", "d7d5", "c2c4", "d5c4", "e2e3", "b7b5", "a2a4", "c7c6", "a4b5", "c6b5"],  # en passant chance
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "e1g1"],  # castling
    ]
    for moves in lines:
        cases.append(moves)
    for g in range(12):
        board = chess.Board()
        moves = []
        n = rng.randint(2, 60)
        for _ in range(n):
            if board.is_game_over():
                break
            mv = rng.choice(list(board.legal_moves))
            moves.append(mv.uci())
            board.push(mv)
            if rng.random() < 0.12:
                cases.append(list(moves))
        cases.append(list(moves))
    out = []
    for moves in cases:
        board, ghost = build_board(AnalyzeRequest(moves=moves))
        r = analyze(board, ghost)
        out.append({
            "moves": moves,
            "fen": r.fen,
            "turn": r.turn,
            "game_over": r.game_over,
            "move": r.move.uci if r.move else None,
            "top_moves": [{"uci": m.uci, "san": m.san, "prob": m.prob} for m in r.top_moves],
            "value": r.value,
            "n_legal": r.n_legal,
            "entropy_bits": r.entropy_bits,
            "legal_mass": r.legal_mass,
            "raw_from": r.raw["from"], "raw_to": r.raw["to"],
            "legal_from": r.legal["from"], "legal_to": r.legal["to"],
            "planes": board_to_matrix(board).numpy().reshape(-1).tolist(),
        })
    FIXTURE_OUT.write_text(json.dumps(out))
    print(f"wrote {FIXTURE_OUT.relative_to(ROOT)}: {len(out)} positions")


if __name__ == "__main__":
    m = export()
    verify(m)
    fixture()
