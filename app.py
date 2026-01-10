import streamlit as st
import chess
import chess.svg
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import base64
import time

# --- SETUP ---
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from src.model import ChessNet
    from src.data_processor import board_to_tensor
except ImportError:
    st.error("⚠️ Could not import src/model.py. Make sure your folder structure is correct.")
    st.stop()

st.set_page_config(page_title="SicilianZero", page_icon="♟️", layout="wide")

# --- 1. LOAD BRAIN ---
@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = ChessNet()
    model_path = "models/chess_model.pth"
    
    if not os.path.exists(model_path):
        return None
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        return None

model = load_model()

# --- 2. GAME LOGIC ---
def get_ai_move(board):
    if model is None: 
        import random
        move = random.choice(list(board.legal_moves))
        return move, "Model not loaded. Playing Random."
    
    tensor_np = board_to_tensor(board)
    tensor = torch.from_numpy(tensor_np).unsqueeze(0).float()
    
    with torch.no_grad():
        out_from, out_to = model(tensor)
        prob_from = F.softmax(out_from, dim=1).squeeze()
        prob_to = F.softmax(out_to, dim=1).squeeze()
        
        move_scores = []
        for move in board.legal_moves:
            score = prob_from[move.from_square].item() * prob_to[move.to_square].item()
            move_scores.append((score, move))
        
        move_scores.sort(key=lambda x: x[0], reverse=True)
        best_move = move_scores[0][1]
        
        # Format Top 3 (Compact)
        top_3 = [f"<b>{i+1}. {m.uci()}</b> ({s*100:.1f}%)" for i, (s, m) in enumerate(move_scores[:3])]
        return best_move, " &nbsp;&nbsp; ".join(top_3)

# --- 3. UI HELPER ---
def render_board(board, is_white=True):
    svg = chess.svg.board(
        board=board,
        size=450,
        flipped=not is_white,
        colors={'square light': '#e9edcc', 'square dark': '#779954'}
    )
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    
    # CSS HACK 1: Negative margin here pulls the BUTTONS closer to the board
    return f"""
    <div style="display: flex; justify-content: center; margin-bottom: -65px;">
        <img src="data:image/svg+xml;base64,{b64}" width="100%"/>
    </div>
    """

# --- 4. MAIN APP ---
st.markdown(f"""
<div margin-bottom: -200px>
</div>""",
unsafe_allow_html=True)
st.title("♟️ SicilianZero")

if 'board' not in st.session_state:
    st.session_state.board = chess.Board()
if 'view_index' not in st.session_state:
    st.session_state.view_index = 0
if 'ai_stats' not in st.session_state:
    st.session_state.ai_stats = "Waiting for game start..."

# -- LAYOUT --
left_col, mid_col, right_col = st.columns([1, 2, 1], gap="medium")

# --- LEFT: CONTROLS ---
with left_col:
    st.subheader("Controls")
    player_color = st.radio("Play as:", ["White", "Black"], horizontal=True, index=0)
    user_is_white = (player_color == "White")

    st.markdown("---")
    
    with st.form("move_form", clear_on_submit=True):
        move_str = st.text_input("Enter Move (UCI)", placeholder="e.g. e2e4")
        submitted = st.form_submit_button("Play Move", use_container_width=True)

    st.markdown("---")
    if st.button("New Game", type="primary", use_container_width=True):
        st.session_state.board = chess.Board()
        st.session_state.view_index = 0
        st.session_state.ai_stats = "New Game Started."
        st.rerun()

# --- MIDDLE: BOARD & NAV ---
with mid_col:
    # 1. AI Confidence Banner
    # CSS HACK 2: Negative margin here pulls the BOARD closer to the banner
    st.markdown(
        f"""
        <div style="background-color: rgba(97, 218, 251, 0.1); padding: 8px; border-radius: 5px; margin-bottom: -80px; text-align: center; border: 1px solid rgba(97, 218, 251, 0.3);">
            <small style="color: #ccc; margin-right: 10px;">🧠 AI Confidence:</small>
            <span style="color: white;">{st.session_state.ai_stats}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Board State Logic
    move_stack = st.session_state.board.move_stack
    total_moves = len(move_stack)
    if 'view_index' not in st.session_state or st.session_state.view_index > total_moves:
        st.session_state.view_index = total_moves

    if st.session_state.view_index == total_moves:
        display_board = st.session_state.board
    else:
        display_board = chess.Board()
        for i in range(st.session_state.view_index):
            display_board.push(move_stack[i])

    # 3. Render Board
    st.markdown(render_board(display_board, is_white=user_is_white), unsafe_allow_html=True)
    
    # 4. Navigation Buttons
    col_space_L, c1, c2, c3, c4, col_space_R = st.columns([2, 1, 1, 1, 1, 2])
    
    if c1.button("⏮", use_container_width=True, help="Start"): 
        st.session_state.view_index = 0
        st.rerun()
    if c2.button("◀", use_container_width=True, help="Back"): 
        st.session_state.view_index = max(0, st.session_state.view_index - 1)
        st.rerun()
    if c3.button("▶", use_container_width=True, help="Next"): 
        st.session_state.view_index = min(total_moves, st.session_state.view_index + 1)
        st.rerun()
    if c4.button("⏭", use_container_width=True, help="End"): 
        st.session_state.view_index = total_moves
        st.rerun()

# --- RIGHT: LOG ---
with right_col:
    st.subheader("Move Log")
    pgn_text = ""
    for i, move in enumerate(move_stack):
        move_num = i // 2 + 1
        if i % 2 == 0:
            pgn_text += f"**{move_num}.** "
        
        if i == st.session_state.view_index - 1:
            pgn_text += f"__`{move.uci()}`__ "
        else:
            pgn_text += f"{move.uci()} "
            
        if i % 2 == 1:
            pgn_text += "\n"

    with st.container(height=500):
        st.markdown(pgn_text)

# --- GAME LOOP ---
if submitted and move_str:
    try:
        move = chess.Move.from_uci(move_str)
        if move in st.session_state.board.legal_moves:
            st.session_state.board.push(move)
            st.session_state.view_index = len(st.session_state.board.move_stack)
            st.rerun()
        else:
            st.toast("Illegal move!", icon="❌")
    except ValueError:
        st.toast("Invalid UCI format.", icon="⚠️")

# AI Turn
is_human_turn = (st.session_state.board.turn == chess.WHITE and user_is_white) or \
                (st.session_state.board.turn == chess.BLACK and not user_is_white)

if not st.session_state.board.is_game_over() and not is_human_turn:
    with st.spinner("AI is thinking..."):
        time.sleep(0.2)
        ai_move, stats = get_ai_move(st.session_state.board)
        st.session_state.ai_stats = stats
        st.session_state.board.push(ai_move)
        st.session_state.view_index = len(st.session_state.board.move_stack)
        st.rerun()