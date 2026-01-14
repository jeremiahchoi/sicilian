import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# --- IMPORTS ---
# Assuming you run this from the project root (e.g., python src/train.py)
import sys
sys.path.append(os.getcwd()) 

from src.dataset import ChessDataset
from src.model import ChessNet

# --- CONFIGURATION ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
DATA_PATH = "data/processed/training_data.npz"
MODEL_SAVE_DIR = "models/v2" # Saving to v2 since we upgraded to ResNet

def train():
    # 1. Setup Environment
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Auto-detect hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps") # For Mac M1/M2/M3
        
    print(f"🚀 Training on device: {device}")

    # 2. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file {DATA_PATH} not found.")
        print("   Run 'python src/process_data.py' (or whatever your generator is) first.")
        return

    print("Loading dataset...")
    dataset = ChessDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✅ Loaded {len(dataset)} training positions.")

    # 3. Initialize Model
    model = ChessNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Define Loss Functions
    # Policy Loss: Predict the correct move index (0-4095) -> Cross Entropy
    policy_criterion = nn.CrossEntropyLoss()
    
    # Value Loss: Predict the win probability (-1 to 1) -> Mean Squared Error
    value_criterion = nn.MSELoss()

    # 5. Training Loop
    model.train()
    
    for epoch in range(EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for boards, target_policy, target_value in loop:
            # Move data to GPU/MPS
            boards = boards.to(device)         # (Batch, 18, 8, 8)
            target_policy = target_policy.to(device) # (Batch,)
            target_value = target_value.to(device).view(-1, 1) # (Batch, 1)

            optimizer.zero_grad()

            # --- Forward Pass ---
            pred_policy, pred_value = model(boards)

            # --- Calculate Loss ---
            loss_p = policy_criterion(pred_policy, target_policy)
            loss_v = value_criterion(pred_value, target_value)
            
            # COMBINED LOSS
            # We just sum them up. AlphaZero typically uses a weighted sum, 
            # but 1:1 works fine for starting out.
            loss = loss_p + loss_v

            # --- Backward Pass ---
            loss.backward()
            optimizer.step()

            # --- Logging ---
            total_policy_loss += loss_p.item()
            total_value_loss += loss_v.item()
            
            loop.set_postfix(p_loss=loss_p.item(), v_loss=loss_v.item())

        # End of Epoch Stats
        avg_p_loss = total_policy_loss / len(dataloader)
        avg_v_loss = total_value_loss / len(dataloader)
        
        print(f"   Stats: Policy Loss: {avg_p_loss:.4f} | Value Loss: {avg_v_loss:.4f}")

        # Save Checkpoint every epoch (optional, but good for safety)
        checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    # 6. Final Save
    final_path = os.path.join(MODEL_SAVE_DIR, "chess_resnet_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Training Complete. Best model saved to: {final_path}")

if __name__ == "__main__":
    train()