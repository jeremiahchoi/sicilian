import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# === IMPORTS ===
from value_model import ValueNet 

# === CONFIGURATION ===
DATA_FILE = "data/Lichess.pt" 
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    print(f"Loading tensors from {DATA_FILE}...")
    try:
        data = torch.load(DATA_FILE, map_location='cpu')
        inputs = data['inputs']
        targets = data['targets']
        print(f"Loaded {len(inputs)} positions.")
        
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
    except FileNotFoundError:
        print(f"ERROR: Could not find {DATA_FILE}.")
        return

    # 2. Initialize Model
    model = ValueNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() 

    # 3. Training Loop
    model.train()
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch_idx, (b_inputs, b_targets) in enumerate(dataloader):
            b_inputs, b_targets = b_inputs.to(DEVICE), b_targets.to(DEVICE)

            # Ensure Float
            b_inputs = b_inputs.float()
            b_targets = b_targets.float()

            # === THE FIX IS HERE ===
            # Reshape target from [64] to [64, 1] to match output
            b_targets = b_targets.unsqueeze(1) 

            optimizer.zero_grad()
            outputs = model(b_inputs) 
            loss = criterion(outputs, b_targets) 
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 500 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | MSE Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"=== Epoch {epoch+1} Finished. Avg MSE: {avg_loss:.4f} ===")
        
    torch.save(model.state_dict(), "value_model.pth")
    print("Training complete.")

if __name__ == "__main__":
    train()