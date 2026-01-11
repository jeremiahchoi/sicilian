import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# === IMPORTS FROM YOUR MODULES ===
from engines.torch.policy_model import ChessNet
from policy_dataset import ChessDataset

# === CONFIGURATION ===
PGN_FILE = "data/Carlsen.pgn" 
BATCH_SIZE = 32        # Standard batch size
LEARNING_RATE = 0.001
EPOCHS = 10            # Increased: 5 is too low for 3,000 games
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    # We pass None for max_games so it loads the full cached .pt file
    dataset = ChessDataset(PGN_FILE, max_games=None)
    
    # Check if data actually loaded
    if len(dataset) == 0:
        print("Error: Dataset is empty. Did you run 'python dataset.py' first?")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Training on {len(dataset)} positions...")

    # 2. Initialize Model
    model = ChessNet().to(DEVICE)
    
    # 3. Setup Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()           # Reset
            outputs = model(inputs)         # Predict
            loss = criterion(outputs, labels) # Measure error
            loss.backward()                 # Calculate corrections
            optimizer.step()                # Update weights

            total_loss += loss.item()

            if batch_idx % 500 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"=== Epoch {epoch+1} Finished. Average Loss: {avg_loss:.4f} ===")
        
        # Optional: Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"chess_model_epoch_{epoch+1}.pth")

    # 5. Final Save
    torch.save(model.state_dict(), "chess_model.pth")
    print("Training complete. Model saved to 'chess_model.pth'")

if __name__ == "__main__":
    train()