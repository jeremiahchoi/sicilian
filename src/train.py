import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from src.model import ChessNet

def load_data(npz_path, batch_size=32):
    """
    Loads the .npz file and creates a PyTorch DataLoader
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset not found at {npz_path}. Run make_dataset.py first!")
    
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)

    # Convert Numpy arrays to PyTorch Tensors
    # Inputs: float32 
    # Labels: long (int64) (req. for CrossEntropyLoss)
    inputs = torch.from_numpy(data['inputs'])
    labels = torch.from_numpy(data['labels']) # Shape: (N, 2)

    # Create Dataset and Loader
    dataset = TensorDataset(inputs, labels)

    # shuffle=True is crucial - it prevents the model from memorizing the order of games
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def train(npz_path, epochs=10, batch_size=32, learning_rate=0.001):
    # 1. Setup Device (GPU if available, else CPU)
    # MPS is for Mac (Metal Performance Shaders), CUDA for Nvidia
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 2. Load Data
    train_loader = load_data(npz_path, batch_size)

    # 3. Init model
    model = ChessNet().to(device)

    # Load existing weights if they exist (Resume Training) ---
    model_path = "models/chess_model.pth"
    if os.path.exists(model_path):
        print("Loading existing model to continue training...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Starting training from scratch...")
    
    # 4. Setup Optimizer & Scheduler
    # CrossEntropyLoss is standard for classification (pick 1 out of 64 squares)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Every 5 epochs, cut the learning rate in half (0.001 -> 0.0005 -> 0.00025)
    # This helps the model settle on precise answers (d2 vs c2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 5. Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train() # set mode (enables Batch Norm / Dropout)
        total_loss = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data to GPU/CPU
            data = data.to(device)
            targets = targets.to(device)

            # Reset gradients (PyTorch accumulates them by default)
            optimizer.zero_grad()

            # --- Forward Pass ---
            # out_from, out_to are the 64-sized vectors of raw scores
            out_from, out_to = model(data)

            # --- Loss Calculation ---
            # targets[:, 0] is the true "from" square
            # targets[:, 1] is the true "to" square
            loss_from = criterion(out_from, targets[:, 0])
            loss_to = criterion(out_to, targets[:, 1])

            # Total Loss is just the sum
            loss = loss_from + loss_to

            # --- Backward Pass ---
            loss.backward() # calc gradients
            optimizer.step() # update weights

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    # 6. Save the trained brain
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/chess_model.pth")
    print("Model saved to models/chess_model.pth")

if __name__ == "__main__":
    train("data/processed/chess_dataset.npz", epochs=15)