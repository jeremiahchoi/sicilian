import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # --- Feature Extractor (The "Eye") ---
        # Input: 12 channels (the pieces), Output: 64 filters (patterns)
        # Kernel size 3 means it looks at 3x3 square blocks
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64) # Normalizes data to speed up training

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # --- The Decision Heads (The "Brain") ---
        # We flatten the 256x8x8 tensor into a 1D vector of size 16384
        self.fc_input_dim = 256 * 8 * 8
        
        # Head 1: Predict "From" Square (0-63)
        self.fc_from = nn.Linear(self.fc_input_dim, 64)
        
        # Head 2: Predict "To" Square (0-63)
        self.fc_to = nn.Linear(self.fc_input_dim, 64)

    def forward(self, x):
        # x shape: (Batch_Size, 12, 8, 8)
        
        # Layer 1: Conv -> Batch Norm -> ReLU (Activation)
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Layer 2
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Layer 3
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten: Turn the 3D cube into a 1D line so Linear layers can read it
        x = x.view(-1, self.fc_input_dim)
        
        # Calculate outputs
        out_from = self.fc_from(x) # Raw scores (logits) for "From" square
        out_to = self.fc_to(x)     # Raw scores (logits) for "To" square
        
        return out_from, out_to