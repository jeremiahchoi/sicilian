import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # === 1. Convolutional Layers (The "Eyes") ===
        # Input: 13 channels (our board representation)
        # We use padding=1 to keep the board size 8x8 throughout these layers.
        
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=128, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(128)
        
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        # === 2. Fully Connected Layers (The "Decision Maker") ===
        # Flatten input: 128 channels * 8 * 8 = 8192 features
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 4096) # Output: 4096 probabilities (64 from * 64 to)

    def forward(self, x):
        # Layer 1
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Layer 2
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Layer 3
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten: (Batch_Size, 128, 8, 8) -> (Batch_Size, 8192)
        x = x.view(-1, 128 * 8 * 8)
        
        # Dense Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # No Softmax here! CrossEntropyLoss handles it.
        
        return x