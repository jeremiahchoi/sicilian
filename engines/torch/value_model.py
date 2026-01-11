import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        
        # === 1. Convolutional Body (The "Eyes") ===
        # Same input as before: 13x8x8
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        # === 2. The Judgment Head ===
        # Flattening 128 channels * 8 * 8 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1) # Output: 1 single scalar

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(-1, 128 * 8 * 8)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Tanh squashes the output to be between -1 (Loss) and 1 (Win)
        return torch.tanh(x)