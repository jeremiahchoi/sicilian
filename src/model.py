import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A standard ResNet block: Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    This allows the model to learn 'residuals' (changes) rather than full transformations,
    making deep training stable.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # Skip Connection
        out = F.relu(out)
        return out

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # --- INPUT LAYER ---
        # 18 Channels (Relative Board + Rights + En Passant + Ghost Layer)
        # We output 128 features (Filters)
        self.conv_input = nn.Conv2d(18, 128, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(128)
        
        # --- RESIDUAL TOWER ---
        # 4 Blocks is a good starting point (AlphaZero used ~20-40 blocks)
        self.res_tower = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        
        # --- POLICY HEAD (Move Prediction) ---
        # Reduces depth to 2 channels, then flattens
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1) 
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096) # 4096 = 64*64 moves
        
        # --- VALUE HEAD (Win Probability) ---
        # Reduces depth to 1 channel
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # 1. Input Processing
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # 2. Residual Tower
        x = self.res_tower(x)
        
        # 3. Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 2 * 8 * 8) # Flatten
        policy_logits = self.policy_fc(p)
        
        # 4. Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 8 * 8) # Flatten
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)) # Output between -1 and 1
        
        return policy_logits, value