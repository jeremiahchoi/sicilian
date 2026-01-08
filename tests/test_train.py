import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import ChessNet

def test_single_training_step():
    """
    Run one single optimization step and ensure weights change.
    This proves the 'learning' mechanism is connected correctly.
    """
    model = ChessNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Fake Batch: 5 games
    inputs = torch.randn(5, 12, 8, 8)
    
    # Fake Labels: 5 pairs of (from, to) squares (0-63)
    # torch.randint(low, high, shape)
    targets = torch.randint(0, 64, (5, 2))
    
    # 1. Check weights BEFORE update
    # We look at the first layer's weights
    weight_before = model.conv1.weight.clone()
    
    # 2. Run Forward/Backward
    optimizer.zero_grad()
    out_from, out_to = model(inputs)
    
    loss = criterion(out_from, targets[:, 0]) + criterion(out_to, targets[:, 1])
    loss.backward()
    optimizer.step()
    
    # 3. Check weights AFTER update
    weight_after = model.conv1.weight
    
    # 4. Assert they are DIFFERENT (meaning the model learned something)
    assert not torch.equal(weight_before, weight_after)