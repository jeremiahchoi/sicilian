import pytest
import torch
from src.model import ChessNet

def test_model_dimensions():
    """Does the model accept a board and output 2 vectors of size 64?"""
    model = ChessNet()
    
    # Create a fake batch of data
    # Batch Size = 1, Channels = 12, Height = 8, Width = 8
    dummy_input = torch.randn(1, 12, 8, 8)
    
    # Push it through the model
    from_logits, to_logits = model(dummy_input)
    
    # Check output shapes
    # We expect (1, 64) because batch_size=1 and there are 64 squares
    assert from_logits.shape == (1, 64)
    assert to_logits.shape == (1, 64)

def test_batch_processing():
    """Can it handle a batch of 10 games at once?"""
    model = ChessNet()
    batch_size = 10
    dummy_input = torch.randn(batch_size, 12, 8, 8)
    
    from_logits, to_logits = model(dummy_input)
    
    assert from_logits.shape == (batch_size, 64)
    assert to_logits.shape == (batch_size, 64)