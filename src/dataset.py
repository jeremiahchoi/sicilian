import numpy as np
import torch
from torch.utils.data import Dataset

class ChessDataset(Dataset):
    def __init__(self, npz_file):
        """
        Loads the processed training data from an .npz file.
        Expected keys: 'inputs', 'policy', 'value'
        """
        try:
            print(f"Loading dataset from {npz_file}...")
            data = np.load(npz_file)
            
            # Inputs: (N, 18, 8, 8)
            self.inputs = torch.from_numpy(data['inputs']).float()
            
            # Policy Target: (N,) indices of the move played
            self.policy = torch.from_numpy(data['policy']).long()
            
            # Value Target: (N,) Outcome (-1.0 to 1.0)
            self.value = torch.from_numpy(data['value']).float()
            
            print(f"✅ Loaded {len(self.inputs)} samples.")
            
        except FileNotFoundError:
            print(f"❌ Error: File {npz_file} not found.")
            # Create dummy data to prevent immediate crash during debugging
            self.inputs = torch.zeros(1, 18, 8, 8)
            self.policy = torch.zeros(1).long()
            self.value = torch.zeros(1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.policy[idx], self.value[idx]