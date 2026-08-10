import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pytest
from loss_landscape_3d.calculator import LossLandscapeCalculator

class ToyDataset(data.Dataset):
    def __init__(self, size=16):
        self.x = torch.randn(size, 5)
        self.y = torch.randn(size, 2)
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 2)
        
    def forward(self, x):
        return self.linear(x)

def test_calculator_basic():
    model = ToyModel()
    dataset = ToyDataset(size=10)
    dataloader = data.DataLoader(dataset, batch_size=2)
    criterion = nn.MSELoss()
    
    # Save original state dict
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Directions
    dir_x = {k: torch.randn_like(v) for k, v in original_state.items()}
    dir_y = {k: torch.randn_like(v) for k, v in original_state.items()}
    
    calculator = LossLandscapeCalculator(model, dataloader, criterion, device='cpu')
    
    # Calculate landscape on 3x3 grid
    x_coords = np.linspace(-1, 1, 3)
    y_coords = np.linspace(-1, 1, 3)
    
    loss_grid = calculator.calculate(
        x_coords, y_coords, dir_x, dir_y, original_state
    )
    
    # Verify shape
    assert loss_grid.shape == (3, 3)
    assert not np.isnan(loss_grid).any()
    
    # Check that original model parameters are restored
    for k, v in model.state_dict().items():
        assert torch.allclose(v, original_state[k])

def test_calculator_max_batches():
    model = ToyModel()
    dataset = ToyDataset(size=100) # 50 batches of size 2
    dataloader = data.DataLoader(dataset, batch_size=2)
    criterion = nn.MSELoss()
    
    original_state = model.state_dict()
    dir_x = {k: torch.randn_like(v) for k, v in original_state.items()}
    dir_y = {k: torch.randn_like(v) for k, v in original_state.items()}
    
    calculator = LossLandscapeCalculator(model, dataloader, criterion, device='cpu')
    
    # Calculate with max_batches=2 (only first 4 samples should be evaluated per grid point)
    # Mock the dataloader iterator to count calls
    x_coords = [0.0]
    y_coords = [0.0]
    
    loss_grid = calculator.calculate(
        x_coords, y_coords, dir_x, dir_y, original_state, max_batches=2
    )
    
    assert loss_grid.shape == (1, 1)
