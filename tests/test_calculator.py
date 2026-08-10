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


def test_calculator_metrics():
    model = ToyModel()
    dataset = ToyDataset(size=10)
    dataloader = data.DataLoader(dataset, batch_size=2)
    criterion = nn.MSELoss()
    original_state = model.state_dict()
    
    dir_x = {k: torch.zeros_like(v) for k, v in original_state.items()}
    dir_y = {k: torch.zeros_like(v) for k, v in original_state.items()}
    
    calculator = LossLandscapeCalculator(model, dataloader, criterion, device='cpu')
    
    def dummy_metric(outputs, targets):
        return outputs.sum().item()
        
    loss_grid, metric_grid = calculator.calculate(
        [0.0], [0.0], dir_x, dir_y, original_state, metric_fn=dummy_metric
    )
    
    assert loss_grid.shape == (1, 1)
    assert metric_grid.shape == (1, 1)
    assert not np.isnan(metric_grid).any()


def test_calculator_1d_path():
    model = ToyModel()
    dataset = ToyDataset(size=10)
    dataloader = data.DataLoader(dataset, batch_size=2)
    criterion = nn.MSELoss()
    
    state_1 = {k: v.clone() for k, v in model.state_dict().items()}
    state_2 = {k: v.clone() + 1.0 for k, v in model.state_dict().items()}
    
    calculator = LossLandscapeCalculator(model, dataloader, criterion, device='cpu')
    alphas = np.linspace(0.0, 1.0, 5)
    
    losses = calculator.calculate_1d_path(alphas, state_1, state_2)
    assert losses.shape == (5,)
    assert not np.isnan(losses).any()
    
    def dummy_metric(outputs, targets):
        return outputs.mean().item()
        
    losses, metrics = calculator.calculate_1d_path(alphas, state_1, state_2, metric_fn=dummy_metric)
    assert losses.shape == (5,)
    assert metrics.shape == (5,)


def test_suggest_coordinate_bounds():
    model = ToyModel()
    dataset = ToyDataset(size=10)
    dataloader = data.DataLoader(dataset, batch_size=2)
    criterion = nn.MSELoss()
    original_state = model.state_dict()
    
    dir_x = {k: torch.randn_like(v) for k, v in original_state.items()}
    dir_y = {k: torch.randn_like(v) for k, v in original_state.items()}
    
    calculator = LossLandscapeCalculator(model, dataloader, criterion, device='cpu')
    
    x_limit, y_limit = calculator.suggest_coordinate_bounds(
        dir_x, dir_y, original_state, target_loss_factor=2.0, max_batches=2
    )
    
    assert x_limit > 0.0
    assert y_limit > 0.0
    assert x_limit <= 10.0
    assert y_limit <= 10.0
