import os
import tempfile
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from loss_landscape_3d import (
    TrajectoryTracker,
    generate_pca_directions,
    LossLandscapeCalculator,
    LossLandscapeVisualizer
)

# 1. Simple Multi-Layer Perceptron (MLP)
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# 2. Synthetic regression dataset
class SyntheticDataset(data.Dataset):
    def __init__(self, size=32):
        self.x = torch.randn(size, 4)
        self.y = torch.randn(size, 1)
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def test_end_to_end_trajectory_and_landscape():
    # Set seed
    torch.manual_seed(123)
    np.random.seed(123)
    
    # Initialize components
    model = TinyMLP()
    dataset = SyntheticDataset(size=64)
    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    tracker = TrajectoryTracker()
    
    # Track the initial state
    tracker.save_checkpoint(model, step_or_epoch=0)
    
    trajectory_losses = []
    # Evaluate initial loss
    initial_loss = 0.0
    with torch.no_grad():
        for x_b, y_b in dataloader:
            initial_loss += criterion(model(x_b), y_b).item() * x_b.size(0)
    trajectory_losses.append(initial_loss / len(dataset))
    
    # Train the model for 5 steps and record state dicts & losses
    model.train()
    step = 1
    for epoch in range(1):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Save checkpoint
            tracker.save_checkpoint(model, step_or_epoch=step)
            
            # Evaluate step loss (on training data for simplicity)
            model.eval()
            total_l = 0.0
            with torch.no_grad():
                for x_b, y_b in dataloader:
                    total_l += criterion(model(x_b), y_b).item() * x_b.size(0)
            trajectory_losses.append(total_l / len(dataset))
            model.train()
            
            step += 1
            if step > 5:
                break
        if step > 5:
            break
            
    checkpoints = tracker.get_checkpoints()
    assert len(checkpoints) == 6
    assert len(trajectory_losses) == 6
    
    # Generate PCA direction vectors from checkpoints
    dir_x, dir_y, center_state, trajectory_coords = generate_pca_directions(
        checkpoints, center_index=-1, center_on_mean=False, normalize=False
    )
    
    assert len(trajectory_coords) == 6
    
    # Evaluate loss landscape on a 5x5 grid
    # Grid coordinate range should cover the projected trajectory coordinate space
    # Find bounding box of trajectory coords to scale grid appropriately
    xs = [coord[0] for coord in trajectory_coords]
    ys = [coord[1] for coord in trajectory_coords]
    
    x_min, x_max = min(xs) - 0.2, max(xs) + 0.2
    y_min, y_max = min(ys) - 0.2, max(ys) + 0.2
    
    x_coords = np.linspace(x_min, x_max, 5)
    y_coords = np.linspace(y_min, y_max, 5)
    
    calculator = LossLandscapeCalculator(model, dataloader, criterion, device='cpu')
    loss_grid = calculator.calculate(
        x_coords, y_coords, dir_x, dir_y, center_state
    )
    
    assert loss_grid.shape == (5, 5)
    assert not np.isnan(loss_grid).any()
    
    # Visualize results
    visualizer = LossLandscapeVisualizer(x_coords, y_coords, loss_grid)
    
    temp_dir = tempfile.mkdtemp()
    try:
        plotly_html = os.path.join(temp_dir, "landscape_3d.html")
        matplotlib_png = os.path.join(temp_dir, "landscape_3d.png")
        contour_png = os.path.join(temp_dir, "landscape_contour.png")
        
        # Test plotly interactive 3D rendering
        fig_plotly = visualizer.plot_3d_plotly(
            trajectory_coords=trajectory_coords,
            trajectory_losses=trajectory_losses,
            title="Interactive Loss Landscape 3D",
            save_path=plotly_html
        )
        
        # Test static matplotlib rendering
        fig_mpl = visualizer.plot_3d_matplotlib(
            trajectory_coords=trajectory_coords,
            trajectory_losses=trajectory_losses,
            title="Static Loss Landscape 3D",
            save_path=matplotlib_png
        )
        
        # Test 2D contour rendering
        fig_contour = visualizer.plot_contour_matplotlib(
            trajectory_coords=trajectory_coords,
            title="Loss Landscape Contours",
            save_path=contour_png
        )
        
        # Assert file generation
        assert os.path.exists(plotly_html)
        assert os.path.exists(matplotlib_png)
        assert os.path.exists(contour_png)
        
        assert os.path.getsize(plotly_html) > 0
        assert os.path.getsize(matplotlib_png) > 0
        assert os.path.getsize(contour_png) > 0
        
    finally:
        shutil.rmtree(temp_dir)
