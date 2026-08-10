# Step-by-Step Integration Guide: Loss Landscape 3D Visualizer

This guide takes you through the step-by-step integration of `loss-landscape-3d` into a PyTorch training pipeline. We will build a complete, runnable script that trains a small Multi-Layer Perceptron (MLP) on a synthetic classification dataset, tracks its parameter trajectory, projects it onto a 2D PCA subspace, evaluates the loss landscape grid, and renders interactive 3D charts.

---

## Step 1: Prepare the Model, Dataset, and Optimizer

First, define your neural network, dataset, loss function (criterion), and optimizer. The optimizer will navigate the high-dimensional weight space toward a local minimum.

```python
from loss_landscape_3d import torch, nn, optim, data

# A simple 2-layer MLP for classification
class ClassifierMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 2)
        )
    def forward(self, x):
        return self.net(x)

# Create synthetic dataset & DataLoader
# Using synthetic data allows this tutorial to run instantly anywhere without internet
class SyntheticClassification(data.Dataset):
    def __init__(self, num_samples=128):
        self.x = torch.randn(num_samples, 8)
        # Class targets (0 or 1)
        self.y = torch.randint(0, 2, (num_samples,))
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

model = ClassifierMLP()
dataset = SyntheticClassification()
dataloader = data.DataLoader(dataset, batch_size=16, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
```

---

## Step 2: Setup Trajectory Tracking in the Training Loop

Wrap your training loop with `TrajectoryTracker`. We save the model weights (checkpoints) and corresponding losses at specific intervals (e.g., at the end of each epoch or every few training steps). 

> [!IMPORTANT]
> To save memory and prevent model mutations from corrupting our history, the tracker clones all parameters and transfers them to the CPU.

```python
from loss_landscape_3d import TrajectoryTracker

# Initialize tracker (save checkpoints in-memory for this tutorial)
tracker = TrajectoryTracker()

# Record the initial (randomly initialized) state of the weights before training
tracker.save_checkpoint(model, step_or_epoch=0)
trajectory_losses = []

# Helper to calculate and record validation/training loss at each step
def evaluate_current_loss():
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / max(total_samples, 1)

trajectory_losses.append(evaluate_current_loss())

# Run Training Loop
model.train()
step = 1
for epoch in range(10): # 10 epochs
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Save checkpoints and compute loss every 2 steps
        if step % 2 == 0:
            tracker.save_checkpoint(model, step_or_epoch=step)
            trajectory_losses.append(evaluate_current_loss())
            model.train() # restore train mode
            
        step += 1
```

---

## Step 3: Generate PCA Projection Directions

Because neural network weights reside in a space of thousands or millions of dimensions, we must project them onto a 2D plane to visualize them. 
We run Principal Component Analysis (PCA) on the sequence of weight checkpoints. The two top eigenvectors define a 2D plane capturing the directions of maximum optimizer movement.

```python
from loss_landscape_3d import generate_pca_directions

# Retrieve all tracked weight state dicts
checkpoints = tracker.get_checkpoints()

# Extract PCA directions, center state, and projection coordinates
# We center the coordinate system on the final trained parameters (center_index=-1)
dir_x, dir_y, center_state, trajectory_coords = generate_pca_directions(
    checkpoints, 
    center_index=-1, 
    center_on_mean=False,
    normalize=False # Keep normalization False for exact trajectory projections
)
```

---

## Step 4: Calculate the Loss Landscape Surface Grid

Now, we evaluate the loss value for a grid of points on the 2D coordinate plane we defined. The model parameters at coordinate $(x, y)$ are perturbed as:
$$\theta(x, y) = \theta^* + x \cdot d_x + y \cdot d_y$$
where $\theta^*$ is our final trained weights (`center_state`). We sweep through a grid of $(x, y)$ values to compute the loss values.

```python
import numpy as np
from loss_landscape_3d import LossLandscapeCalculator

# Determine grid boundaries based on the optimizer's path with a buffer padding
xs = [coord[0] for coord in trajectory_coords]
ys = [coord[1] for coord in trajectory_coords]
x_min, x_max = min(xs) - 0.5, max(xs) + 0.5
y_min, y_max = min(ys) - 0.5, max(ys) + 0.5

# Define a 20x20 grid resolution
x_coords = np.linspace(x_min, x_max, 20)
y_coords = np.linspace(y_min, y_max, 20)

# Instantiate the calculator and evaluate the grid
calculator = LossLandscapeCalculator(model, dataloader, criterion)
loss_grid = calculator.calculate(x_coords, y_coords, dir_x, dir_y, center_state)
```

---

## Step 5: Render and Save the Visualizations

Pass the coordinate ranges, the loss grid, and the trajectory data to `LossLandscapeVisualizer`. We can export:
1.  **Plotly 3D Interactive Plot**: A gorgeous self-contained HTML page. Drag, rotate, and zoom the "mountain" of loss and hover over points to see step-by-step progress.
2.  **Matplotlib 2D Contour Plot**: A standard contour map that avoids 3D perspective occlusion, clearly showing the path's curves.

```python
from loss_landscape_3d import LossLandscapeVisualizer

visualizer = LossLandscapeVisualizer(x_coords, y_coords, loss_grid)

# 1. Save Interactive 3D HTML Plotly Chart
visualizer.plot_3d_plotly(
    trajectory_coords=trajectory_coords,
    trajectory_losses=trajectory_losses,
    title="Interactive 3D Loss Landscape & Optimizer Descent",
    theme="dark",
    save_path="loss_landscape_3d.html"
)
print("Interactive 3D plot saved to 'loss_landscape_3d.html'!")

# 2. Save 2D Contour Plot
visualizer.plot_contour_matplotlib(
    trajectory_coords=trajectory_coords,
    title="Loss Landscape Contours & Optimization Path",
    save_path="loss_landscape_contours.png"
)
print("Contour plot saved to 'loss_landscape_contours.png'!")
```

---

## Complete Runnable Script

Here is the complete, unified script compiling all the steps above. You can copy this script, save it as `run_pipeline.py`, and run it:

```python
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Safe for headless environments

from loss_landscape_3d import (
    torch,
    nn,
    optim,
    data,
    TrajectoryTracker,
    generate_pca_directions,
    LossLandscapeCalculator,
    LossLandscapeVisualizer
)

# 1. Simple Multi-Layer Perceptron (MLP)
class ClassifierMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 2)
        )
    def forward(self, x):
        return self.net(x)

# 2. Synthetic dataset
class SyntheticClassification(data.Dataset):
    def __init__(self, num_samples=128):
        self.x = torch.randn(num_samples, 8)
        self.y = torch.randint(0, 2, (num_samples,))
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

def run_visualization_pipeline():
    print("Initializing dataset and model...")
    model = ClassifierMLP()
    dataset = SyntheticClassification()
    dataloader = data.DataLoader(dataset, batch_size=16, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    
    # 3. Track training trajectory
    print("Training model and tracking parameter checkpoints...")
    tracker = TrajectoryTracker()
    tracker.save_checkpoint(model, step_or_epoch=0)
    
    trajectory_losses = []
    
    def eval_loss():
        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for x, y in dataloader:
                loss = criterion(model(x), y)
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
        return total_loss / max(total_samples, 1)

    trajectory_losses.append(eval_loss())
    
    model.train()
    step = 1
    for epoch in range(15):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if step % 2 == 0:
                tracker.save_checkpoint(model, step_or_epoch=step)
                trajectory_losses.append(eval_loss())
                model.train()
                
            step += 1
            
    print(f"Recorded {len(tracker.get_checkpoints())} checkpoints.")
    
    # 4. Compute PCA projection directions
    print("Computing optimized Gram SVD PCA projections...")
    checkpoints = tracker.get_checkpoints()
    dir_x, dir_y, center_state, trajectory_coords = generate_pca_directions(
        checkpoints, center_index=-1
    )
    
    # 5. Evaluate loss landscape
    print("Evaluating 2D loss grid...")
    xs = [coord[0] for coord in trajectory_coords]
    ys = [coord[1] for coord in trajectory_coords]
    x_min, x_max = min(xs) - 0.4, max(xs) + 0.4
    y_min, y_max = min(ys) - 0.4, max(ys) + 0.4
    
    x_coords = np.linspace(x_min, x_max, 25)
    y_coords = np.linspace(y_min, y_max, 25)
    
    calculator = LossLandscapeCalculator(model, dataloader, criterion)
    loss_grid = calculator.calculate(x_coords, y_coords, dir_x, dir_y, center_state)
    
    # 6. Visualize plots
    print("Generating visualizations...")
    visualizer = LossLandscapeVisualizer(x_coords, y_coords, loss_grid)
    
    # Interactive HTML
    visualizer.plot_3d_plotly(
        trajectory_coords=trajectory_coords,
        trajectory_losses=trajectory_losses,
        title="Optimizer Descent Path on Loss Landscape",
        theme="dark",
        save_path="loss_landscape_3d.html"
    )
    
    # Static PNG contours
    visualizer.plot_contour_matplotlib(
        trajectory_coords=trajectory_coords,
        title="Loss Landscape Contours & Path",
        save_path="loss_landscape_contours.png"
    )
    print("Success! Created 'loss_landscape_3d.html' and 'loss_landscape_contours.png'.")

if __name__ == "__main__":
    run_visualization_pipeline()
```

---

## How to Interpret the Topography

When you load your `loss_landscape_3d.html` file into a web browser, zoom and rotate the 3D graph. Look for the following features:

1.  **Valleys and Basins**: The deep blue/purple sections are regions of low loss (local/global minima). A good training path starts in high yellow/green regions and lands in a basin.
2.  **Optimizer Behavior**:
    *   **SGD with Momentum**: Notice the path curved smoothly down the hill. Too high a learning rate will cause the path to oscillate wildly from side to side in narrow valleys.
    *   **Adam**: It might adaptively navigate around saddle points or steep ridges more direct than vanilla SGD.
3.  **Saddle Points**: Look for regions that slope up in one direction but down in another (like a horse saddle). Optimizers often slow down in these flat regions before finding the escape path.
4.  **Local Minima**: Look for "craters" separated by ridges of high loss. If you run multiple training sessions with different seeds, you can plot their end-points to compare which minima are flatter. (Flatter minima are generally associated with better generalization).

---

## 🔬 Advanced Diagnostics & Features

The library also exposes advanced features for deeper research into optimization curvature and mode connectivity:

### 1. 1D Path Interpolation (Linear Mode Connectivity)
Plot the loss and accuracy barriers between two different training states:

```python
import numpy as np
from loss_landscape_3d import LossLandscapeCalculator, LossLandscapeVisualizer

# Define two model parameter checkpoints
state_1 = tracker.get_checkpoints()[0]   # Start
state_2 = tracker.get_checkpoints()[-1]  # End

# Calculate 1D interpolation
calculator = LossLandscapeCalculator(model, dataloader, criterion)
alphas = np.linspace(-0.2, 1.2, 20)

# Track loss along the path
losses = calculator.calculate_1d_path(alphas, state_1, state_2)

# Plot the 1D path curve
visualizer = LossLandscapeVisualizer(None, None, None)
fig = visualizer.plot_1d_matplotlib(
    alphas, losses, value_name="Loss", title="1D Loss Interpolation Path", save_path="lmc_path.png"
)
```

### 2. Auto-Suggesting Step Limits (Adaptive Bounds)
Avoid flat plateaus or exploded values by automatically searching for step limits:

```python
# Probes along the direction axes to suggest coordinate limits
x_limit, y_limit = calculator.suggest_coordinate_bounds(
    dir_x, dir_y, center_state, target_loss_factor=5.0
)

# Use the recommended bounds to define grid coords
x_coords = np.linspace(-x_limit, x_limit, 15)
y_coords = np.linspace(-y_limit, y_limit, 15)
```

### 3. Hessian Eigenvector Directions (Sharpness Visualization)
Compute directions corresponding to the largest second derivatives (highest curvature) at a local minimum:

```python
from loss_landscape_3d import generate_hessian_directions

# Compute eigenvectors corresponding to sharpest curvature
dir_x, dir_y, center = generate_hessian_directions(
    model, dataloader, criterion, max_batches=5, max_iter=15
)

# Evaluate and visualize the worst-case sharpness landscape
loss_grid = calculator.calculate(x_coords, y_coords, dir_x, dir_y, center)
```
