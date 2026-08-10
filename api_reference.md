# loss-landscape-3d API Reference

This document provides a comprehensive guide to all classes, methods, functions, and exposed namespaces in the `loss-landscape-3d` library.

---

## 📦 Exposed Namespaces

For convenience, the library re-exports core PyTorch modules. You do not need separate import statements for standard modeling and data loading utilities.

```python
from loss_landscape_3d import torch, nn, optim, data
```

*   `torch`: The core PyTorch library namespace.
*   `nn`: PyTorch Neural Network module (`torch.nn`), containing layers (`nn.Linear`, `nn.Conv2d`) and loss functions (`nn.CrossEntropyLoss`, `nn.MSELoss`).
*   `optim`: PyTorch Optimizers (`torch.optim`), containing algorithms (`optim.SGD`, `optim.Adam`).
*   `data`: PyTorch Utilities Data (`torch.utils.data`), containing data management classes (`data.Dataset`, `data.DataLoader`).

---

## 🛠️ Public Functions

### 1. `generate_random_directions`
Generates two random parameter directions (represented as PyTorch state dicts) for a given model. These directions define the 2D coordinate plane used to perturb the model's parameters.

#### Signature:
```python
def generate_random_directions(model, normalize=True):
```

#### Parameters:
*   `model` (*torch.nn.Module*): The PyTorch model to perturb.
*   `normalize` (*bool*, default=`True`): If `True`, applies filter-wise normalization to both generated directions relative to the model's weight magnitudes. Recommended to enable to address scale-invariance.

#### Returns:
*   *tuple of (dict, dict)*: A pair of state dicts `(dir_x, dir_y)` containing direction tensors matching the keys and shapes of `model.state_dict()`.

---

### 2. `generate_pca_directions`
Extracts two principal components from a recorded training trajectory (list of parameter checkpoints) using an optimized Gram SVD solver. These directions represent the axes of maximum variation along the optimization path.

#### Signature:
```python
def generate_pca_directions(checkpoints, center_index=-1, center_on_mean=False, normalize=False):
```

#### Parameters:
*   `checkpoints` (*list of dict*): A list of parameter state dicts recorded during training (typically from `TrajectoryTracker.get_checkpoints()`). At least 2 checkpoints are required.
*   `center_index` (*int*, default=`-1`): The list index of the checkpoint to use as the center of coordinate projection. Default is `-1` (final weights / local minimum).
*   `center_on_mean` (*bool*, default=`False`): If `True`, centers the projection coordinates at the mean parameter state of the trajectory instead of a specific checkpoint.
*   `normalize` (*bool*, default=`False`): If `True`, applies filter-wise normalization to the resulting PCA directions. Default is `False` because scaling PCA directions can warp the trajectory subspace and distort path projections.

#### Returns:
*   *tuple*: A tuple containing:
    1.  `dir_x` (*dict*): Direction state dict for the x-axis.
    2.  `dir_y` (*dict*): Direction state dict for the y-axis.
    3.  `center_state_dict` (*dict*): The state dict corresponding to the center coordinate $(0, 0)$.
    4.  `trajectory_coords` (*list of tuple*): A list of 2D coordinates `[(x_0, y_0), (x_1, y_1), ...]` representing the projected path of each checkpoint on the 2D plane.

---

### 3. `filter_wise_normalize`
Applies filter-wise (for 4D Conv weights) or neuron-wise (for 2D Linear weights) normalization to a direction vector relative to a reference model parameter state dict.

#### Signature:
```python
def filter_wise_normalize(direction, weights):
```

#### Parameters:
*   `direction` (*dict*): The direction state dict containing perturbation tensors to normalize.
*   `weights` (*dict*): The reference state dict (e.g. final model weights) used to determine norms.

#### Returns:
*   `normalized_dir` (*dict*): The normalized direction state dict.

---

## 🗄️ Public Classes

### 1. `TrajectoryTracker`
Tracks and saves model parameter checkpoints during training. It can store checkpoints in memory or directly to a folder on disk to preserve GPU/system RAM.

#### Constructor:
```python
def __init__(self, dir_path=None):
```
*   `dir_path` (*str*, optional): The folder path to save checkpoints on disk as `.pt` files. If `None` (default), checkpoints are cached in memory.

#### Methods:
*   `save_checkpoint(model, step_or_epoch=None)`:
    Clones the model's current weights, moves them to the CPU, and saves the checkpoint.
    *   `model` (*torch.nn.Module*): The model to record parameters from.
    *   `step_or_epoch` (*int*, optional): A numerical identifier (e.g. training step). If `None`, an auto-incrementing counter is used.
*   `get_checkpoints()`:
    Returns the list of recorded parameter state dicts, sorted numerically by their step/epoch numbers.
*   `clear()`:
    Deletes all recorded checkpoints (both from RAM and disk) and resets counters.

---

### 2. `LossLandscapeCalculator`
Evaluates the loss function on a grid of parameter perturbations.

#### Constructor:
```python
def __init__(self, model, dataloader, criterion, device=None):
```
*   `model` (*torch.nn.Module*): The PyTorch model to evaluate.
*   `dataloader` (*torch.utils.data.DataLoader*): The dataset loader for computing loss.
*   `criterion` (*callable*): The loss function (e.g., `nn.CrossEntropyLoss()`).
*   `device` (*torch.device* or *str*, optional): The evaluation device ('cuda' or 'cpu'). If `None`, it is auto-detected from the model's parameters.

#### Methods:
*   `calculate(x_coords, y_coords, dir_x, dir_y, center_state, max_batches=None)`:
    Runs grid perturbations under `torch.no_grad()`, loading and restoring model states automatically.
    *   `x_coords` (*array-like*): 1D array of coordinates for the x-axis grid.
    *   `y_coords` (*array-like*): 1D array of coordinates for the y-axis grid.
    *   `dir_x` (*dict*): The state dict containing the x-axis direction vector.
    *   `dir_y` (*dict*): The state dict containing the y-axis direction vector.
    *   `center_state` (*dict*): The center state dict (e.g., final weights) corresponding to grid origin $(0,0)$.
    *   `max_batches` (*int*, optional): Limits the number of data batches evaluated per grid point to accelerate calculation on large datasets.
    *   *Returns*: A 2D NumPy array of shape `(len(x_coords), len(y_coords))` containing the loss values.

---

### 3. `LossLandscapeVisualizer`
Renders interactive and static 3D plots of the loss landscape, with support for optimizer path overlays.

#### Constructor:
```python
def __init__(self, x_coords, y_coords, loss_grid):
```
*   `x_coords` (*array-like*): 1D array of grid coordinates for the x-axis.
*   `y_coords` (*array-like*): 1D array of grid coordinates for the y-axis.
*   `loss_grid` (*np.ndarray*): The 2D grid of loss values (from `LossLandscapeCalculator.calculate`).

#### Methods:
*   `plot_3d_plotly(trajectory_coords=None, trajectory_losses=None, title="Loss Landscape 3D", theme='dark', save_path=None)`:
    Renders an interactive 3D surface plot using Plotly.
    *   `trajectory_coords` (*list of tuple*, optional): List of $(x, y)$ coordinate points representing the optimizer's path.
    *   `trajectory_losses` (*list of float*, optional): Loss values corresponding to each trajectory point.
    *   `title` (*str*, default=`"Loss Landscape 3D"`): Title of the plot.
    *   `theme` (*str*, default=`'dark'`): Visual theme, `'dark'` or `'light'`.
    *   `save_path` (*str*, optional): Filepath to save the interactive HTML graph.
    *   *Returns*: A `plotly.graph_objects.Figure` object.
*   `plot_3d_matplotlib(trajectory_coords=None, trajectory_losses=None, title="Loss Landscape 3D", theme='light', save_path=None)`:
    Renders a static 3D surface plot using Matplotlib. Safe for headless environments (automatically falls back to `'Agg'` backend if Tkinter GUI fails to initialize).
    *   `trajectory_coords` / `trajectory_losses` / `title` / `save_path`: Same as above.
    *   `theme` (*str*, default=`'light'`): Visual style, `'light'` or `'dark'`.
    *   *Returns*: A `matplotlib.figure.Figure` object.
*   `plot_contour_matplotlib(trajectory_coords=None, levels=25, title="Loss Landscape Contours", theme='light', save_path=None)`:
    Renders a 2D contour map of the loss landscape using Matplotlib, useful for seeing trajectory curves without 3D occlusion.
    *   `trajectory_coords` (*list of tuple*, optional): Optimizer path coordinates to overlay.
    *   `levels` (*int*, default=`25`): Resolution level density of contour lines.
    *   `title` / `theme` / `save_path`: Same as above.
    *   *Returns*: A `matplotlib.figure.Figure` object.
