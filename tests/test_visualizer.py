import os
import tempfile
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pytest
from loss_landscape_3d.visualizer import LossLandscapeVisualizer

def test_visualizer_plotly():
    # Make dummy grid
    x_coords = np.linspace(-1, 1, 5)
    y_coords = np.linspace(-1, 1, 5)
    loss_grid = np.random.randn(5, 5)
    
    visualizer = LossLandscapeVisualizer(x_coords, y_coords, loss_grid)
    
    # Trajectory data
    trajectory_coords = [(0.5, 0.5), (0.0, 0.0)]
    trajectory_losses = [1.2, 0.4]
    
    # Test plotly figure creation
    fig = visualizer.plot_3d_plotly(
        trajectory_coords=trajectory_coords,
        trajectory_losses=trajectory_losses,
        title="Test Plotly Landscape",
        theme='dark'
    )
    
    # Verify trace counts (1 surface + 1 trajectory line + 1 start + 1 end = 4 traces)
    assert len(fig.data) == 4
    assert fig.layout.title.text == "Test Plotly Landscape"

    # Test saving plotly to HTML
    temp_dir = tempfile.mkdtemp()
    try:
        html_path = os.path.join(temp_dir, "plot.html")
        visualizer.plot_3d_plotly(
            trajectory_coords=trajectory_coords,
            trajectory_losses=trajectory_losses,
            save_path=html_path
        )
        assert os.path.exists(html_path)
        assert os.path.getsize(html_path) > 0
    finally:
        shutil.rmtree(temp_dir)

def test_visualizer_matplotlib():
    x_coords = np.linspace(-1, 1, 5)
    y_coords = np.linspace(-1, 1, 5)
    loss_grid = np.random.randn(5, 5)
    
    visualizer = LossLandscapeVisualizer(x_coords, y_coords, loss_grid)
    trajectory_coords = [(0.5, 0.5), (0.0, 0.0)]
    trajectory_losses = [1.2, 0.4]
    
    # 3D Matplotlib
    fig_3d = visualizer.plot_3d_matplotlib(
        trajectory_coords=trajectory_coords,
        trajectory_losses=trajectory_losses,
        title="Test Matplotlib Landscape"
    )
    assert fig_3d is not None
    
    # 2D Contour Matplotlib
    fig_2d = visualizer.plot_contour_matplotlib(
        trajectory_coords=trajectory_coords,
        title="Test Contour Landscape"
    )
    assert fig_2d is not None

    # Test saving matplotlib images
    temp_dir = tempfile.mkdtemp()
    try:
        img_3d_path = os.path.join(temp_dir, "plot_3d.png")
        img_2d_path = os.path.join(temp_dir, "plot_2d.png")
        
        visualizer.plot_3d_matplotlib(
            trajectory_coords=trajectory_coords,
            trajectory_losses=trajectory_losses,
            save_path=img_3d_path
        )
        visualizer.plot_contour_matplotlib(
            trajectory_coords=trajectory_coords,
            save_path=img_2d_path
        )
        
        assert os.path.exists(img_3d_path)
        assert os.path.exists(img_2d_path)
        assert os.path.getsize(img_3d_path) > 0
        assert os.path.getsize(img_2d_path) > 0
    finally:
        shutil.rmtree(temp_dir)
