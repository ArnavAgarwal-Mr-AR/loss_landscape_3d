import torch
import torch.nn as nn
import pytest
from loss_landscape_3d.directions import (
    filter_wise_normalize,
    generate_random_directions,
    generate_pca_directions
)

class ConvMLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.fc = nn.Linear(18, 4)
        self.bias_only = nn.Parameter(torch.ones(4))
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x) + self.bias_only

def test_filter_wise_normalize():
    model = ConvMLPModel()
    weights = model.state_dict()
    
    # Generate random directions
    dir_x = {k: torch.randn_like(v) for k, v in weights.items()}
    
    # Run filter-wise normalization
    norm_dir = filter_wise_normalize(dir_x, weights)
    
    # 1. Check Conv2d weights (4D: out_channels, in_channels, height, width)
    # The norm of each output filter in norm_dir must match that in weights
    conv_w_ref = weights['conv.weight']
    conv_w_dir = norm_dir['conv.weight']
    for i in range(conv_w_ref.size(0)):
        ref_norm = torch.norm(conv_w_ref[i])
        dir_norm = torch.norm(conv_w_dir[i])
        assert torch.allclose(ref_norm, dir_norm, atol=1e-5)
        
    # 2. Check Linear weights (2D: out_features, in_features)
    # The norm of each neuron row in norm_dir must match that in weights
    fc_w_ref = weights['fc.weight']
    fc_w_dir = norm_dir['fc.weight']
    for i in range(fc_w_ref.size(0)):
        ref_norm = torch.norm(fc_w_ref[i])
        dir_norm = torch.norm(fc_w_dir[i])
        assert torch.allclose(ref_norm, dir_norm, atol=1e-5)
        
    # 3. Check Bias weights (1D)
    # The norm of the entire vector must match
    fc_b_ref = weights['fc.bias']
    fc_b_dir = norm_dir['fc.bias']
    assert torch.allclose(torch.norm(fc_b_ref), torch.norm(fc_b_dir), atol=1e-5)

def test_filter_wise_normalize_mismatch():
    weights = {'w': torch.randn(2, 2)}
    direction = {'w': torch.randn(3, 3)} # Mismatched shape
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        filter_wise_normalize(direction, weights)

def test_generate_random_directions():
    model = ConvMLPModel()
    dir_x, dir_y = generate_random_directions(model, normalize=True)
    
    # Keys should match
    assert dir_x.keys() == model.state_dict().keys()
    assert dir_y.keys() == model.state_dict().keys()
    
    # Norms should match weights
    weights = model.state_dict()
    assert torch.allclose(torch.norm(dir_x['fc.bias']), torch.norm(weights['fc.bias']), atol=1e-5)
    assert torch.allclose(torch.norm(dir_y['fc.bias']), torch.norm(weights['fc.bias']), atol=1e-5)

def test_generate_pca_directions():
    # Set seed for reproducible SVD / random values
    torch.manual_seed(42)
    
    model = ConvMLPModel()
    weights = model.state_dict()
    
    # Simulate a path of checkpoints (e.g. 5 steps of training)
    checkpoints = []
    for step in range(5):
        sd = {}
        for k, v in weights.items():
            if v.is_floating_point():
                # Add step-dependent perturbation
                sd[k] = v + 0.1 * step * torch.ones_like(v)
            else:
                sd[k] = v.clone()
        checkpoints.append(sd)
        
    # Generate PCA directions
    dir_x, dir_y, center, trajectory_coords = generate_pca_directions(
        checkpoints, center_index=-1, center_on_mean=False, normalize=False
    )
    
    # 1. Assert keys and types
    assert dir_x.keys() == weights.keys()
    assert dir_y.keys() == weights.keys()
    
    # 2. Check center matches the final checkpoint
    assert torch.allclose(center['fc.weight'], checkpoints[-1]['fc.weight'])
    
    # 3. Check orthogonality of direction vectors:
    # Flatten direction tensors and compute dot product
    flat_x = torch.cat([v.flatten() for k, v in dir_x.items() if v.is_floating_point()])
    flat_y = torch.cat([v.flatten() for k, v in dir_y.items() if v.is_floating_point()])
    
    dot_prod = torch.dot(flat_x, flat_y).item()
    assert abs(dot_prod) < 1e-4 # Orthogonal
    
    # Norms of principal components should be 1.0 (standard PCA components are normalized to unit length)
    assert torch.allclose(torch.norm(flat_x), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(torch.norm(flat_y), torch.tensor(1.0), atol=1e-5)
    
    # 4. Trajectory coordinates check
    assert len(trajectory_coords) == 5
    # The center_index checkpoint should have projection coordinates (0.0, 0.0)
    cx_final, cy_final = trajectory_coords[-1]
    assert abs(cx_final) < 1e-5
    assert abs(cy_final) < 1e-5

def test_generate_pca_directions_errors():
    # Less than 2 checkpoints
    with pytest.raises(ValueError, match="At least 2 checkpoints"):
        generate_pca_directions([{'w': torch.tensor([1.0])}])


def test_generate_hessian_directions():
    from loss_landscape_3d.directions import generate_hessian_directions
    from torch.utils.data import TensorDataset, DataLoader
    
    torch.manual_seed(42)
    model = ConvMLPModel()
    
    # Create simple dummy dataset
    x = torch.randn(8, 1, 3, 3)
    y = torch.randint(0, 4, (8,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=4)
    criterion = nn.CrossEntropyLoss()
    
    dir_x, dir_y, center = generate_hessian_directions(
        model, dataloader, criterion, max_batches=2, max_iter=5, tol=1e-2
    )
    
    # 1. Assert keys and types
    assert dir_x.keys() == model.state_dict().keys()
    assert dir_y.keys() == model.state_dict().keys()
    
    # 2. Check center weights
    assert torch.allclose(center['fc.weight'], model.state_dict()['fc.weight'])
    
    # 3. Check eigenvectors are unit norm (for trainable weights)
    trainable_keys = [k for k, p in model.named_parameters() if p.requires_grad]
    flat_x = torch.cat([dir_x[k].flatten() for k in trainable_keys])
    flat_y = torch.cat([dir_y[k].flatten() for k in trainable_keys])
    
    assert torch.allclose(torch.norm(flat_x), torch.tensor(1.0), atol=1e-3)
    assert torch.allclose(torch.norm(flat_y), torch.tensor(1.0), atol=1e-3)
    
    # Check orthogonality
    dot_prod = torch.dot(flat_x, flat_y).item()
    assert abs(dot_prod) < 1e-3
