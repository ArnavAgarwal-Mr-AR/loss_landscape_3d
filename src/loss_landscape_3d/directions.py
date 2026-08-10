import torch

def filter_wise_normalize(direction, weights):
    """
    Applies filter-wise normalization to a direction vector relative to the reference weights.
    Matches the norm of filters/neurons/vectors in the direction to the corresponding norms
    in the reference weights, addressing scale invariance issues in loss visualization.
    
    Args:
        direction (dict of str: torch.Tensor): The direction state dict to normalize.
        weights (dict of str: torch.Tensor): The reference weights state dict.
        
    Returns:
        dict of str: torch.Tensor: The normalized direction state dict.
    """
    normalized_dir = {}
    for k, d_tensor in direction.items():
        if k not in weights:
            normalized_dir[k] = d_tensor.clone()
            continue
            
        w_tensor = weights[k]
        
        # Only normalize floating point tensors
        if not d_tensor.is_floating_point():
            normalized_dir[k] = d_tensor.clone()
            continue
            
        if d_tensor.shape != w_tensor.shape:
            raise ValueError(f"Shape mismatch for key '{k}': direction shape {d_tensor.shape} != weights shape {w_tensor.shape}")
            
        d_norm = d_tensor.clone()
        
        # 4D tensor: Conv filters (out_channels, in_channels, height, width)
        if w_tensor.dim() == 4:
            for i in range(w_tensor.size(0)):
                w_f = w_tensor[i]
                d_f = d_tensor[i]
                norm_w = torch.norm(w_f)
                norm_d = torch.norm(d_f)
                if norm_d > 1e-10:
                    d_norm[i] = d_f * (norm_w / norm_d)
                else:
                    d_norm[i] = 0.0
                    
        # 2D tensor: Linear neuron weights (out_features, in_features)
        elif w_tensor.dim() == 2:
            for i in range(w_tensor.size(0)):
                w_r = w_tensor[i]
                d_r = d_tensor[i]
                norm_w = torch.norm(w_r)
                norm_d = torch.norm(d_r)
                if norm_d > 1e-10:
                    d_norm[i] = d_r * (norm_w / norm_d)
                else:
                    d_norm[i] = 0.0
                    
        # Other dimensions: biases, batch-norm scales/biases, 1D or 3D weights
        else:
            norm_w = torch.norm(w_tensor)
            norm_d = torch.norm(d_tensor)
            if norm_d > 1e-10:
                d_norm = d_tensor * (norm_w / norm_d)
            else:
                d_norm = torch.zeros_like(d_tensor)
                
        normalized_dir[k] = d_norm
        
    return normalized_dir


def generate_random_directions(model, normalize=True):
    """
    Generates two random parameter direction vectors (state dicts) for the given model.
    
    Args:
        model (torch.nn.Module): The PyTorch model.
        normalize (bool): If True, apply filter-wise normalization relative to the model's weights.
        
    Returns:
        tuple of (dict, dict): Two direction state dicts (dir_x, dir_y).
    """
    weights = model.state_dict()
    dir_x = {}
    dir_y = {}
    
    for k, v in weights.items():
        if v.is_floating_point():
            dir_x[k] = torch.randn_like(v)
            dir_y[k] = torch.randn_like(v)
        else:
            dir_x[k] = torch.zeros_like(v)
            dir_y[k] = torch.zeros_like(v)
            
    if normalize:
        dir_x = filter_wise_normalize(dir_x, weights)
        dir_y = filter_wise_normalize(dir_y, weights)
        
    return dir_x, dir_y


def generate_pca_directions(checkpoints, center_index=-1, center_on_mean=False, normalize=False):
    """
    Generates two projection directions by computing PCA on a trajectory of weights.
    Uses an optimized Gram-matrix SVD algorithm that is extremely fast for model weights.
    
    Args:
        checkpoints (list of dict): List of state dicts representing model checkpoints during training.
        center_index (int): Index of the checkpoint to use as the center of coordinate projection.
                            Defaults to -1 (the final trained weights).
        center_on_mean (bool): If True, the coordinate system is centered at the mean of the trajectory.
        normalize (bool): If True, apply filter-wise normalization to the resulting PCA directions.
                          (Usually False for trajectory plots to keep projection exact).
                          
    Returns:
        tuple: (dir_x, dir_y, center_state_dict, trajectory_coords)
            dir_x (dict): Direction state dict for x-axis.
            dir_y (dict): Direction state dict for y-axis.
            center_state_dict (dict): State dict at the center coordinate.
            trajectory_coords (list of tuple): List of (x, y) coordinates representing each checkpoint.
    """
    if len(checkpoints) < 2:
        raise ValueError("At least 2 checkpoints are required to compute PCA directions.")
        
    first_sd = checkpoints[0]
    param_keys = [k for k, v in first_sd.items() if v.is_floating_point()]
    
    if not param_keys:
        raise ValueError("No floating point parameters found in checkpoints to compute PCA.")
        
    # Flatten weights for each checkpoint
    M = len(checkpoints)
    flat_checkpoints = []
    for sd in checkpoints:
        tensors = [sd[k].flatten() for k in param_keys]
        flat_checkpoints.append(torch.cat(tensors))
    
    W = torch.stack(flat_checkpoints)  # Shape (M, N)
    N = W.shape[1]
    
    # Compute mean of trajectory
    mean_W = W.mean(dim=0, keepdim=True)  # Shape (1, N)
    
    # Compute Gram matrix K = X @ X.T (Shape M x M)
    X = W - mean_W
    K = torch.matmul(X, X.t())
    
    # Solve Eigendecomposition of K: K = U * L * U^T
    eigenvalues, U = torch.linalg.eigh(K)
    
    # Sort descending
    eigenvalues = torch.flip(eigenvalues, dims=[0])
    U = torch.flip(U, dims=[1])
    
    # Extract singular values (sqrt of eigenvalues)
    singular_values = torch.sqrt(torch.clamp(eigenvalues, min=1e-10))
    
    # Retrieve top 2 principal components: V = X^T @ U @ Sigma^-1
    # Handle cases where singular values are extremely small (e.g. collinear checkpoints)
    if singular_values[0] > 1e-5:
        dir_x_flat = torch.matmul(X.t(), U[:, 0]) / singular_values[0]
    else:
        dir_x_flat = torch.randn(N)
        dir_x_flat = dir_x_flat / torch.clamp(torch.norm(dir_x_flat), min=1e-10)
        
    if M > 1 and singular_values[1] > 1e-4 * singular_values[0] and singular_values[1] > 1e-5:
        dir_y_flat = torch.matmul(X.t(), U[:, 1]) / singular_values[1]
    else:
        rnd = torch.randn_like(dir_x_flat)
        dir_y_flat = rnd - torch.dot(rnd, dir_x_flat) * dir_x_flat
        dir_y_flat = dir_y_flat / torch.clamp(torch.norm(dir_y_flat), min=1e-10)
        
    # Run a final Gram-Schmidt step to ensure perfect orthogonality in float32
    dir_y_flat = dir_y_flat - torch.dot(dir_y_flat, dir_x_flat) * dir_x_flat
    dir_y_flat = dir_y_flat / torch.clamp(torch.norm(dir_y_flat), min=1e-10)
        
    # Determine the center coordinate state dict
    if center_on_mean:
        center_flat = mean_W.squeeze(0)
    else:
        center_flat = flat_checkpoints[center_index]
        
    # Unflatten coordinates back into state dicts
    dir_x = {}
    dir_y = {}
    center_state_dict = {}
    
    idx = 0
    for k in param_keys:
        shape = first_sd[k].shape
        numel = first_sd[k].numel()
        
        dir_x[k] = dir_x_flat[idx : idx + numel].view(shape).clone()
        dir_y[k] = dir_y_flat[idx : idx + numel].view(shape).clone()
        center_state_dict[k] = center_flat[idx : idx + numel].view(shape).clone()
        
        idx += numel
        
    # Retain non-floating-point parameters (like batch norm counters) from the center checkpoint
    ref_sd = checkpoints[center_index]
    for k in first_sd.keys():
        if k not in center_state_dict:
            center_state_dict[k] = ref_sd[k].clone()
            dir_x[k] = torch.zeros_like(ref_sd[k])
            dir_y[k] = torch.zeros_like(ref_sd[k])
            
    if normalize:
        dir_x = filter_wise_normalize(dir_x, center_state_dict)
        dir_y = filter_wise_normalize(dir_y, center_state_dict)
        
    # Calculate projection coordinates for the trajectory
    trajectory_coords = []
    for sd in checkpoints:
        flat_t = torch.cat([sd[k].flatten() for k in param_keys])
        centered_t = flat_t - center_flat
        cx = torch.dot(centered_t, dir_x_flat).item()
        cy = torch.dot(centered_t, dir_y_flat).item()
        trajectory_coords.append((cx, cy))
        
    return dir_x, dir_y, center_state_dict, trajectory_coords


def compute_hvp(model, dataloader, criterion, vector_dict, device, max_batches=5):
    """
    Computes the Hessian-vector product H * v for a given model, dataloader, and vector.
    Uses Pearlmutter's algorithm for efficient second-order gradients.
    """
    model.zero_grad()
    params_with_grad = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not params_with_grad:
        return {}
        
    names = [item[0] for item in params_with_grad]
    tensors = [item[1] for item in params_with_grad]
    
    hvp_tensors = [torch.zeros_like(p) for p in tensors]
    batch_count = 0
    
    # Enable grad computation for Hessian calculations
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
            
        model.zero_grad()
        
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch
        else:
            continue
            
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # First backward pass with create_graph=True
        grads = torch.autograd.grad(loss, tensors, create_graph=True)
        
        # Calculate inner product: sum(grad_i * v_i)
        inner_prod = 0.0
        for g, name in zip(grads, names):
            if name in vector_dict:
                v = vector_dict[name].to(device)
                inner_prod += torch.sum(g * v)
                
        # Second backward pass to get H * v
        hvp_batch = torch.autograd.grad(inner_prod, tensors)
        
        for i, hvp_val in enumerate(hvp_batch):
            if hvp_val is not None:
                hvp_tensors[i] += hvp_val.detach()
                
        batch_count += 1
        
    if batch_count == 0:
        return {name: torch.zeros_like(p) for name, p in params_with_grad}
        
    # Return average HVP over the evaluated batches
    return {name: hvp_val / batch_count for name, hvp_val in zip(names, hvp_tensors)}


def generate_hessian_directions(model, dataloader, criterion, device=None, max_batches=5, max_iter=20, tol=1e-3):
    """
    Generates two direction vectors corresponding to the top two eigenvectors of the Hessian matrix.
    Uses Power Iteration with Deflation.
    
    Args:
        model (torch.nn.Module): The PyTorch model.
        dataloader (DataLoader): Data loader for gradient evaluations.
        criterion (callable): Loss function.
        device (torch.device or str, optional): Device context.
        max_batches (int): Max batches to average Hessian evaluations over.
        max_iter (int): Maximum Power Iteration steps.
        tol (float): Convergence tolerance.
        
    Returns:
        tuple: (dir_x, dir_y, center_state_dict)
    """
    import numpy as np
    
    if device is None:
        device = next(model.parameters()).device
        
    params_dict = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not params_dict:
        raise ValueError("Model has no trainable parameters to compute Hessian.")
        
    def get_norm(v_dict):
        total_sq = sum(torch.sum(v ** 2).item() for v in v_dict.values())
        return np.sqrt(total_sq)
        
    def normalize_dict(v_dict):
        norm = get_norm(v_dict)
        if norm < 1e-10:
            norm = 1e-10
        return {k: v / norm for k, v in v_dict.items()}
        
    def dot_dict(v1, v2):
        return sum(torch.sum(v1[k] * v2[k]).item() for k in v1.keys())
        
    # 1. Power Iteration for v1 (first eigenvector)
    v1 = {k: torch.randn_like(p) for k, p in params_dict.items()}
    v1 = normalize_dict(v1)
    
    last_eigenvalue = 0.0
    for _ in range(max_iter):
        hvp = compute_hvp(model, dataloader, criterion, v1, device, max_batches)
        eigenvalue = dot_dict(hvp, v1)
        v1 = normalize_dict(hvp)
        
        if abs(eigenvalue - last_eigenvalue) < tol:
            break
        last_eigenvalue = eigenvalue
        
    # 2. Power Iteration for v2 (second eigenvector) with deflation
    v2 = {k: torch.randn_like(p) for k, p in params_dict.items()}
    dot_v2_v1 = dot_dict(v2, v1)
    v2 = {k: v2[k] - dot_v2_v1 * v1[k] for k in v2.keys()}
    v2 = normalize_dict(v2)
    
    last_eigenvalue_2 = 0.0
    for _ in range(max_iter):
        hvp = compute_hvp(model, dataloader, criterion, v2, device, max_batches)
        
        # Deflate: w = w - (w . v1) * v1
        dot_hvp_v1 = dot_dict(hvp, v1)
        hvp_deflated = {k: hvp[k] - dot_hvp_v1 * v1[k] for k in hvp.keys()}
        
        eigenvalue = dot_dict(hvp_deflated, v2)
        v2 = normalize_dict(hvp_deflated)
        
        if abs(eigenvalue - last_eigenvalue_2) < tol:
            break
        last_eigenvalue_2 = eigenvalue
        
    # Assemble final direction state dicts
    center_state_dict = {k: p.data.clone() for k, p in model.named_parameters()}
    
    dir_x = {}
    dir_y = {}
    for k in center_state_dict.keys():
        if k in v1:
            dir_x[k] = v1[k].clone()
            dir_y[k] = v2[k].clone()
        else:
            dir_x[k] = torch.zeros_like(center_state_dict[k])
            dir_y[k] = torch.zeros_like(center_state_dict[k])
            
    return dir_x, dir_y, center_state_dict
