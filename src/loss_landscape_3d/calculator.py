import numpy as np
import torch
import copy

class LossLandscapeCalculator:
    """
    Evaluates the model loss over a 2D grid of parameter perturbations.
    """
    def __init__(self, model, dataloader, criterion, device=None):
        """
        Initialize the calculator.
        
        Args:
            model (torch.nn.Module): The model to evaluate.
            dataloader (torch.utils.data.DataLoader): Data loader for evaluation.
            criterion (callable): Loss function (e.g. nn.CrossEntropyLoss()).
            device (torch.device or str, optional): Device to run evaluation on.
                                                   If None, auto-detected from model parameters.
        """
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        
        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
    def calculate(self, x_coords, y_coords, dir_x, dir_y, center_state, max_batches=None, metric_fn=None):
        """
        Evaluate the loss and optional metrics on a 2D grid defined by x_coords and y_coords.
        
        Args:
            x_coords (array-like): 1D array of coordinates for the x-axis.
            y_coords (array-like): 1D array of coordinates for the y-axis.
            dir_x (dict): Direction state dict for the x-axis.
            dir_y (dict): Direction state dict for the y-axis.
            center_state (dict): Reference state dict at the center of the grid.
            max_batches (int, optional): Maximum number of batches to evaluate per grid point.
            metric_fn (callable, optional): Custom metric function of signature (outputs, targets) -> float
                                          to compute alongside the loss.
                                          
        Returns:
            np.ndarray or tuple: If metric_fn is None, returns the 2D loss grid.
                                 If metric_fn is provided, returns (loss_grid, metric_grid).
        """
        # Save original state dict to restore later
        original_state = {k: v.cpu().clone().detach() for k, v in self.model.state_dict().items()}
        
        # Initialize grids
        loss_grid = np.zeros((len(x_coords), len(y_coords)))
        metric_grid = np.zeros((len(x_coords), len(y_coords))) if metric_fn is not None else None
        
        # Set model to evaluation mode
        self.model.eval()
        self.model.to(self.device)
        
        try:
            for i, x in enumerate(x_coords):
                for j, y in enumerate(y_coords):
                    # Compute perturbed state dict
                    perturbed_state = {}
                    for k in center_state.keys():
                        c_val = center_state[k]
                        dx_val = dir_x.get(k, torch.zeros_like(c_val))
                        dy_val = dir_y.get(k, torch.zeros_like(c_val))
                        
                        if c_val.is_floating_point():
                            perturbed_state[k] = c_val + x * dx_val + y * dy_val
                        else:
                            perturbed_state[k] = c_val
                            
                    self.model.load_state_dict(perturbed_state)
                    
                    total_loss = 0.0
                    total_metric = 0.0
                    total_samples = 0
                    
                    with torch.no_grad():
                        for b_idx, batch in enumerate(self.dataloader):
                            if max_batches is not None and b_idx >= max_batches:
                                break
                                
                            if isinstance(batch, (list, tuple)):
                                inputs, targets = batch
                            else:
                                continue
                                
                            inputs = inputs.to(self.device)
                            targets = targets.to(self.device)
                            
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, targets)
                            
                            batch_size = inputs.size(0)
                            total_loss += loss.item() * batch_size
                            
                            if metric_fn is not None:
                                total_metric += metric_fn(outputs, targets) * batch_size
                                
                            total_samples += batch_size
                            
                    loss_grid[i, j] = total_loss / max(total_samples, 1)
                    if metric_fn is not None:
                        metric_grid[i, j] = total_metric / max(total_samples, 1)
        finally:
            self.model.load_state_dict(original_state)
            
        if metric_fn is not None:
            return loss_grid, metric_grid
        return loss_grid

    def calculate_1d_path(self, alphas, state_dict_1, state_dict_2, max_batches=None, metric_fn=None):
        """
        Evaluate the loss and optional metrics along a 1D path between state_dict_1 and state_dict_2.
        $\theta(\alpha) = (1 - \alpha)\theta_1 + \alpha\theta_2$
        
        Args:
            alphas (array-like): 1D array of interpolation parameters (usually between 0.0 and 1.0).
            state_dict_1 (dict): Starting state dict (alpha = 0.0).
            state_dict_2 (dict): Ending state dict (alpha = 1.0).
            max_batches (int, optional): Max batches to evaluate per point.
            metric_fn (callable, optional): Custom metric function of signature (outputs, targets) -> float.
            
        Returns:
            np.ndarray or tuple: If metric_fn is None, returns a 1D array of losses.
                                 If metric_fn is provided, returns (loss_array, metric_array).
        """
        original_state = {k: v.cpu().clone().detach() for k, v in self.model.state_dict().items()}
        
        losses = np.zeros(len(alphas))
        metrics = np.zeros(len(alphas)) if metric_fn is not None else None
        
        self.model.eval()
        self.model.to(self.device)
        
        try:
            for idx, alpha in enumerate(alphas):
                interpolated_state = {}
                for k in state_dict_1.keys():
                    v1 = state_dict_1[k]
                    v2 = state_dict_2.get(k, v1)
                    
                    if v1.is_floating_point():
                        interpolated_state[k] = (1 - alpha) * v1 + alpha * v2
                    else:
                        interpolated_state[k] = v1
                        
                self.model.load_state_dict(interpolated_state)
                
                total_loss = 0.0
                total_metric = 0.0
                total_samples = 0
                
                with torch.no_grad():
                    for b_idx, batch in enumerate(self.dataloader):
                        if max_batches is not None and b_idx >= max_batches:
                            break
                            
                        if isinstance(batch, (list, tuple)):
                            inputs, targets = batch
                        else:
                            continue
                            
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        
                        batch_size = inputs.size(0)
                        total_loss += loss.item() * batch_size
                        
                        if metric_fn is not None:
                            total_metric += metric_fn(outputs, targets) * batch_size
                            
                        total_samples += batch_size
                        
                losses[idx] = total_loss / max(total_samples, 1)
                if metric_fn is not None:
                    metrics[idx] = total_metric / max(total_samples, 1)
        finally:
            self.model.load_state_dict(original_state)
            
        if metric_fn is not None:
            return losses, metrics
        return losses

    def suggest_coordinate_bounds(self, dir_x, dir_y, center_state, target_loss_factor=5.0, max_batches=None):
        """
        Suggests appropriate step bounds for the coordinate axes x and y.
        Explores along the positive axes of dir_x and dir_y until the loss increases
        by target_loss_factor, returning recommended bounds.
        
        Returns:
            tuple: (x_limit, y_limit) representing suggested bounds [-limit, limit].
        """
        original_state = {k: v.cpu().clone().detach() for k, v in self.model.state_dict().items()}
        
        self.model.eval()
        self.model.to(self.device)
        
        # Calculate baseline loss at center_state
        self.model.load_state_dict(center_state)
        center_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for b_idx, batch in enumerate(self.dataloader):
                if max_batches is not None and b_idx >= max_batches:
                    break
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    continue
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                batch_size = inputs.size(0)
                center_loss += loss.item() * batch_size
                total_samples += batch_size
                
        center_loss = center_loss / max(total_samples, 1)
        target_loss = center_loss * target_loss_factor
        
        def evaluate_loss_at(x, y):
            perturbed_state = {}
            for k in center_state.keys():
                c_val = center_state[k]
                dx_val = dir_x.get(k, torch.zeros_like(c_val))
                dy_val = dir_y.get(k, torch.zeros_like(c_val))
                if c_val.is_floating_point():
                    perturbed_state[k] = c_val + x * dx_val + y * dy_val
                else:
                    perturbed_state[k] = c_val
                    
            self.model.load_state_dict(perturbed_state)
            tot_loss = 0.0
            tot_samples = 0
            with torch.no_grad():
                for b_idx, batch in enumerate(self.dataloader):
                    if max_batches is not None and b_idx >= max_batches:
                        break
                    if isinstance(batch, (list, tuple)):
                        inputs, targets = batch
                    else:
                        continue
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    batch_size = inputs.size(0)
                    tot_loss += loss.item() * batch_size
                    tot_samples += batch_size
            return tot_loss / max(tot_samples, 1)

        try:
            limits = []
            for dx, dy in [(1.0, 0.0), (0.0, 1.0)]:
                step = 0.1
                current_loss = center_loss
                
                # Double step size until target loss is exceeded or we hit a max limit
                while current_loss < target_loss and step <= 10.0:
                    current_loss = evaluate_loss_at(step * dx, step * dy)
                    if current_loss >= target_loss:
                        break
                    step *= 2.0
                    
                limits.append(min(step, 10.0))
        finally:
            self.model.load_state_dict(original_state)
            
        return limits[0], limits[1]
