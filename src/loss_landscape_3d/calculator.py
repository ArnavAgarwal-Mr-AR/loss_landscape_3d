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
            
    def calculate(self, x_coords, y_coords, dir_x, dir_y, center_state, max_batches=None):
        """
        Evaluate the loss on a 2D grid defined by x_coords and y_coords.
        
        Args:
            x_coords (array-like): 1D array of coordinates for the x-axis.
            y_coords (array-like): 1D array of coordinates for the y-axis.
            dir_x (dict): Direction state dict for the x-axis.
            dir_y (dict): Direction state dict for the y-axis.
            center_state (dict): Reference state dict at the center of the grid.
            max_batches (int, optional): Maximum number of batches to evaluate per grid point
                                         (useful for speeding up evaluation on large datasets).
                                         
        Returns:
            np.ndarray: 2D array of shape (len(x_coords), len(y_coords)) containing loss values.
        """
        # Save original state dict to restore later
        # Move to CPU and clone to save GPU memory and prevent mutation
        original_state = {k: v.cpu().clone().detach() for k, v in self.model.state_dict().items()}
        
        # Initialize grid
        loss_grid = np.zeros((len(x_coords), len(y_coords)))
        
        # Set model to evaluation mode
        self.model.eval()
        self.model.to(self.device)
        
        try:
            for i, x in enumerate(x_coords):
                for j, y in enumerate(y_coords):
                    # Compute perturbed state dict on CPU
                    perturbed_state = {}
                    for k in center_state.keys():
                        c_val = center_state[k]
                        dx_val = dir_x.get(k, torch.zeros_like(c_val))
                        dy_val = dir_y.get(k, torch.zeros_like(c_val))
                        
                        if c_val.is_floating_point():
                            perturbed_state[k] = c_val + x * dx_val + y * dy_val
                        else:
                            perturbed_state[k] = c_val
                            
                    # Load perturbed weights into model (PyTorch handles moving to appropriate device)
                    self.model.load_state_dict(perturbed_state)
                    
                    # Evaluate loss
                    total_loss = 0.0
                    total_samples = 0
                    
                    with torch.no_grad():
                        for b_idx, (inputs, targets) in enumerate(self.dataloader):
                            if max_batches is not None and b_idx >= max_batches:
                                break
                                
                            inputs = inputs.to(self.device)
                            targets = targets.to(self.device)
                            
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, targets)
                            
                            # Accumulate weighted batch loss
                            batch_size = inputs.size(0)
                            total_loss += loss.item() * batch_size
                            total_samples += batch_size
                            
                    # Store average loss
                    loss_grid[i, j] = total_loss / max(total_samples, 1)
        finally:
            # Always restore the original weights, even if evaluation is interrupted
            self.model.load_state_dict(original_state)
            
        return loss_grid
