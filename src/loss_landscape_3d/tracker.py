import os
import glob
import re
import copy
import torch

class TrajectoryTracker:
    """
    Tracks and records the trajectory of model parameters during training.
    Can store checkpoints in-memory (cloned to CPU) or on disk to save memory.
    """
    def __init__(self, dir_path=None):
        """
        Initialize the tracker.
        
        Args:
            dir_path (str, optional): Directory path to save checkpoints on disk.
                                      If None, checkpoints are stored in memory.
        """
        self.dir_path = dir_path
        self.checkpoints_in_memory = []
        self._count = 0
        
        if self.dir_path is not None:
            os.makedirs(self.dir_path, exist_ok=True)

    def save_checkpoint(self, model, step_or_epoch=None):
        """
        Save the current state dict of the model. Tensors are cloned and moved to CPU
        to prevent memory leakage and mutation during subsequent training steps.
        
        Args:
            model (torch.nn.Module): The model to track.
            step_or_epoch (int, optional): Identifier for the checkpoint (e.g., training step).
                                           If not provided, an auto-incrementing count is used.
        """
        # Deep clone state dict and move to CPU
        state_dict_copy = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
        
        if step_or_epoch is None:
            step_or_epoch = self._count
            
        if self.dir_path is not None:
            filename = f"checkpoint_{step_or_epoch:06d}.pt"
            filepath = os.path.join(self.dir_path, filename)
            torch.save(state_dict_copy, filepath)
        else:
            self.checkpoints_in_memory.append(state_dict_copy)
            
        self._count += 1

    def get_checkpoints(self):
        """
        Retrieve all tracked checkpoints.
        
        Returns:
            list of dict: A list of state dicts sorted by step/epoch.
        """
        if self.dir_path is not None:
            # Find all checkpoint files matching the pattern
            pattern = os.path.join(self.dir_path, "checkpoint_*.pt")
            files = glob.glob(pattern)
            
            # Sort files numerically based on the checkpoint number
            def extract_number(filepath):
                match = re.search(r'checkpoint_(\d+)\.pt$', filepath)
                return int(match.group(1)) if match else 0
                
            files.sort(key=extract_number)
            
            checkpoints = []
            for f in files:
                try:
                    checkpoints.append(torch.load(f, map_location='cpu'))
                except Exception as e:
                    # Fallback for newer torch version warning/safetensors if applicable
                    checkpoints.append(torch.load(f, map_location='cpu', weights_only=True))
            return checkpoints
        else:
            return self.checkpoints_in_memory

    def clear(self):
        """
        Clear all saved checkpoints (both in-memory and on-disk).
        """
        self.checkpoints_in_memory.clear()
        self._count = 0
        
        if self.dir_path is not None and os.path.exists(self.dir_path):
            pattern = os.path.join(self.dir_path, "checkpoint_*.pt")
            files = glob.glob(pattern)
            for f in files:
                try:
                    os.remove(f)
                except OSError:
                    pass
