import os
import shutil
import tempfile
import torch
import torch.nn as nn
import pytest
from loss_landscape_3d.tracker import TrajectoryTracker

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)
        
    def forward(self, x):
        return self.fc(x)

def test_tracker_in_memory():
    model = SimpleModel()
    tracker = TrajectoryTracker()
    
    # Save a couple checkpoints
    tracker.save_checkpoint(model, step_or_epoch=1)
    tracker.save_checkpoint(model, step_or_epoch=2)
    
    checkpoints = tracker.get_checkpoints()
    assert len(checkpoints) == 2
    assert 'fc.weight' in checkpoints[0]
    assert 'fc.bias' in checkpoints[0]
    
    # Check that in-memory checkpoints are cloned and detached (not sharing same tensor)
    original_weight = checkpoints[0]['fc.weight'].clone()
    with torch.no_grad():
        model.fc.weight.fill_(0.0)
        
    assert not torch.allclose(checkpoints[0]['fc.weight'], model.fc.weight)
    assert torch.allclose(checkpoints[0]['fc.weight'], original_weight)
    
    # Test clear
    tracker.clear()
    assert len(tracker.get_checkpoints()) == 0

def test_tracker_on_disk():
    # Create temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp()
    
    try:
        model = SimpleModel()
        tracker = TrajectoryTracker(dir_path=temp_dir)
        
        # Save checkpoints out-of-order to test numerical sorting
        tracker.save_checkpoint(model, step_or_epoch=10)
        tracker.save_checkpoint(model, step_or_epoch=2)
        tracker.save_checkpoint(model, step_or_epoch=5)
        
        checkpoints = tracker.get_checkpoints()
        assert len(checkpoints) == 3
        
        # Verify checkpoints are sorted numerically: 2, 5, 10
        # By inspecting directory files
        files = os.listdir(temp_dir)
        assert len(files) == 3
        assert "checkpoint_000002.pt" in files
        assert "checkpoint_000005.pt" in files
        assert "checkpoint_000010.pt" in files
        
        # Clear files
        tracker.clear()
        assert len(tracker.get_checkpoints()) == 0
        assert len(os.listdir(temp_dir)) == 0
        
    finally:
        shutil.rmtree(temp_dir)
