"""
Utility functions for training, evaluation, and prediction:
  - Reproducibility (set_seed)
  - Checkpoint saving and loading
  - AverageMeter for tracking running averages of losses
"""

import os
import random
import numpy as np
import torch


def set_seed(seed):
    """Set seeds for Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Makes convolution operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    state = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_val_loss = checkpoint.get("val_loss", float("inf"))
    return start_epoch, best_val_loss



class AverageMeter:
    """
    Tracks a running average of a scalar value.

    Example:
    meter = AverageMeter()
    meter.update(loss.item(), batch_size)
    print(meter.avg)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        if self.count == 0:
            return 0.0
        return self.sum / self.count