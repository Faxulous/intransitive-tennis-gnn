"""
Reproducibility utilities for setting random seeds across all libraries.
"""
import torch
import numpy as np
import random
import os


def set_seed(seed: int = 8):
    """
    Set all random seeds for reproducibility across PyTorch, NumPy, Python's random, and CUDA.

    Args:
        seed: Random seed value (default: 8)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # For CUDA determinism (if using GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # For Python's hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)