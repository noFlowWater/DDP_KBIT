"""
Training Configuration Module

This module contains all training-related configurations including:
- Model configurations
- Optimizer settings
- Training hyperparameters
- Validation options
- Loss functions and metrics

All configuration values are preserved exactly as they appear in the original notebook.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.metrics import Accuracy, Loss


class NeuralNetwork(nn.Module):
    """
    Simple neural network for MNIST classification.
    This is the same model definition used in the original notebook.
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Training Configuration
TRAINING_CONFIG = {
    "base_model": NeuralNetwork(),
    "optimizer_class": torch.optim.Adam,
    "optimizer_params": {"lr": 0.001},
    "loss_fn": torch.nn.CrossEntropyLoss(),
    "perform_validation": True,  # 검증을 수행하려면 True, 그렇지 않으면 False
    "num_epochs": 1,
    "batch_size": 32,
}

# Metrics Configuration
# Note: This is added to the training_config after creation in the original notebook
def get_training_metrics(loss_fn):
    """
    Returns the metrics configuration for training.
    
    Args:
        loss_fn: The loss function to use for the Loss metric
        
    Returns:
        dict: Dictionary of metrics
    """
    return {
        'loss': Loss(loss_fn),
        'accuracy': Accuracy()
    }

# Alternative training configurations used in experiments
TRAINING_CONFIG_EXTENDED = {
    "base_model": NeuralNetwork(),
    "optimizer_class": torch.optim.Adam,
    "optimizer_params": {"lr": 0.001},
    "loss_fn": torch.nn.CrossEntropyLoss(),
    "perform_validation": True,
    "num_epochs": 30,
    "batch_size": 128,
}

# Batch size configurations for experiments
BATCH_SIZE_EXPERIMENTS = [8, 16, 32, 64, 128, 256, 512]

# Training configuration with complete setup
def get_complete_training_config():
    """
    Returns a complete training configuration with metrics included.
    """
    config = TRAINING_CONFIG.copy()
    config["metrics"] = get_training_metrics(config['loss_fn'])
    return config

def get_extended_training_config():
    """
    Returns an extended training configuration for longer experiments.
    """
    config = TRAINING_CONFIG_EXTENDED.copy()
    config["metrics"] = get_training_metrics(config['loss_fn'])
    return config