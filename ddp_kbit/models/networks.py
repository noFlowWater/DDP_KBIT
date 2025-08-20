"""
Neural Network Models for KBIT DDP Training

This module contains neural network model definitions extracted from the 
sparkDL_KBIT_gpu_lightning.ipynb notebook. It includes two main model 
architectures for MNIST digit classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    Simple Convolutional Neural Network for MNIST digit classification.
    
    This is a classic CNN architecture with two convolutional layers followed
    by two fully connected layers. It uses dropout for regularization and
    max pooling for dimensionality reduction.
    
    Architecture:
    - Conv2d(1, 10, kernel_size=5) + ReLU + MaxPool2d(2)
    - Conv2d(10, 20, kernel_size=5) + Dropout2d + ReLU + MaxPool2d(2)  
    - Linear(320, 50) + ReLU + Dropout
    - Linear(50, 10) + LogSoftmax
    
    Input: (batch_size, 1, 28, 28) - MNIST images
    Output: (batch_size, 10) - Log probabilities for 10 digit classes
    """
    
    def __init__(self):
        """Initialize the CNN model layers."""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            torch.Tensor: Log softmax output of shape (batch_size, 10)
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
    
    # TODO: loss_fn(self) / optimizer(self) / metric(self) -> Config implementation needed


class NeuralNetwork(nn.Module):
    """
    Alternative fully connected neural network for MNIST digit classification.
    
    This is a simpler architecture using only fully connected layers with
    ReLU activation and softmax output. It flattens the input images and
    processes them through a sequential stack of linear transformations.
    
    Architecture:
    - Flatten(28*28)
    - Linear(784, 128) + ReLU
    - Linear(128, 10) + Softmax
    
    Input: (batch_size, 1, 28, 28) - MNIST images  
    Output: (batch_size, 10) - Softmax probabilities for 10 digit classes
    """
    
    def __init__(self):
        """Initialize the fully connected model layers."""
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            torch.Tensor: Softmax output of shape (batch_size, 10)
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def create_cnn_model():
    """
    Factory function to create a CNN model instance.
    
    Returns:
        Net: An instance of the CNN model
    """
    return Net()


def create_feedforward_model():
    """
    Factory function to create a feedforward neural network model instance.
    
    Returns:
        NeuralNetwork: An instance of the feedforward neural network model
    """
    return NeuralNetwork()


def get_model_by_name(model_name: str):
    """
    Factory function to get a model instance by name.
    
    Args:
        model_name (str): Name of the model ('cnn' or 'feedforward')
        
    Returns:
        nn.Module: The requested model instance
        
    Raises:
        ValueError: If model_name is not recognized
    """
    model_name = model_name.lower()
    
    if model_name == 'cnn' or model_name == 'net':
        return create_cnn_model()
    elif model_name == 'feedforward' or model_name == 'neuralnetwork':
        return create_feedforward_model()
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                         f"Available models: 'cnn', 'net', 'feedforward', 'neuralnetwork'")