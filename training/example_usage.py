"""
Example usage of the extracted training modules.

This file demonstrates how to use the modularized training components
extracted from the original notebook. It shows both simple and advanced
usage patterns for distributed training.

Run this example with:
    python -m torch.distributed.launch --nproc_per_node=2 example_usage.py
"""

import torch
import torch.nn as nn
from ignite.metrics import Loss, Accuracy

# Import the extracted modules
from DDP_KBIT.training import (
    main_fn, 
    initialize_distributed_training,
    create_simple_training_config,
    create_local_data_config,
    TrainingMetricsTracker,
    validate_distributed_environment
)


# Simple neural network for demonstration
class SimpleNet(nn.Module):
    """Simple neural network for MNIST classification."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def simple_training_example():
    """
    Simple example of distributed training using the extracted modules.
    """
    print("=== Simple Training Example ===")
    
    # Check if environment is set up for distributed training
    if not validate_distributed_environment():
        print("Environment not configured for distributed training")
        print("Run with: python -m torch.distributed.launch --nproc_per_node=2 example_usage.py")
        return
    
    # Create a simple model
    model = SimpleNet()
    
    # Create training configuration using helper function
    training_config = create_simple_training_config(
        model=model,
        num_epochs=5,
        batch_size=64,
        learning_rate=0.001,
        optimizer_class=torch.optim.Adam
    )
    
    # Create data loader configuration for local data
    data_loader_config = create_local_data_config("/path/to/your/data")
    
    # Kafka configuration (empty for local training)
    kafka_config = {}
    
    # Note: This would fail without actual data, but shows the usage pattern
    try:
        results = main_fn(
            training_config=training_config,
            kafka_config=kafka_config,
            data_loader_config=data_loader_config,
            use_gpu=True
        )
        
        print("Training completed successfully!")
        print("Results keys:", list(results.keys()))
        
    except Exception as e:
        print(f"Training failed (expected without real data): {e}")


def advanced_training_example():
    """
    Advanced example showing custom configuration and metrics tracking.
    """
    print("\n=== Advanced Training Example ===")
    
    # Check distributed environment
    if not validate_distributed_environment():
        print("Environment not configured for distributed training")
        return
    
    # Initialize distributed training manually
    try:
        dist_config = initialize_distributed_training(use_gpu=True)
        print(f"Initialized distributed training:")
        print(f"  Rank: {dist_config['global_rank']}")
        print(f"  World size: {dist_config['world_size']}")
        print(f"  Device: {dist_config['device']}")
        print(f"  Backend: {dist_config['backend']}")
        
    except Exception as e:
        print(f"Failed to initialize distributed training: {e}")
        return
    
    # Create custom model and configuration
    model = SimpleNet(input_size=784, hidden_size=256, num_classes=10)
    loss_fn = nn.CrossEntropyLoss()
    
    # Custom training configuration
    training_config = {
        'num_epochs': 10,
        'batch_size': 128,
        'perform_validation': True,
        'base_model': model,
        'optimizer_class': torch.optim.SGD,
        'optimizer_params': {
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 1e-4
        },
        'loss_fn': loss_fn,
        'metrics': {
            'accuracy': Accuracy(),
            'loss': Loss(loss_fn)
        }
    }
    
    # Kafka-based data configuration example
    data_loader_config = {
        'data_loader_type': 'kafka',
        'consumer_params': {
            'bootstrap_servers': 'localhost:9092',
            'group_id': 'training-group',
            'auto_offset_reset': 'earliest',
            'enable_auto_commit': False
        },
        'api_config': {
            'base_url': 'http://localhost:8080/api',
            'params': {
                'send_topic': 'training-data',
                'dataset_name': 'mnist',
                'data_size': 70000
            }
        },
        'dataset_split_config': [
            {'rate': 0.7, 'fillup': False},  # Training data
            {'rate': 0.2, 'fillup': False},  # Validation data
            {'rate': 0.1, 'fillup': True}    # Test data
        ]
    }
    
    print("Configuration created successfully")
    print(f"Model: {type(model).__name__}")
    print(f"Optimizer: {training_config['optimizer_class'].__name__}")
    print(f"Data loader type: {data_loader_config['data_loader_type']}")


def metrics_tracking_example():
    """
    Example of using the metrics tracking functionality.
    """
    print("\n=== Metrics Tracking Example ===")
    
    # Create a metrics tracker
    tracker = TrainingMetricsTracker()
    tracker.start_training()
    
    # Simulate training epochs
    for epoch in range(5):
        tracker.start_epoch()
        
        # Simulate training metrics
        train_metrics = {
            'loss': 1.0 - (epoch * 0.15),
            'accuracy': 0.6 + (epoch * 0.08)
        }
        tracker.save_metrics(train_metrics, 'train')
        
        # Simulate validation metrics
        val_metrics = {
            'loss': 1.1 - (epoch * 0.12),
            'accuracy': 0.55 + (epoch * 0.075)
        }
        tracker.save_metrics(val_metrics, 'val')
        
        tracker.end_epoch()
    
    # Print metrics summary
    tracker.print_summary(rank=0)
    
    # Get specific metric information
    latest_train = tracker.get_latest_metrics('train')
    print(f"\nLatest training metrics: {latest_train}")
    
    best_acc, best_epoch = tracker.get_best_metric('accuracy', 'val', mode='max')
    print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    print("Training Modules Usage Examples")
    print("=" * 40)
    
    # Run examples
    simple_training_example()
    advanced_training_example()
    metrics_tracking_example()
    
    print("\nExamples completed!")