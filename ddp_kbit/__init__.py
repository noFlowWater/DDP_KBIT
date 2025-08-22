"""
DDP_KBIT: Distributed Deep Learning with Kafka-Based Intelligent Training

A modular distributed deep learning system extracted from sparkDL_KBIT_gpu_lightning.ipynb
for scalable machine learning with Apache Spark, PyTorch, and Apache Kafka.

This package provides:
- Distributed training capabilities with PyTorch and PySpark
- Kafka-based data streaming and consumption
- Multiple data format support (JSON, Avro, Protobuf, MongoDB)
- Performance benchmarking and experiment management
- GPU acceleration support with RAPIDS

Modules:
- config: Configuration management (training, data, spark)
- data: Data loading, transformation, and Kafka utilities
- models: Neural network definitions and model factories
- training: Distributed training logic and metrics
- experiments: Experiment orchestration and benchmarking
- utils: Utilities for Spark, checkpoints, and visualization
"""

__version__ = "0.1.0"
__author__ = "DDP_KBIT Team"
__email__ = "ddp_kbit@example.com"

# Import main entry point
from .main import main

# Import key classes and functions for easy access
try:
    from .models.networks import Net, NeuralNetwork, create_cnn_model, create_feedforward_model
    from .training.trainer import main_fn
    from .training.distributed import initialize_distributed_training
    from .experiments.runner import exp_fn, run_multiple_experiments
    from .utils.spark_utils import create_spark_session
    from .utils.checkpoint import load_checkpoint, save_checkpoint
    from .utils.visualization import calculate_boxplot_stats, plot_training_metrics
    
    # Define what gets imported with "from ddp_kbit import *"
    __all__ = [
        # Main entry point
        'main',
        
        # Model classes
        'Net',
        'NeuralNetwork',
        'create_cnn_model',
        'create_feedforward_model',
        
        # Training functions
        'main_fn',
        'initialize_distributed_training',
        
        # Experiment functions
        'exp_fn',
        'run_multiple_experiments',
        
        # Utility functions
        'create_spark_session',
        'load_checkpoint',
        'save_checkpoint',
        'calculate_boxplot_stats',
        'plot_training_metrics',
    ]
    
except ImportError as e:
    # Handle cases where some dependencies might not be available
    import warnings
    warnings.warn(f"Some ddp_kbit modules could not be imported: {e}")
    
    # Minimal exports in case of import errors
    __all__ = ['main']


def get_version():
    """Return the current version of ddp_kbit."""
    return __version__


def get_info():
    """Return information about the ddp_kbit package."""
    return {
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'description': 'Distributed Deep Learning with Kafka-Based Intelligent Training'
    }