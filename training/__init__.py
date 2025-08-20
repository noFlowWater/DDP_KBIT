"""
Training Module for DDP_KBIT

This module contains training-related functionality for the DDP_KBIT
distributed deep learning system, including distributed training setup,
metrics tracking, and the main training orchestration.
"""

from .trainer import main_fn, TrainingConfig, DataLoaderConfig

from .distributed import (
    initialize_distributed_training,
    cleanup_distributed_training,
    DistributedTrainingConfig,
    is_distributed_available,
    get_rank,
    get_world_size
)
from .metrics import (
    reduce_dict,
    save_metrics,
    debug_dataloader,
    TrainingMetricsTracker,
    PerformanceProfiler
)

__all__ = [
    # Main training function
    'main_fn',
    'TrainingConfig',
    'DataLoaderConfig',
    
    # Distributed training
    'initialize_distributed_training',
    'cleanup_distributed_training',
    'DistributedTrainingConfig',
    'is_distributed_available',
    'get_rank',
    'get_world_size',
    
    # Metrics and profiling
    'reduce_dict',
    'save_metrics',
    'debug_dataloader',
    'TrainingMetricsTracker',
    'PerformanceProfiler',
]