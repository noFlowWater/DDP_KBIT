"""
Experiments Module for DDP_KBIT

This module contains experiment orchestration and benchmarking functionality
for the DDP_KBIT distributed deep learning system.
"""

from DDP_KBIT.experiments.runner import (
    exp_fn,
    run_multiple_experiments,
    DistributedDataFetcher,
    initialize_distributed_training
)
from DDP_KBIT.experiments.benchmarks import (
    calculate_boxplot_stats,
    PerformanceTimer,
)

__all__ = [
    # Experiment runner
    'exp_fn',
    'run_multiple_experiments',
    'DistributedDataFetcher',
    'initialize_distributed_training',
    
    # Benchmarking
    'calculate_boxplot_stats',
    'PerformanceTimer',
]