"""
Experiments Module for DDP_KBIT

This module contains experiment orchestration and benchmarking functionality
for the DDP_KBIT distributed deep learning system.
"""

from .runner import (
    exp_fn,
    run_multiple_experiments,
    DistributedDataFetcher,
    initialize_distributed_training
)
from .benchmarks import (
    calculate_boxplot_stats,
    PerformanceTimer,
    BenchmarkSuite
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
    'BenchmarkSuite',
]