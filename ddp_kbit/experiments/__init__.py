"""
Experiments Module for ddp_kbit

This module contains experiment orchestration and benchmarking functionality
for the ddp_kbit distributed deep learning system.
"""

from ddp_kbit.experiments.runner import (
    exp_fn,
    run_multiple_experiments,
    initialize_distributed_training
)
from ddp_kbit.experiments.benchmarks import (
    calculate_boxplot_stats,
    PerformanceTimer,
)

__all__ = [
    # Experiment runner
    'exp_fn',
    'run_multiple_experiments',
    'initialize_distributed_training',
    
    # Benchmarking
    'calculate_boxplot_stats',
    'PerformanceTimer',
]