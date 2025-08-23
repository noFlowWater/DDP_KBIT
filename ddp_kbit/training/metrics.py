"""
Training metrics and evaluation utilities for distributed machine learning.

This module provides comprehensive functionality for tracking, computing, and managing 
training metrics in distributed environments. It includes utilities for metric 
aggregation across processes, performance measurement, and evaluation logic.

Features:
- Distributed metric reduction and aggregation
- Training progress tracking and logging
- Evaluation metrics computation
- Performance measurement utilities
- Integration with PyTorch Ignite metrics
- Custom metric implementations

Example:
    >>> from metrics import reduce_dict, save_metrics, TrainingMetricsTracker
    >>> 
    >>> # Reduce metrics across distributed processes
    >>> metrics = {'loss': torch.tensor(0.5), 'accuracy': torch.tensor(0.85)}
    >>> reduced_metrics = reduce_dict(metrics, average=True)
    >>> 
    >>> # Track training metrics
    >>> tracker = TrainingMetricsTracker()
    >>> tracker.save_metrics({'loss': 0.3, 'accuracy': 0.90}, 'train')
"""

import time
import torch
import torch.distributed as dist
from typing import Dict, List, Any, Optional, Union, Callable
from collections import defaultdict, deque
import numpy as np


def reduce_dict(input_dict: Dict[str, torch.Tensor], average: bool = True) -> Dict[str, torch.Tensor]:
    """
    Reduces a dictionary of tensors across all processes in a distributed setting.
    
    This function performs an all-reduce operation on a dictionary of tensors,
    aggregating values from all processes in the distributed training setup.
    
    Args:
        input_dict (Dict[str, torch.Tensor]): Dictionary containing tensors to be reduced.
        average (bool): If True, compute the average across processes. 
                       If False, compute the sum. Defaults to True.
    
    Returns:
        Dict[str, torch.Tensor]: Dictionary with reduced tensors.
    
    Example:
        >>> metrics = {
        ...     'loss': torch.tensor(0.5),
        ...     'accuracy': torch.tensor(0.85)
        ... }
        >>> reduced = reduce_dict(metrics, average=True)
        >>> print(f"Average loss: {reduced['loss'].item()}")
    
    Note:
        This function requires distributed training to be initialized.
        If world_size < 2, returns the input dictionary unchanged.
    """
    world_size = float(dist.get_world_size()) if dist.is_initialized() else 1.0
    
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names, values = [], []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])

        values = torch.stack(values, dim=0)  # Stack all values

        # Sum values across all processes
        dist.all_reduce(values, op=dist.ReduceOp.SUM)

        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}
    
    return reduced_dict


def save_metrics(engine_metrics_items: List[tuple], epochs_metric_dict: Dict[str, List]) -> None:
    """
    Saves engine metrics to a structured dictionary for tracking over epochs.
    
    This function extracts metrics from a PyTorch Ignite engine and stores them
    in a dictionary structure that tracks metric values across training epochs.
    
    Args:
        engine_metrics_items (List[tuple]): List of (metric_name, metric_value) tuples
                                           from engine.state.metrics.items().
        epochs_metric_dict (Dict[str, List]): Dictionary to store metric history,
                                            where keys are metric names and values
                                            are lists of metric values.
    
    Example:
        >>> results = {'train_metrics': {}, 'val_metrics': {}}
        >>> engine_metrics = [('loss', 0.3), ('accuracy', 0.85)]
        >>> save_metrics(engine_metrics, results['train_metrics'])
        >>> print(results['train_metrics'])  # {'loss': [0.3], 'accuracy': [0.85]}
    """
    for k, v in engine_metrics_items:
        try:
            epochs_metric_dict[k].append(v)
        except KeyError:
            epochs_metric_dict[k] = [v]


class TrainingMetricsTracker:
    """
    Comprehensive training metrics tracking and management system.
    
    This class provides functionality to track, store, and analyze training metrics
    across multiple epochs and phases (train/validation/test). It supports both
    scalar and tensor metrics and provides utilities for metric aggregation.
    
    Attributes:
        metrics_history (Dict): Dictionary storing metric history for each phase
        current_epoch (int): Current training epoch
        start_time (float): Training start timestamp
        epoch_times (List[float]): List of epoch durations
    """
    
    def __init__(self):
        self.metrics_history = {
            'train_metrics': defaultdict(list),
            'val_metrics': defaultdict(list),
            'test_metrics': defaultdict(list)
        }
        self.current_epoch = 0
        self.start_time = None
        self.epoch_times = []
        self.epoch_start_time = None
    
    def start_training(self) -> None:
        """Marks the start of training and initializes timing."""
        self.start_time = time.time()
        self.current_epoch = 0
        print("Training metrics tracking started")
    
    def start_epoch(self) -> None:
        """Marks the start of an epoch."""
        self.epoch_start_time = time.time()
        self.current_epoch += 1
    
    def end_epoch(self) -> None:
        """Marks the end of an epoch and records timing."""
        if self.epoch_start_time is not None:
            epoch_duration = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_duration)
            self.epoch_start_time = None
    
    def save_metrics(self, metrics: Dict[str, Any], phase: str = 'train') -> None:
        """
        Saves metrics for a specific training phase.
        
        Args:
            metrics (Dict[str, Any]): Dictionary of metric values
            phase (str): Training phase ('train', 'val', 'test')
        """
        phase_key = f'{phase}_metrics'
        if phase_key not in self.metrics_history:
            self.metrics_history[phase_key] = defaultdict(list)
        
        for name, value in metrics.items():
            # Convert tensor values to Python scalars
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.detach().cpu().numpy()
            
            self.metrics_history[phase_key][name].append(value)
    
    def get_latest_metrics(self, phase: str = 'train') -> Dict[str, Any]:
        """
        Gets the latest metrics for a specific phase.
        
        Args:
            phase (str): Training phase ('train', 'val', 'test')
        
        Returns:
            Dict[str, Any]: Latest metric values
        """
        phase_key = f'{phase}_metrics'
        latest_metrics = {}
        
        for name, values in self.metrics_history[phase_key].items():
            if values:
                latest_metrics[name] = values[-1]
        
        return latest_metrics
    
    def get_metric_history(self, metric_name: str, phase: str = 'train') -> List[Any]:
        """
        Gets the complete history of a specific metric.
        
        Args:
            metric_name (str): Name of the metric
            phase (str): Training phase ('train', 'val', 'test')
        
        Returns:
            List[Any]: List of metric values across epochs
        """
        phase_key = f'{phase}_metrics'
        return self.metrics_history[phase_key].get(metric_name, [])
    
    def get_average_metric(self, metric_name: str, phase: str = 'train', last_n: Optional[int] = None) -> float:
        """
        Computes the average value of a metric.
        
        Args:
            metric_name (str): Name of the metric
            phase (str): Training phase ('train', 'val', 'test')
            last_n (Optional[int]): If specified, average only the last n values
        
        Returns:
            float: Average metric value
        """
        history = self.get_metric_history(metric_name, phase)
        if not history:
            return 0.0
        
        if last_n is not None:
            history = history[-last_n:]
        
        return np.mean(history)
    
    def get_best_metric(self, metric_name: str, phase: str = 'train', mode: str = 'max') -> tuple:
        """
        Gets the best value and epoch for a specific metric.
        
        Args:
            metric_name (str): Name of the metric
            phase (str): Training phase ('train', 'val', 'test')
            mode (str): 'max' for highest value, 'min' for lowest value
        
        Returns:
            tuple: (best_value, best_epoch)
        """
        history = self.get_metric_history(metric_name, phase)
        if not history:
            return None, None
        
        # Check if history contains scalar values or arrays
        # For arrays (like per-class precision/recall), compute mean for comparison
        scalar_history = []
        for value in history:
            if isinstance(value, np.ndarray):
                # For multi-dimensional metrics, use mean
                scalar_history.append(np.mean(value))
            else:
                scalar_history.append(value)
        
        if not scalar_history:
            return None, None
        
        if mode == 'max':
            best_idx = np.argmax(scalar_history)
        else:
            best_idx = np.argmin(scalar_history)
        
        return scalar_history[best_idx], best_idx + 1
    
    def print_summary(self, rank: int = 0) -> None:
        """
        Prints a summary of training metrics.
        
        Args:
            rank (int): Process rank (only rank 0 prints to avoid duplication)
        """
        if rank != 0:
            return
        
        print("\n" + "="*50)
        print("TRAINING METRICS SUMMARY")
        print("="*50)
        
        if self.start_time is not None:
            total_time = time.time() - self.start_time
            print(f"Total training time: {total_time:.2f} seconds")
        
        if self.epoch_times:
            avg_epoch_time = np.mean(self.epoch_times)
            print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
            print(f"Total epochs completed: {len(self.epoch_times)}")
        
        for phase in ['train', 'val', 'test']:
            phase_key = f'{phase}_metrics'
            if phase_key in self.metrics_history and self.metrics_history[phase_key]:
                print(f"\n{phase.upper()} METRICS:")
                for metric_name in self.metrics_history[phase_key]:
                    latest = self.get_latest_metrics(phase).get(metric_name)
                    best_val, best_epoch = self.get_best_metric(metric_name, phase, 'max')
                    
                    if latest is not None:
                        print(f"  {metric_name}:")
                        # Handle both scalar values and numpy arrays
                        if isinstance(latest, np.ndarray):
                            if latest.size == 1:
                                print(f"    Latest: {latest.item():.4f}")
                            else:
                                print(f"    Latest: {np.mean(latest):.4f} (array mean)")
                        else:
                            print(f"    Latest: {latest:.4f}")
                        
                        if best_val is not None:
                            print(f"    Best: {best_val:.4f} (Epoch {best_epoch})")
        
        print("="*50)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the metrics tracker to a dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of all metrics and metadata
        """
        return {
            'metrics_history': dict(self.metrics_history),
            'current_epoch': self.current_epoch,
            'total_training_time': time.time() - self.start_time if self.start_time else None,
            'epoch_times': self.epoch_times,
            'average_epoch_time': np.mean(self.epoch_times) if self.epoch_times else None
        }


class PerformanceProfiler:
    """
    Performance profiling utilities for distributed training.
    
    This class provides tools for measuring and analyzing the performance
    of different components in the training pipeline.
    """
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.active_timers = {}
    
    def start_timer(self, name: str) -> None:
        """Starts a timer with the given name."""
        self.active_timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """
        Ends a timer and records the duration.
        
        Args:
            name (str): Timer name
        
        Returns:
            float: Duration in seconds
        """
        if name not in self.active_timers:
            raise ValueError(f"Timer '{name}' was not started")
        
        duration = time.time() - self.active_timers[name]
        self.timings[name].append(duration)
        del self.active_timers[name]
        
        return duration
    
    def get_average_time(self, name: str) -> float:
        """Gets the average time for a specific timer."""
        if name not in self.timings:
            return 0.0
        return np.mean(self.timings[name])
    
    def get_total_time(self, name: str) -> float:
        """Gets the total time for a specific timer."""
        if name not in self.timings:
            return 0.0
        return sum(self.timings[name])
    
    def print_summary(self, rank: int = 0) -> None:
        """Prints a performance summary."""
        if rank != 0:
            return
        
        print("\n" + "="*50)
        print("PERFORMANCE PROFILE")
        print("="*50)
        
        for name, times in self.timings.items():
            avg_time = np.mean(times)
            total_time = sum(times)
            count = len(times)
            
            print(f"{name}:")
            print(f"  Count: {count}")
            print(f"  Total time: {total_time:.4f}s")
            print(f"  Average time: {avg_time:.4f}s")
            print(f"  Min time: {min(times):.4f}s")
            print(f"  Max time: {max(times):.4f}s")
            print()


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes classification accuracy.
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth labels
    
    Returns:
        float: Accuracy as a percentage
    """
    pred_classes = torch.argmax(predictions, dim=1)
    correct = (pred_classes == targets).float()
    accuracy = correct.mean().item() * 100.0
    return accuracy


def compute_top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """
    Computes top-k accuracy.
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth labels
        k (int): Number of top predictions to consider
    
    Returns:
        float: Top-k accuracy as a percentage
    """
    _, top_k_preds = torch.topk(predictions, k, dim=1)
    targets_expanded = targets.view(-1, 1).expand_as(top_k_preds)
    correct = (top_k_preds == targets_expanded).any(dim=1).float()
    accuracy = correct.mean().item() * 100.0
    return accuracy


def debug_dataloader(dataloader, rank: int = 0) -> None:
    """
    Debug utility to inspect dataloader properties and first batch.
    
    This function provides detailed information about a DataLoader including
    dataset size, batch properties, and data types for debugging purposes.
    
    Args:
        dataloader: PyTorch DataLoader to inspect
        rank (int): Process rank (only rank 0 prints to avoid duplication)
    
    Example:
        >>> debug_dataloader(train_loader, rank=0)
        Total number of samples in the dataset: 60000
        Batch index: 0
        Data shape: torch.Size([32, 1, 28, 28])
        Data type: torch.float32
        Target shape: torch.Size([32])
        Target type: torch.int64
    """
    if rank != 0:
        return
    
    try:
        # Get total dataset size
        total_samples = len(dataloader.dataset)
        print(f"    Total number of samples in the dataset: {total_samples}")
        
        # Inspect first batch
        for batch_idx, (data, target) in enumerate(dataloader):
            print(f"    Batch index: {batch_idx}")
            
            # Data information
            print(f"    Data shape: {data.shape}")
            print(f"    Data type: {data.dtype}")
            
            # Target information
            print(f"    Target shape: {target.shape}")
            print(f"    Target type: {target.dtype}")
            
            # Additional statistics
            if hasattr(data, 'min') and hasattr(data, 'max'):
                print(f"    Data range: [{data.min().item():.4f}, {data.max().item():.4f}]")
            
            if hasattr(target, 'unique'):
                unique_targets = target.unique()
                print(f"    Unique targets in batch: {unique_targets.tolist()}")
            
            # Stop after the first batch
            break
            
    except Exception as e:
        print(f"    Error during dataloader debugging: {e}")


class MetricsAggregator:
    """
    Utility class for aggregating metrics across distributed processes.
    
    This class provides methods for collecting and aggregating various types
    of metrics in distributed training scenarios.
    """
    
    @staticmethod
    def aggregate_scalar_metrics(
        metrics: Dict[str, float],
        reduction: str = 'mean'
    ) -> Dict[str, float]:
        """
        Aggregates scalar metrics across processes.
        
        Args:
            metrics (Dict[str, float]): Dictionary of scalar metrics
            reduction (str): Reduction operation ('mean', 'sum', 'min', 'max')
        
        Returns:
            Dict[str, float]: Aggregated metrics
        """
        if not dist.is_initialized() or dist.get_world_size() < 2:
            return metrics
        
        # Convert to tensors
        tensor_metrics = {k: torch.tensor(v, dtype=torch.float32) for k, v in metrics.items()}
        
        # Perform reduction
        for name, tensor in tensor_metrics.items():
            if reduction == 'mean':
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor /= dist.get_world_size()
            elif reduction == 'sum':
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            elif reduction == 'min':
                dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            elif reduction == 'max':
                dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            else:
                raise ValueError(f"Unsupported reduction: {reduction}")
        
        # Convert back to scalars
        return {k: v.item() for k, v in tensor_metrics.items()}
    
    @staticmethod
    def gather_metrics(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Gathers metrics from all processes to rank 0.
        
        Args:
            metrics (Dict[str, Any]): Local metrics dictionary
        
        Returns:
            List[Dict[str, Any]]: List of metrics from all processes (only on rank 0)
        """
        if not dist.is_initialized():
            return [metrics]
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Gather metrics to rank 0
        gathered_metrics = [None] * world_size
        dist.all_gather_object(gathered_metrics, metrics)
        
        return gathered_metrics if rank == 0 else []


# Custom output transform for PyTorch Ignite
def custom_output_transform(x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Custom output transform function for PyTorch Ignite engines.
    
    This function formats the output of training/evaluation steps for use
    with PyTorch Ignite metrics and engines.
    
    Args:
        x (torch.Tensor): Input data
        y (torch.Tensor): Ground truth labels
        y_pred (torch.Tensor): Model predictions
        loss (torch.Tensor): Loss value
    
    Returns:
        Dict[str, torch.Tensor]: Formatted output dictionary
    """
    return {
        "y": y,
        "y_pred": y_pred,
        "criterion_kwargs": {}
    }