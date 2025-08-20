"""
Main training module for distributed machine learning with PyTorch Ignite.

This module provides the core training functionality for distributed machine learning,
including the main training function, PyTorch Ignite integration, and comprehensive
training loop management. It supports both Kafka-based and local data loading,
with full distributed training capabilities.

Features:
- Main training function with distributed support
- PyTorch Ignite engine integration
- Flexible data loading (Kafka/local)
- Comprehensive evaluation and testing
- Performance monitoring and metrics tracking
- Error handling and cleanup

Example:
    >>> from trainer import main_fn
    >>> from distributed import initialize_distributed_training
    >>> 
    >>> # Configure training
    >>> training_config = {
    ...     'num_epochs': 10,
    ...     'batch_size': 64,
    ...     'perform_validation': True,
    ...     'base_model': model,
    ...     'optimizer_class': torch.optim.Adam,
    ...     'optimizer_params': {'lr': 0.001},
    ...     'loss_fn': torch.nn.CrossEntropyLoss(),
    ...     'metrics': {'accuracy': Accuracy(), 'loss': Loss(loss_fn)}
    ... }
    >>> 
    >>> # Run distributed training
    >>> results = main_fn(training_config, kafka_config, data_loader_config, use_gpu=True)
"""

import os
import time
import torch
import torch.distributed as dist
from typing import Dict, List, Any, Optional, Tuple, Union

# PyTorch Ignite imports
import ignite.distributed as idist
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Metric

# Local imports
from DDP_KBIT.training.distributed import initialize_distributed_training, create_distributed_dataloader, cleanup_distributed_training
from DDP_KBIT.training.metrics import (
    save_metrics, custom_output_transform, debug_dataloader, 
    TrainingMetricsTracker, PerformanceProfiler
)
from DDP_KBIT.data.data_fetcher import DistributedDataFetcher
from DDP_KBIT.data.loaders import create_dataloaders

def main_fn(
    training_config: Dict[str, Any],
    kafka_config: Dict[str, Any],
    data_loader_config: Dict[str, Any],
    use_gpu: bool = True
) -> Dict[str, Dict[str, List]]:
    """
    Main distributed training function with PyTorch Ignite integration.
    
    This function orchestrates the complete distributed training process including:
    - Distributed environment initialization
    - Data loading (Kafka or local)
    - Model and optimizer setup
    - Training loop with validation and testing
    - Metrics tracking and performance monitoring
    - Cleanup and result aggregation
    
    Args:
        training_config (Dict[str, Any]): Configuration for training including:
            - num_epochs (int): Number of training epochs
            - batch_size (int): Batch size for training
            - perform_validation (bool): Whether to perform validation
            - base_model (torch.nn.Module): Base model to train
            - optimizer_class (type): Optimizer class (e.g., torch.optim.Adam)
            - optimizer_params (Dict): Parameters for optimizer initialization
            - loss_fn (callable): Loss function
            - metrics (Dict[str, Metric]): Dictionary of Ignite metrics
        
        kafka_config (Dict[str, Any]): Kafka configuration for data streaming
        
        data_loader_config (Dict[str, Any]): Data loader configuration including:
            - data_loader_type (str): 'kafka' or 'local'
            - local_data_path (str): Path for local data (if data_loader_type='local')
            - Other data loading parameters
        
        use_gpu (bool): Whether to use GPU for training. Defaults to True.
    
    Returns:
        Dict[str, Dict[str, List]]: Training results containing:
            - train_metrics (Dict[str, List]): Training metrics history
            - val_metrics (Dict[str, List]): Validation metrics history
            - test_metrics (Dict[str, List]): Test metrics history
    
    Raises:
        ValueError: If data loader configuration is invalid or no datasets found
        RuntimeError: If distributed training initialization fails
        FileNotFoundError: If local data path is invalid (for local data loading)
    
    Example:
        >>> training_config = {
        ...     'num_epochs': 10,
        ...     'batch_size': 64,
        ...     'perform_validation': True,
        ...     'base_model': model,
        ...     'optimizer_class': torch.optim.Adam,
        ...     'optimizer_params': {'lr': 0.001},
        ...     'loss_fn': torch.nn.CrossEntropyLoss(),
        ...     'metrics': {'accuracy': Accuracy(), 'loss': Loss(loss_fn)}
        ... }
        >>> 
        >>> results = main_fn(training_config, kafka_config, data_loader_config, True)
        >>> print(f"Final test accuracy: {results['test_metrics']['accuracy'][-1]}")
    """

    # Initialize distributed training environment
    print("Starting distributed training initialization...")
    
    try:
        init_config = initialize_distributed_training(use_gpu)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize distributed training: {e}")
    
    # Get distributed info from ignite
    print(f"{idist.get_rank()=}\n{idist.backend()=}\n{idist.get_world_size()=}")
    device = idist.device()
    print(f"{device=} & {type(device)=}")
    
    # Log process information
    print(f"[PID {os.getpid()}] Initializing process group with: {init_config['env_dict']}")
    print(f"[PID {os.getpid()}] world_size = {init_config['world_size']}, "
          f"global_rank = {init_config['global_rank']}, local_rank = {init_config['local_rank']}, "
          f"backend = {init_config['backend']}, device_ids = {init_config['device_ids']}")
    
    # Initialize performance profiler and metrics tracker
    profiler = PerformanceProfiler()
    metrics_tracker = TrainingMetricsTracker()
    metrics_tracker.start_training()
    
    # Start data loading
    profiler.start_timer('data_loading')
    start_time = time.time()
    
    print(f"RANK[{init_config['global_rank']}] DataLoader Config: {data_loader_config}")
    
    try:
        train_loader, val_loader, test_loader = _load_data(
            data_loader_config, training_config, init_config, profiler
        )
    except Exception as e:
        cleanup_distributed_training()
        raise ValueError(f"Failed to load data: {e}")
    
    end_time = time.time()
    data_load_time = profiler.end_timer('data_loading')
    print(f'Finished Data Loading. Total time: {data_load_time:.2f} seconds')
    
    # Debug data loaders
    _debug_data_loaders(train_loader, val_loader, test_loader, init_config['global_rank'])
    
    # Initialize results dictionary
    results = {
        'train_metrics': {},
        'val_metrics': {},
        'test_metrics': {}
    }
    
    # Setup model, optimizer, and training components
    try:
        model, optimizer, trainer, evaluator, tester = _setup_training_components(
            training_config, device, init_config['global_rank']
        )
    except Exception as e:
        cleanup_distributed_training()
        raise RuntimeError(f"Failed to setup training components: {e}")
    
    # Setup training event handlers
    _setup_training_handlers(
        trainer, evaluator, tester, val_loader, test_loader, 
        results, metrics_tracker, init_config, training_config
    )
    
    # Execute training
    try:
        _execute_training(
            trainer, train_loader, training_config, 
            profiler, metrics_tracker, init_config['global_rank']
        )
    except Exception as e:
        cleanup_distributed_training()
        raise RuntimeError(f"Training failed: {e}")
    
    # Execute testing if test data is available
    if test_loader is not None:
        try:
            _execute_testing(
                tester, test_loader, results, 
                profiler, init_config['global_rank']
            )
        except Exception as e:
            print(f"Warning: Testing failed: {e}")
    
    # Print final summaries
    if init_config['global_rank'] == 0:
        metrics_tracker.print_summary(rank=0)
        profiler.print_summary(rank=0)
    
    # Cleanup distributed environment
    cleanup_distributed_training()
    
    return results


def _load_data(
    data_loader_config: Dict[str, Any],
    training_config: Dict[str, Any],
    init_config: Dict[str, Any],
    profiler: PerformanceProfiler
) -> Tuple[Any, Any, Any]:
    """
    Loads training, validation, and test data based on configuration.
    
    Args:
        data_loader_config: Data loader configuration
        training_config: Training configuration
        init_config: Distributed initialization configuration
        profiler: Performance profiler instance
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if data_loader_config['data_loader_type'] == "kafka":
        return _load_kafka_data(data_loader_config, training_config, init_config, profiler)
    elif data_loader_config['data_loader_type'] == "local":
        return _load_local_data(data_loader_config, training_config, init_config)
    else:
        raise ValueError(f"Unsupported data_loader_type: {data_loader_config['data_loader_type']}")


def _load_kafka_data(
    data_loader_config: Dict[str, Any],
    training_config: Dict[str, Any],
    init_config: Dict[str, Any],
    profiler: PerformanceProfiler
) -> Tuple[Any, Any, Any]:
    """Loads data from Kafka streams."""
    profiler.start_timer('kafka_data_fetch')
    
    # Fetch data using DistributedDataFetcher
    try:
        data_fetcher = DistributedDataFetcher(data_loader_config, device=init_config["device"])
        splited_datasets = data_fetcher.fetch_n_pull_splited_datasets()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data from Kafka: {e}")
    
    profiler.end_timer('kafka_data_fetch')
    
    if len(splited_datasets) == 0:
        raise ValueError("No datasets found. At least 1 dataset is required.")
    
    # Adjust batch size for distributed training
    adjust_batch_size = int(training_config["batch_size"] / float(init_config['world_size']))
    print(f"Rank {init_config['global_rank']}: Creating DataLoader with batch size {adjust_batch_size}")
    
    train_loader, val_loader, test_loader = create_dataloaders(splited_datasets, adjust_batch_size)
    
    return train_loader, val_loader, test_loader


def _load_local_data(
    data_loader_config: Dict[str, Any],
    training_config: Dict[str, Any],
    init_config: Dict[str, Any]
) -> Tuple[Any, Any, Any]:
    """Loads data from local files."""
    if 'local_data_path' not in data_loader_config:
        raise ValueError("local_data_path is required for local data loading")
    
    train_loader, val_loader, test_loader = create_distributed_dataloader(
        data_loader_config['local_data_path'],
        init_config['global_rank'],
        init_config['world_size'],
        training_config['batch_size'],
        training_config['perform_validation']
    )
    
    return train_loader, val_loader, test_loader


def _debug_data_loaders(
    train_loader: Any,
    val_loader: Optional[Any],
    test_loader: Optional[Any],
    rank: int
) -> None:
    """Debug data loaders by printing information about their contents."""
    print(f"RANK[{rank}] Train Loader Debug:")
    debug_dataloader(train_loader, rank)
    
    if val_loader is not None:
        print(f"RANK[{rank}] Validation Loader Debug:")
        debug_dataloader(val_loader, rank)
    
    if test_loader is not None:
        print(f"RANK[{rank}] Test Loader Debug:")
        debug_dataloader(test_loader, rank)


def _setup_training_components(
    training_config: Dict[str, Any],
    device: torch.device,
    rank: int
) -> Tuple[Any, Any, Any, Any, Any]:
    """
    Sets up model, optimizer, and training engines.
    
    Returns:
        Tuple of (model, optimizer, trainer, evaluator, tester)
    """
    # Setup model and optimizer with ignite auto wrappers
    model = idist.auto_model(training_config['base_model'])
    optimizer = idist.auto_optim(
        training_config['optimizer_class'](
            model.parameters(), 
            **training_config['optimizer_params']
        )
    )
    
    if rank == 0:
        print(f"{type(model)=}, {model=}\n"
              f"{type(optimizer)=}, {optimizer=}\n"
              f"{type(training_config['loss_fn'])=}, {training_config['loss_fn']=}")
    
    # Create trainer engine
    trainer = create_supervised_trainer(
        model, 
        optimizer, 
        training_config['loss_fn'], 
        device=device, 
        output_transform=custom_output_transform
    )
    
    # Attach metrics to trainer
    for name, metric in training_config['metrics'].items():
        metric.attach(trainer, name)
    
    # Create evaluator for validation
    evaluator = None
    if training_config['perform_validation']:
        evaluator = create_supervised_evaluator(
            model, 
            metrics=training_config['metrics'], 
            device=device
        )
    
    # Create tester for final evaluation
    tester = create_supervised_evaluator(
        model, 
        metrics=training_config['metrics'], 
        device=device
    )
    
    return model, optimizer, trainer, evaluator, tester


def _setup_training_handlers(
    trainer: Any,
    evaluator: Optional[Any],
    tester: Any,
    val_loader: Optional[Any],
    test_loader: Optional[Any],
    results: Dict[str, Dict[str, List]],
    metrics_tracker: TrainingMetricsTracker,
    init_config: Dict[str, Any],
    training_config: Dict[str, Any]
) -> None:
    """Sets up event handlers for training, validation, and testing."""
    
    @trainer.on(Events.EPOCH_STARTED)
    def on_epoch_started(engine):
        metrics_tracker.start_epoch()
        if init_config['global_rank'] == 0:
            print(f"\nStarting Epoch {engine.state.epoch}/{training_config['num_epochs']}")
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def on_epoch_completed(engine):
        # Save training metrics
        save_metrics(engine.state.metrics.copy().items(), results['train_metrics'])
        metrics_tracker.save_metrics(
            {k: v for k, v in engine.state.metrics.items()}, 
            phase='train'
        )
        
        # Run validation if configured
        if training_config['perform_validation'] and evaluator is not None and val_loader is not None:
            evaluator.run(val_loader)
        
        # Print metrics summary
        if init_config['global_rank'] == 0:
            print(f"\nTrain Epoch: {engine.state.epoch}")
            for results_key, results_value in results.items():
                for metric_key, metric_value_list in results_value.items():
                    if metric_value_list and metric_value_list[-1] is not None:
                        print(f"{results_key} {metric_key} = {metric_value_list[-1]}")
        
        metrics_tracker.end_epoch()
    
    # Setup validation handler if validation is enabled
    if training_config['perform_validation'] and evaluator is not None:
        @evaluator.on(Events.COMPLETED)
        def get_validation_metrics(engine):
            save_metrics(engine.state.metrics.copy().items(), results['val_metrics'])
            metrics_tracker.save_metrics(
                {k: v for k, v in engine.state.metrics.items()}, 
                phase='val'
            )


def _execute_training(
    trainer: Any,
    train_loader: Any,
    training_config: Dict[str, Any],
    profiler: PerformanceProfiler,
    metrics_tracker: TrainingMetricsTracker,
    rank: int
) -> None:
    """Executes the main training loop."""
    profiler.start_timer('training')
    
    if rank == 0:
        print(f"\nStarting training for {training_config['num_epochs']} epochs...")
    
    try:
        trainer.run(train_loader, max_epochs=training_config['num_epochs'])
    except Exception as e:
        raise RuntimeError(f"Training loop failed: {e}")
    
    training_time = profiler.end_timer('training')
    
    if rank == 0:
        print(f'_ _ _ _ Finished training. Total training time: {training_time:.2f} seconds _ _ _ _')


def _execute_testing(
    tester: Any,
    test_loader: Any,
    results: Dict[str, Dict[str, List]],
    profiler: PerformanceProfiler,
    rank: int
) -> None:
    """Executes testing on the test dataset."""
    
    @tester.on(Events.COMPLETED)
    def get_test_metrics(engine):
        save_metrics(engine.state.metrics.copy().items(), results['test_metrics'])
    
    profiler.start_timer('testing')
    
    try:
        tester.run(test_loader)
    except Exception as e:
        raise RuntimeError(f"Testing failed: {e}")
    
    testing_time = profiler.end_timer('testing')
    
    if rank == 0:
        print("\nTest Results:")
        for key, values_list in results["test_metrics"].items():
            if values_list and values_list[-1] is not None:
                print(f"test_metrics {key} = {values_list[-1]}")
        
        print(f'_ _ _ _ Finished testing. Total testing time: {testing_time:.2f} seconds _ _ _ _')


class TrainingConfig:
    """
    Configuration class for training parameters with validation.
    
    This class encapsulates all training configuration parameters and provides
    validation methods to ensure configuration consistency.
    """
    
    def __init__(
        self,
        num_epochs: int,
        batch_size: int,
        base_model: torch.nn.Module,
        optimizer_class: type,
        optimizer_params: Dict[str, Any],
        loss_fn: callable,
        metrics: Dict[str, Metric],
        perform_validation: bool = True,
        **kwargs
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.base_model = base_model
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.perform_validation = perform_validation
        
        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def validate(self) -> None:
        """Validates the training configuration."""
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if not isinstance(self.base_model, torch.nn.Module):
            raise TypeError("base_model must be a torch.nn.Module")
        
        if not hasattr(self.optimizer_class, '__call__'):
            raise TypeError("optimizer_class must be callable")
        
        if not isinstance(self.optimizer_params, dict):
            raise TypeError("optimizer_params must be a dictionary")
        
        if not hasattr(self.loss_fn, '__call__'):
            raise TypeError("loss_fn must be callable")
        
        if not isinstance(self.metrics, dict):
            raise TypeError("metrics must be a dictionary")
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to dictionary."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if key == 'base_model':
                config_dict[key] = str(type(value))
            elif key == 'optimizer_class':
                config_dict[key] = value.__name__
            elif key == 'loss_fn':
                config_dict[key] = str(type(value))
            elif key == 'metrics':
                config_dict[key] = {k: str(type(v)) for k, v in value.items()}
            else:
                config_dict[key] = value
        
        return config_dict


class DataLoaderConfig:
    """Configuration class for data loading parameters."""
    
    def __init__(
        self,
        data_loader_type: str,
        local_data_path: Optional[str] = None,
        **kwargs
    ):
        self.data_loader_type = data_loader_type
        self.local_data_path = local_data_path
        
        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def validate(self) -> None:
        """Validates the data loader configuration."""
        if self.data_loader_type not in ['kafka', 'local']:
            raise ValueError("data_loader_type must be 'kafka' or 'local'")
        
        if self.data_loader_type == 'local' and self.local_data_path is None:
            raise ValueError("local_data_path is required when data_loader_type is 'local'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to dictionary."""
        return self.__dict__.copy()


# Utility functions for common training scenarios
def create_simple_training_config(
    model: torch.nn.Module,
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    optimizer_class: type = torch.optim.Adam
) -> Dict[str, Any]:
    """
    Creates a simple training configuration with common defaults.
    
    Args:
        model: PyTorch model to train
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        optimizer_class: Optimizer class to use
    
    Returns:
        Dictionary containing training configuration
    """
    from ignite.metrics import Loss, Accuracy
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    return {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'perform_validation': True,
        'base_model': model,
        'optimizer_class': optimizer_class,
        'optimizer_params': {'lr': learning_rate},
        'loss_fn': loss_fn,
        'metrics': {
            'accuracy': Accuracy(),
            'loss': Loss(loss_fn)
        }
    }


def create_local_data_config(data_path: str) -> Dict[str, Any]:
    """
    Creates a data loader configuration for local data.
    
    Args:
        data_path: Path to local data directory
    
    Returns:
        Dictionary containing data loader configuration
    """
    return {
        'data_loader_type': 'local',
        'local_data_path': data_path
    }