"""
Distributed training utilities for PyTorch-based machine learning models.

This module provides functionality for initializing and managing distributed training 
environments using PyTorch's distributed package. It includes utilities for setting up 
communication backends, device management, and integration with TorchDistributor.

Features:
- Automatic backend selection (NCCL for GPU, GLOO for CPU)
- Environment variable validation and parsing
- GPU device assignment based on local rank
- Error handling for distributed training initialization
- Integration with Spark TorchDistributor

Example:
    >>> import torch.distributed as dist
    >>> from distributed import initialize_distributed_training
    >>> 
    >>> # Initialize distributed training
    >>> config = initialize_distributed_training(use_gpu=True)
    >>> print(f"Rank {config['global_rank']} initialized on device {config['device']}")
"""

import os
import datetime
import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Any


def initialize_distributed_training(use_gpu: bool = True) -> Dict[str, Any]:
    """
    Initializes the distributed training environment and sets up the appropriate device.
    
    This function sets up PyTorch's distributed training environment by:
    1. Selecting the appropriate backend (NCCL for GPU, GLOO for CPU)
    2. Initializing the process group with timeout
    3. Parsing environment variables for distributed configuration
    4. Setting up the device based on local rank and GPU availability
    
    Args:
        use_gpu (bool): Whether to use GPU if available. Defaults to True.
    
    Returns:
        dict: A dictionary containing the following keys:
            - 'backend': The backend used for distributed training ('nccl' or 'gloo').
            - 'global_rank': The global rank of the current process.
            - 'local_rank': The local rank of the current process.
            - 'world_size': The total number of processes in the distributed training.
            - 'device': The torch.device object to be used for training.
            - 'device_ids': A list of device IDs for GPU training or None for CPU training.
            - 'env_dict': A dictionary of environment variables related to distributed training.
    
    Raises:
        RuntimeError: If required environment variables are not set or invalid.
        ValueError: If environment variables contain invalid values.
    
    Example:
        >>> config = initialize_distributed_training(use_gpu=True)
        >>> model = model.to(config['device'])
        >>> model = torch.nn.parallel.DistributedDataParallel(
        ...     model, device_ids=config['device_ids']
        ... )
    """
    # Set the appropriate backend based on whether GPU is used
    backend = "nccl" if use_gpu and torch.cuda.is_available() else "gloo"
    
    try:
        # Initialize process group with extended timeout for distributed training
        dist.init_process_group(backend, timeout=datetime.timedelta(seconds=5400))
    except Exception as e:
        raise RuntimeError(f"Failed to initialize process group with backend '{backend}': {e}")
    
    # Gather environment variables needed for distributed training
    env_vars = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]
    env_dict = {key: os.getenv(key) for key in env_vars}
    
    # Validate that all required environment variables are set
    missing_vars = [key for key, value in env_dict.items() if value is None]
    if missing_vars:
        raise RuntimeError(f"Missing required environment variables: {missing_vars}")
    
    try:
        global_rank = int(env_dict["RANK"])
        local_rank = int(env_dict["LOCAL_RANK"])
        world_size = int(env_dict["WORLD_SIZE"])
    except ValueError as e:
        raise ValueError(f"Invalid environment variable values: {e}")
    
    # Validate rank and world_size values
    if global_rank < 0 or global_rank >= world_size:
        raise ValueError(f"Invalid global_rank {global_rank} for world_size {world_size}")
    if local_rank < 0:
        raise ValueError(f"Invalid local_rank {local_rank}")
    
    # Determine the device to use based on local rank
    if use_gpu and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("GPU requested but no CUDA devices available")
        
        device_id = local_rank % num_gpus  # Ensure local_rank maps to a valid GPU on the node
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
        device_ids = [device_id]  # Assign the device_id to DDP
    else:
        device = torch.device("cpu")
        device_ids = None  # No specific device IDs needed for CPU
    
    # Return the gathered information
    return {
        'backend': backend,
        'global_rank': global_rank,
        'local_rank': local_rank,
        'world_size': world_size,
        'device': device,
        'device_ids': device_ids,
        'env_dict': env_dict
    }


def create_distributed_dataloader(
    data_path: str, 
    rank: int, 
    world_size: int, 
    batch_size: int = 64, 
    use_validation: bool = True
):
    """
    Creates distributed data loaders for training, validation, and test datasets.
    
    This function creates PyTorch DataLoaders with DistributedSampler for 
    distributed training scenarios. It loads preprocessed MNIST data and 
    creates appropriate samplers for distributed training.
    
    Args:
        data_path (str): Path to the directory containing processed data files.
        rank (int): Current process rank in the distributed training.
        world_size (int): Total number of processes in distributed training.
        batch_size (int): Batch size for the DataLoaders. Defaults to 64.
        use_validation (bool): If True, uses validation dataset; if False, uses test dataset. 
                              Defaults to True.
    
    Returns:
        tuple: A tuple containing (train_loader, eval_loader, test_loader) where:
            - train_loader: DataLoader for training data with DistributedSampler
            - eval_loader: DataLoader for validation/test data (based on use_validation)
            - test_loader: DataLoader for test data (if use_validation is True)
    
    Raises:
        FileNotFoundError: If the specified data_path does not exist or data files are missing.
        ValueError: If invalid split parameter is provided.
    
    Example:
        >>> train_loader, val_loader, test_loader = create_distributed_dataloader(
        ...     data_path="/path/to/data",
        ...     rank=0,
        ...     world_size=4,
        ...     batch_size=32,
        ...     use_validation=True
        ... )
    """
    from torch.utils.data import DataLoader, DistributedSampler
    # Import ProcessedMNISTDataset from the data module
    from DDP_KBIT.data.datasets import ProcessedMNISTDataset
    
    print(f"Rank {rank}: Creating distributed data loaders...")
    
    try:
        # Create custom datasets
        train_dataset = ProcessedMNISTDataset(data_path, split='train')
        test_dataset = ProcessedMNISTDataset(data_path, split='test')
        val_dataset = ProcessedMNISTDataset(data_path, split='val')
    except Exception as e:
        raise FileNotFoundError(f"Failed to load datasets from {data_path}: {e}")
    
    # Select evaluation dataset
    eval_dataset = val_dataset if use_validation else test_dataset
    eval_split_name = 'validation' if use_validation else 'test'
    
    # Create DistributedSampler for training data
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,  # Total number of processes
        rank=rank,                # Current process rank
        shuffle=True,             # Shuffle data each epoch
        seed=42,                  # Seed for reproducibility
        drop_last=True            # Drop incomplete batches
    )
    
    # Create DistributedSampler for evaluation data
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,            # Don't shuffle evaluation data
        seed=42,
        drop_last=False           # Keep all evaluation data
    )
    
    # Create DistributedSampler for test data
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=42,
        drop_last=False
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=eval_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Rank {rank}: Data loaders created successfully")
    print(f"  - Training dataset: {len(train_dataset)} samples")
    print(f"  - {eval_split_name.capitalize()} dataset: {len(eval_dataset)} samples")
    print(f"  - Test dataset: {len(test_dataset)} samples")
    print(f"  - Batch size: {batch_size}")
    
    return train_loader, eval_loader, test_loader


def validate_distributed_environment() -> bool:
    """
    Validates that the current environment is properly configured for distributed training.
    
    This function checks:
    1. Required environment variables are set
    2. PyTorch distributed package is available
    3. CUDA availability (if GPU training is intended)
    
    Returns:
        bool: True if environment is valid for distributed training, False otherwise.
    
    Example:
        >>> if validate_distributed_environment():
        ...     config = initialize_distributed_training()
        ... else:
        ...     print("Environment not configured for distributed training")
    """
    required_env_vars = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]
    
    # Check if all required environment variables are set
    missing_vars = [var for var in required_env_vars if os.getenv(var) is None]
    if missing_vars:
        print(f"Missing environment variables: {missing_vars}")
        return False
    
    # Check if PyTorch distributed is available
    if not torch.distributed.is_available():
        print("PyTorch distributed is not available")
        return False
    
    # Validate environment variable values
    try:
        rank = int(os.getenv("RANK"))
        world_size = int(os.getenv("WORLD_SIZE"))
        local_rank = int(os.getenv("LOCAL_RANK"))
        
        if rank < 0 or rank >= world_size:
            print(f"Invalid rank {rank} for world_size {world_size}")
            return False
        
        if local_rank < 0:
            print(f"Invalid local_rank {local_rank}")
            return False
            
    except ValueError as e:
        print(f"Invalid environment variable values: {e}")
        return False
    
    return True


def cleanup_distributed_training():
    """
    Cleans up the distributed training environment.
    
    This function should be called at the end of distributed training
    to properly clean up process groups and resources.
    
    Example:
        >>> try:
        ...     # Training code here
        ...     pass
        ... finally:
        ...     cleanup_distributed_training()
    """
    if torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
            print("Distributed process group destroyed successfully")
        except Exception as e:
            print(f"Warning: Failed to destroy process group: {e}")


def get_distributed_info() -> Dict[str, Any]:
    """
    Gets information about the current distributed training setup.
    
    Returns:
        dict: Dictionary containing distributed training information including
              rank, world_size, backend, and device information.
    
    Example:
        >>> info = get_distributed_info()
        >>> print(f"Running on rank {info['rank']} of {info['world_size']}")
    """
    if not torch.distributed.is_initialized():
        return {
            'initialized': False,
            'rank': 0,
            'world_size': 1,
            'backend': None,
            'device': torch.device('cpu')
        }
    
    return {
        'initialized': True,
        'rank': torch.distributed.get_rank(),
        'world_size': torch.distributed.get_world_size(),
        'backend': torch.distributed.get_backend(),
        'device': torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    }


# TorchDistributor integration utilities
class DistributedTrainingConfig:
    """
    Configuration class for distributed training parameters.
    
    This class encapsulates all the configuration needed for distributed training
    and provides validation and default values.
    
    Attributes:
        use_gpu (bool): Whether to use GPU for training
        backend (str): Backend for distributed training ('nccl' or 'gloo')
        timeout (datetime.timedelta): Timeout for process group initialization
        find_unused_parameters (bool): Whether to find unused parameters in DDP
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        backend: Optional[str] = None,
        timeout_seconds: int = 5400,
        find_unused_parameters: bool = False
    ):
        self.use_gpu = use_gpu
        self.backend = backend or ("nccl" if use_gpu and torch.cuda.is_available() else "gloo")
        self.timeout = datetime.timedelta(seconds=timeout_seconds)
        self.find_unused_parameters = find_unused_parameters
    
    def validate(self) -> None:
        """Validates the configuration parameters."""
        if self.use_gpu and not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA is not available")
        
        if self.backend not in ['nccl', 'gloo']:
            raise ValueError(f"Unsupported backend: {self.backend}")
        
        if self.backend == 'nccl' and not torch.cuda.is_available():
            raise ValueError("NCCL backend requires CUDA")
    
    def to_dict(self) -> Dict[str, Any]:
        """Returns configuration as a dictionary."""
        return {
            'use_gpu': self.use_gpu,
            'backend': self.backend,
            'timeout': self.timeout,
            'find_unused_parameters': self.find_unused_parameters
        }


def is_distributed_available() -> bool:
    """
    Check if distributed training is available and properly configured.
    
    Returns:
        bool: True if distributed training is available, False otherwise.
    """
    return torch.distributed.is_available() and validate_distributed_environment()


def get_rank() -> int:
    """
    Get the rank of the current process in distributed training.
    
    Returns:
        int: Current process rank, or 0 if not in distributed mode.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    """
    Get the total number of processes in distributed training.
    
    Returns:
        int: Total number of processes, or 1 if not in distributed mode.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1