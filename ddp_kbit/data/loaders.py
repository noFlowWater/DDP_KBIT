"""
DataLoader creation utilities for distributed and standard training scenarios.

This module contains utilities for creating PyTorch DataLoader objects:
- create_dataloaders: Create DataLoaders from dataset list
- Various helper functions for DataLoader configuration
"""

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from ddp_kbit.data.datasets import ProcessedMNISTDataset


def create_dataloaders(datasets, batch_size):
    """
    Create DataLoaders based on datasets.
    
    Args:
        datasets (list): Dataset list (maximum 3 datasets)
        batch_size (int): Batch size to apply to each DataLoader
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
            - train_loader (DataLoader): DataLoader for training dataset
            - val_loader (DataLoader or None): DataLoader for validation dataset (if exists)
            - test_loader (DataLoader or None): DataLoader for test dataset (if exists)
    
    Raises:
        ValueError: If dataset list is empty or has more than 3 elements
    """
    if len(datasets) == 0:
        raise ValueError("데이터셋 리스트가 비어 있습니다. 최소한 1개의 데이터셋이 필요합니다.")
    
    # Train dataset is required
    train_sampler = RandomSampler(datasets[0])
    train_loader = DataLoader(datasets[0], batch_size=batch_size, sampler=train_sampler, pin_memory=True)

    # Create loaders based on number of datasets
    if len(datasets) == 2:
        # Train & Test datasets
        test_loader = DataLoader(datasets[1], batch_size=batch_size, shuffle=False, pin_memory=True)
        val_loader = None
    elif len(datasets) == 3:
        # Train & Validation & Test datasets
        val_loader = DataLoader(datasets[1], batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(datasets[2], batch_size=batch_size, shuffle=False, pin_memory=True)
    else:
        val_loader = None
        test_loader = None

    if len(datasets) > 3:
        raise ValueError("데이터셋 리스트에 4개 이상의 요소가 있습니다. 최대 3개의 데이터셋만 처리할 수 있습니다.")

    return train_loader, val_loader, test_loader


def create_simple_dataloader(dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=False):
    """
    Create a simple DataLoader with standard configurations.
    
    Args:
        dataset: PyTorch dataset
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of worker processes
        pin_memory (bool): Whether to use pinned memory
        drop_last (bool): Whether to drop the last incomplete batch
    
    Returns:
        DataLoader: Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def create_distributed_sampler(dataset, world_size, rank, shuffle=True, seed=42, drop_last=False):
    """
    Create a DistributedSampler for distributed training.
    
    Args:
        dataset: PyTorch dataset
        world_size (int): Total number of processes
        rank (int): Current process rank
        shuffle (bool): Whether to shuffle data
        seed (int): Random seed for reproducibility
        drop_last (bool): Whether to drop the last incomplete batch
    
    Returns:
        DistributedSampler: Configured distributed sampler
    """
    return DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last
    )


def create_dataloader_with_sampler(dataset, sampler, batch_size, num_workers=2, pin_memory=True, persistent_workers=True):
    """
    Create a DataLoader with a custom sampler.
    
    Args:
        dataset: PyTorch dataset
        sampler: Data sampler (e.g., DistributedSampler, RandomSampler)
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        pin_memory (bool): Whether to use pinned memory
        persistent_workers (bool): Whether to keep workers alive between epochs
    
    Returns:
        DataLoader: Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )


def get_optimal_num_workers(max_workers=None):
    """
    Get optimal number of workers for DataLoader based on CPU count.
    
    Args:
        max_workers (int, optional): Maximum number of workers to use
    
    Returns:
        int: Optimal number of workers
    """
    import os
    
    cpu_count = os.cpu_count() or 1
    
    # Generally, 2-4 workers per CPU core works well, but we cap it for memory reasons
    optimal = min(cpu_count * 2, 8)
    
    if max_workers is not None:
        optimal = min(optimal, max_workers)
    
    return optimal


def create_train_val_test_loaders(
    train_dataset, 
    val_dataset, 
    test_dataset, 
    batch_size, 
    world_size=None, 
    rank=None,
    num_workers=None,
    pin_memory=True,
    persistent_workers=True
):
    """
    Create train, validation, and test DataLoaders with optional distributed support.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size (int): Batch size
        world_size (int, optional): Total number of processes for distributed training
        rank (int, optional): Current process rank for distributed training
        num_workers (int, optional): Number of worker processes
        pin_memory (bool): Whether to use pinned memory
        persistent_workers (bool): Whether to keep workers alive between epochs
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if num_workers is None:
        num_workers = get_optimal_num_workers()
    
    # Check if distributed training is requested
    if world_size is not None and rank is not None:
        # Create distributed samplers
        train_sampler = create_distributed_sampler(
            train_dataset, world_size, rank, shuffle=True, drop_last=True
        )
        val_sampler = create_distributed_sampler(
            val_dataset, world_size, rank, shuffle=False, drop_last=False
        )
        test_sampler = create_distributed_sampler(
            test_dataset, world_size, rank, shuffle=False, drop_last=False
        )
        
        # Create DataLoaders with samplers
        train_loader = create_dataloader_with_sampler(
            train_dataset, train_sampler, batch_size, num_workers, pin_memory, persistent_workers
        )
        val_loader = create_dataloader_with_sampler(
            val_dataset, val_sampler, batch_size, num_workers, pin_memory, persistent_workers
        )
        test_loader = create_dataloader_with_sampler(
            test_dataset, test_sampler, batch_size, num_workers, pin_memory, persistent_workers
        )
    else:
        # Create standard DataLoaders
        train_loader = create_simple_dataloader(
            train_dataset, batch_size, shuffle=True, num_workers=num_workers, 
            pin_memory=pin_memory, drop_last=True
        )
        val_loader = create_simple_dataloader(
            val_dataset, batch_size, shuffle=False, num_workers=num_workers, 
            pin_memory=pin_memory, drop_last=False
        )
        test_loader = create_simple_dataloader(
            test_dataset, batch_size, shuffle=False, num_workers=num_workers, 
            pin_memory=pin_memory, drop_last=False
        )
    
    return train_loader, val_loader, test_loader


def create_mnist_dataloaders(
    data_path, 
    batch_size=64, 
    world_size=None, 
    rank=None, 
    use_validation=True,
    num_workers=None
):
    """
    Create MNIST DataLoaders using ProcessedMNISTDataset.
    
    Args:
        data_path (str): Path to processed MNIST data
        batch_size (int): Batch size
        world_size (int, optional): Total number of processes for distributed training
        rank (int, optional): Current process rank for distributed training
        use_validation (bool): Whether to use validation split instead of test for evaluation
        num_workers (int, optional): Number of worker processes
    
    Returns:
        tuple: (train_loader, eval_loader, test_loader)
    """
    # Create datasets
    train_dataset = ProcessedMNISTDataset(data_path, split='train')
    val_dataset = ProcessedMNISTDataset(data_path, split='val')
    test_dataset = ProcessedMNISTDataset(data_path, split='test')
    
    # Select evaluation dataset
    eval_dataset = val_dataset if use_validation else test_dataset
    
    # Create loaders
    train_loader, eval_loader, test_loader = create_train_val_test_loaders(
        train_dataset=train_dataset,
        val_dataset=eval_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers
    )
    
    return train_loader, eval_loader, test_loader


def print_dataloader_info(loader, name="DataLoader"):
    """
    Print information about a DataLoader.
    
    Args:
        loader (DataLoader): DataLoader to inspect
        name (str): Name for the DataLoader (for printing)
    """
    print(f"{name} Information:")
    print(f"  Dataset size: {len(loader.dataset)}")
    print(f"  Batch size: {loader.batch_size}")
    print(f"  Number of batches: {len(loader)}")
    print(f"  Number of workers: {loader.num_workers}")
    print(f"  Pin memory: {loader.pin_memory}")
    print(f"  Drop last: {loader.drop_last}")
    if hasattr(loader, 'sampler'):
        print(f"  Sampler type: {type(loader.sampler).__name__}")


def calculate_dataloader_stats(loaders_dict):
    """
    Calculate statistics for multiple DataLoaders.
    
    Args:
        loaders_dict (dict): Dictionary of {name: DataLoader} pairs
    
    Returns:
        dict: Statistics for each DataLoader
    """
    stats = {}
    
    for name, loader in loaders_dict.items():
        if loader is not None:
            stats[name] = {
                'dataset_size': len(loader.dataset),
                'batch_size': loader.batch_size,
                'num_batches': len(loader),
                'total_samples_per_epoch': len(loader) * loader.batch_size,
                'num_workers': loader.num_workers,
                'pin_memory': loader.pin_memory,
                'drop_last': loader.drop_last,
                'sampler_type': type(loader.sampler).__name__ if hasattr(loader, 'sampler') else None
            }
        else:
            stats[name] = None
    
    return stats