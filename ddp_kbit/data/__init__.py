"""
DDP_KBIT Data Module

This package contains data processing utilities for distributed deep learning:
- datasets: Dataset classes for distributed Kafka consumption and MNIST data
- data_fetcher: Coordinated data fetching across distributed processes
- transforms: Data transformation functions for various formats
- kafka_utils: Kafka utilities for partition management and offset splitting
- loaders: DataLoader creation utilities for distributed and standard training
"""

from ddp_kbit.data.datasets import (
    DistributedDataset,
    ProcessedMNISTDataset,
    create_distributed_dataloader
)

from ddp_kbit.data.data_fetcher import DistributedDataFetcher

from ddp_kbit.data.transforms import (
    transform_MNISTData,
    transform_RawMNISTData,
    transform_mongodb_image,
    mnist_avro_schema,
    create_transform_pipeline,
    transform_image_from_pil,
    transform_numpy_to_tensor
)

from ddp_kbit.data.kafka_utils import (
    split_offsets,
    validate_split_config,
    parse_offset_data,
    calculate_partition_ranges,
    update_offset_diffs,
    print_results,
    create_dynamic_url,
    check_partitions,
    get_kafka_topic_info,
    validate_kafka_connection,
    estimate_data_distribution
)

from ddp_kbit.data.loaders import (
    create_dataloaders,
    create_simple_dataloader,
    create_distributed_sampler,
    create_dataloader_with_sampler,
    create_train_val_test_loaders,
    create_mnist_dataloaders,
    get_optimal_num_workers,
    print_dataloader_info,
    calculate_dataloader_stats
)

__all__ = [
    # datasets
    'DistributedDataset',
    'ProcessedMNISTDataset', 
    'create_distributed_dataloader',
    
    # data_fetcher
    'DistributedDataFetcher',
    
    # transforms
    'transform_MNISTData',
    'transform_RawMNISTData',
    'transform_mongodb_image',
    'mnist_avro_schema',
    'create_transform_pipeline',
    'transform_image_from_pil',
    'transform_numpy_to_tensor',
    
    # kafka_utils
    'split_offsets',
    'validate_split_config', 
    'parse_offset_data',
    'calculate_partition_ranges',
    'update_offset_diffs',
    'print_results',
    'create_dynamic_url',
    'check_partitions',
    'get_kafka_topic_info',
    'validate_kafka_connection',
    'estimate_data_distribution',
    
    # loaders
    'create_dataloaders',
    'create_simple_dataloader',
    'create_distributed_sampler',
    'create_dataloader_with_sampler', 
    'create_train_val_test_loaders',
    'create_mnist_dataloaders',
    'get_optimal_num_workers',
    'print_dataloader_info',
    'calculate_dataloader_stats'
]