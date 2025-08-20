"""
Configuration Module for DDP_KBIT

This module contains all configuration-related functionality for the DDP_KBIT
distributed deep learning system, including training parameters, data settings,
and Spark configuration.
"""

from DDP_KBIT.config import training_config
from DDP_KBIT.config import data_config
from DDP_KBIT.config import spark_config

# Import key configuration functions and classes
from DDP_KBIT.config.training_config import (
    get_extended_training_config,
    NeuralNetwork
)

from DDP_KBIT.config.data_config import (
    KAFKA_CONFIG,
    PAYLOAD_CONFIG,
    DATA_LOADER_CONFIG,
    MONGO_CONFIG,
    MNIST_AVRO_SCHEMA_V1,
    MNIST_AVRO_SCHEMA_V2,
    transform_RawMNISTData,
    transform_mongodb_image,
    get_data_loader_config_for_experiment,
    get_data_loader_config_with_topic
)

from DDP_KBIT.config.spark_config import (
    SPARK_CONFIG,
    JAR_URLS,
    create_spark_session,
    create_spark_session_original_style,
    get_spark_config_for_num_processes,
    load_external_config,
    validate_spark_config,
    print_spark_config_summary
)

__all__ = [
    # Modules
    'training_config',
    'data_config',
    'spark_config',
    
    # Training configuration
    'get_extended_training_config',
    'NeuralNetwork',
    
    # Data configuration
    'KAFKA_CONFIG',
    'PAYLOAD_CONFIG',
    'DATA_LOADER_CONFIG',
    'MONGO_CONFIG',
    'MNIST_AVRO_SCHEMA_V1',
    'MNIST_AVRO_SCHEMA_V2',
    'transform_RawMNISTData',
    'transform_mongodb_image',
    'get_data_loader_config_for_experiment',
    'get_data_loader_config_with_topic',
    
    # Spark configuration
    'SPARK_CONFIG',
    'JAR_URLS',
    'create_spark_session',
    'create_spark_session_original_style',
    'get_spark_config_for_num_processes',
    'load_external_config',
    'validate_spark_config',
    'print_spark_config_summary',
]