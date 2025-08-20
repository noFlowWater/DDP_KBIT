"""
Configuration Module for ddp_kbit

This module contains all configuration-related functionality for the ddp_kbit
distributed deep learning system, including training parameters, data settings,
and Spark configuration.
"""

from ddp_kbit.config import training_config
from ddp_kbit.config import data_config
from ddp_kbit.config import spark_config

# Import key configuration functions and classes
from ddp_kbit.config.training_config import (
    get_extended_training_config,
    NeuralNetwork
)

from ddp_kbit.config.data_config import (
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

from ddp_kbit.config.spark_config import (
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