"""
Configuration Module for DDP_KBIT

This module contains all configuration-related functionality for the DDP_KBIT
distributed deep learning system, including training parameters, data settings,
and Spark configuration.
"""

from . import training_config
from . import data_config
from . import spark_config

# Import key configuration functions and classes
from .training_config import (
    get_extended_training_config,
    NeuralNetwork
)

from .data_config import (
    get_payload_config,
    get_data_loader_config,
    get_mongo_config,
    transform_RawMNISTData,
    transform_mongodb_image,
    mnist_avro_schema_v1,
    mnist_avro_schema_v2
)

from .spark_config import (
    get_spark_config,
    load_config_from_file,
    validate_config,
    jar_files
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
    'get_payload_config',
    'get_data_loader_config',
    'get_mongo_config',
    'transform_RawMNISTData',
    'transform_mongodb_image',
    'mnist_avro_schema_v1',
    'mnist_avro_schema_v2',
    
    # Spark configuration
    'get_spark_config',
    'load_config_from_file',
    'validate_config',
    'jar_files',
]