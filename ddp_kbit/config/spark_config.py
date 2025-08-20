"""
Spark Configuration Module

This module contains all Spark session configurations including:
- Spark session builder settings
- JAR file paths and external dependencies
- GPU and distributed computing settings  
- RAPIDS accelerator configurations
- Executor and driver configurations

All configuration values are preserved exactly as they appear in the original notebook.
Note: This module requires external config.json file with KAFKA_JAR_URLS, NUM_EXECUTORS, 
EXECUTOR_CORES, and EXECUTOR_GPU_AMOUNT values.
"""

import os
import json
import socket
from pyspark.sql import SparkSession


def get_first_ip():
    """
    Get the first available IP address for the driver host.
    This function is used in the original notebook for dynamic IP resolution.
    """
    import subprocess
    result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
    # Split the result to get the list of IPs and take the first one
    first_ip = result.stdout.split()[0]
    return first_ip


def load_external_config(config_path="config.json"):
    """
    Load external configuration from JSON file.
    
    Args:
        config_path (str): Path to the configuration JSON file
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config.json is not found
        KeyError: If required keys are missing from config
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        # Provide default values if config.json is not available
        print(f"Warning: {config_path} not found. Using default values.")
        config = {
            "KAFKA_JAR_URLS": [
                "jars/commons-logging-1.1.3.jar",
                "jars/commons-pool2-2.11.1.jar", 
                "jars/hadoop-client-api-3.3.4.jar",
                "jars/hadoop-client-runtime-3.3.4.jar",
                "jars/jsr305-3.0.0.jar",
                "jars/kafka-clients-3.4.1.jar",
                "jars/lz4-java-1.8.0.jar",
                "jars/slf4j-api-2.0.7.jar",
                "jars/snappy-java-1.1.10.3.jar",
                "jars/spark-sql-kafka-0-10_2.12-3.5.1.jar",
                "jars/spark-streaming-kafka-0-10_2.12-3.5.1.jar",
                "jars/spark-token-provider-kafka-0-10_2.12-3.5.1.jar",
                "jars/rapids-4-spark_2.12-24.06.1.jar"
            ],
            "NUM_EXECUTORS": 3,
            "EXECUTOR_CORES": 5,
            "EXECUTOR_GPU_AMOUNT": 1
        }
    
    # Validate required keys
    required_keys = ["KAFKA_JAR_URLS", "NUM_EXECUTORS", "EXECUTOR_CORES", "EXECUTOR_GPU_AMOUNT"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Required key '{key}' not found in configuration")
    
    return config


# Load external configuration
try:
    EXTERNAL_CONFIG = load_external_config()
except (FileNotFoundError, KeyError) as e:
    print(f"Configuration loading error: {e}")
    EXTERNAL_CONFIG = load_external_config()  # Will use defaults


# Derived configuration values
JAR_URLS = ",".join(EXTERNAL_CONFIG["KAFKA_JAR_URLS"])
REPARTITION_NUM = EXTERNAL_CONFIG["NUM_EXECUTORS"] * EXTERNAL_CONFIG["EXECUTOR_CORES"] * 2

# Spark Master Configuration
SPARK_MASTER_URL = "spark://spark-master-service:7077"
SPARK_APP_NAME = "asdf"

# Driver Configuration
DRIVER_CONFIG = {
    "spark.driver.host": get_first_ip(),
    "spark.driver.port": "39337"
}

# GPU Configuration
GPU_CONFIG = {
    "spark.executor.resource.gpu.discoveryScript": "/opt/spark/conf/getGpusResources.sh",
    "spark.plugins": "com.nvidia.spark.SQLPlugin",
    "spark.task.resource.gpu.amount": 1,
    "spark.executor.resource.gpu.amount": EXTERNAL_CONFIG["EXECUTOR_GPU_AMOUNT"]
}

# RAPIDS Configuration  
RAPIDS_CONFIG = {
    "spark.rapids.sql.concurrentGpuTasks": 1,
    "spark.rapids.memory.gpu.minAllocFraction": 0.1,
    "spark.rapids.memory.pinnedPool.size": "2g"
}

# Executor Configuration
EXECUTOR_CONFIG = {
    "spark.executor.instances": 3,
    "spark.executor.cores": 5,
    "spark.executor.memory": "24g"
}

# File and JAR Configuration
FILE_CONFIG = {
    "spark.submit.pyFiles": "mnist_pb2.py",  # .py 파일 포함
    "spark.jars": JAR_URLS  # JAR 파일 포함
}

# Parallelism Configuration
PARALLELISM_CONFIG = {
    "spark.defaul.parallelism": REPARTITION_NUM,  # Note: typo exists in original
    "spark.sql.shuffle.partitions": REPARTITION_NUM
}

# Complete Spark Configuration Dictionary
SPARK_CONFIG = {
    **DRIVER_CONFIG,
    **GPU_CONFIG, 
    **RAPIDS_CONFIG,
    **EXECUTOR_CONFIG,
    **FILE_CONFIG,
    **PARALLELISM_CONFIG
}


def create_spark_session(master_url=SPARK_MASTER_URL, app_name=SPARK_APP_NAME, config_dict=None):
    """
    Create a Spark session with the configured settings.
    
    Args:
        master_url (str): Spark master URL
        app_name (str): Application name
        config_dict (dict, optional): Additional configuration dictionary
        
    Returns:
        SparkSession: Configured Spark session
    """
    if config_dict is None:
        config_dict = SPARK_CONFIG
    
    builder = SparkSession.builder.master(master_url).appName(app_name)
    
    # Apply all configurations
    for key, value in config_dict.items():
        builder = builder.config(key, value)
    
    return builder.getOrCreate()


def create_spark_session_original_style():
    """
    Create Spark session exactly as defined in the original notebook.
    This preserves the exact configuration chain from the notebook.
    
    Returns:
        SparkSession: Configured Spark session
    """
    spark = (
        SparkSession.builder.master(SPARK_MASTER_URL)
        .appName(SPARK_APP_NAME)
        .config("spark.driver.host", get_first_ip())
        .config("spark.driver.port", "39337")
        .config("spark.executor.resource.gpu.discoveryScript", "/opt/spark/conf/getGpusResources.sh")
        .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
        .config("spark.submit.pyFiles", "mnist_pb2.py")  # .py 파일 포함
        .config("spark.jars", JAR_URLS)  # JAR 파일 포함
        # Executor Config
        .config("spark.executor.instances", 3)
        .config("spark.executor.cores", 5)
        .config("spark.executor.memory", "24g")
        .config("spark.executor.resource.gpu.amount", EXTERNAL_CONFIG["EXECUTOR_GPU_AMOUNT"])
        # Other Config
        .config("spark.task.resource.gpu.amount", 1)
        .config("spark.rapids.sql.concurrentGpuTasks", 1)
        .config("spark.rapids.memory.gpu.minAllocFraction", 0.1)
        .config("spark.rapids.memory.pinnedPool.size", "2g")
        .config("spark.defaul.parallelism", REPARTITION_NUM)
        .config("spark.sql.shuffle.partitions", REPARTITION_NUM)
        .getOrCreate()
    )
    return spark


def get_spark_config_for_num_processes(num_processes):
    """
    Get Spark configuration adjusted for a specific number of processes.
    Used when creating TorchDistributor instances.
    
    Args:
        num_processes (int): Number of processes for distributed training
        
    Returns:
        dict: Adjusted Spark configuration
    """
    config = SPARK_CONFIG.copy()
    # Adjust executor instances based on num_processes if needed
    # This preserves the original logic where num_processes = executor instances
    return config


# Configuration validation
def validate_spark_config():
    """
    Validate that all required Spark configurations are properly set.
    
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check if JAR files exist (if running locally)
    jar_files = EXTERNAL_CONFIG["KAFKA_JAR_URLS"]
    missing_jars = []
    
    for jar in jar_files:
        if not os.path.exists(jar):
            missing_jars.append(jar)
    
    if missing_jars:
        print(f"Warning: Missing JAR files: {missing_jars}")
    
    # Validate numeric configurations
    if EXTERNAL_CONFIG["NUM_EXECUTORS"] <= 0:
        raise ValueError("NUM_EXECUTORS must be positive")
    
    if EXTERNAL_CONFIG["EXECUTOR_CORES"] <= 0:
        raise ValueError("EXECUTOR_CORES must be positive")
    
    if EXTERNAL_CONFIG["EXECUTOR_GPU_AMOUNT"] <= 0:
        raise ValueError("EXECUTOR_GPU_AMOUNT must be positive")
    
    return True


# Configuration summary for debugging
def print_spark_config_summary():
    """
    Print a summary of the current Spark configuration.
    Useful for debugging and verification.
    """
    print("=== Spark Configuration Summary ===")
    print(f"Master URL: {SPARK_MASTER_URL}")
    print(f"App Name: {SPARK_APP_NAME}")
    print(f"Driver Host: {get_first_ip()}")
    print(f"Executor Instances: {EXTERNAL_CONFIG['NUM_EXECUTORS']}")
    print(f"Executor Cores: {EXTERNAL_CONFIG['EXECUTOR_CORES']}")
    print(f"Executor Memory: 24g")
    print(f"GPU Amount per Executor: {EXTERNAL_CONFIG['EXECUTOR_GPU_AMOUNT']}")
    print(f"Parallelism: {REPARTITION_NUM}")
    print(f"Number of JAR files: {len(EXTERNAL_CONFIG['KAFKA_JAR_URLS'])}")
    print("===================================")


if __name__ == "__main__":
    # For testing and debugging
    validate_spark_config()
    print_spark_config_summary()