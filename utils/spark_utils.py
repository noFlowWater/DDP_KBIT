"""
Spark session utilities and environment helper functions for distributed deep learning.

This module provides utilities for creating and configuring Spark sessions,
environment setup, and Kafka integration for distributed training.
"""

import os
import json
import logging
import subprocess
from typing import Dict, Any, Optional, List
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.conf import SparkConf


def get_first_ip() -> str:
    """
    Get the first IP address from hostname -I command.
    
    Returns:
        str: The first IP address from the system.
    
    Raises:
        RuntimeError: If unable to get IP address.
    """
    try:
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        # Split the result to get the list of IPs and take the first one
        first_ip = result.stdout.split()[0]
        return first_ip
    except Exception as e:
        raise RuntimeError(f"Failed to get first IP address: {e}")


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path (str): Path to the configuration file. Defaults to "config.json".
    
    Returns:
        Dict[str, Any]: Configuration dictionary.
    
    Raises:
        FileNotFoundError: If config file is not found.
        json.JSONDecodeError: If config file is not valid JSON.
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in config file: {e}")


def create_spark_session(
    app_name: str = "SparkDL_KBIT",
    master_url: str = "spark://spark-master-service:7077",
    config_path: str = "config.json",
    driver_port: str = "39337",
    custom_configs: Optional[Dict[str, Any]] = None
) -> SparkSession:
    """
    Create and configure a Spark session for distributed deep learning with GPU support.
    
    Args:
        app_name (str): Name of the Spark application. Defaults to "SparkDL_KBIT".
        master_url (str): Spark master URL. Defaults to "spark://spark-master-service:7077".
        config_path (str): Path to configuration file. Defaults to "config.json".
        driver_port (str): Driver port for Spark. Defaults to "39337".
        custom_configs (Optional[Dict[str, Any]]): Additional custom configurations.
    
    Returns:
        SparkSession: Configured Spark session.
    
    Raises:
        Exception: If unable to create Spark session.
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Extract configuration values
        jar_urls = ",".join(config["KAFKA_JAR_URLS"])
        repartition_num = config["NUM_EXECUTORS"] * config["EXECUTOR_CORES"] * 2
        
        # Get driver host IP
        driver_host = get_first_ip()
        
        # Create Spark session builder
        builder = (
            SparkSession.builder
            .master(master_url)
            .appName(app_name)
            .config("spark.driver.host", driver_host)
            .config("spark.driver.port", driver_port)
            .config("spark.executor.resource.gpu.discoveryScript", "/opt/spark/conf/getGpusResources.sh")
            .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
            .config("spark.submit.pyFiles", "mnist_pb2.py")
            .config("spark.jars", jar_urls)
            # Executor Configuration
            .config("spark.executor.instances", config.get("NUM_EXECUTORS", 3))
            .config("spark.executor.cores", config.get("EXECUTOR_CORES", 5))
            .config("spark.executor.memory", config.get("EXECUTOR_MEMORY", "24g"))
            .config("spark.executor.resource.gpu.amount", config.get("EXECUTOR_GPU_AMOUNT", 1))
            # GPU and Task Configuration
            .config("spark.task.resource.gpu.amount", 1)
            .config("spark.rapids.sql.concurrentGpuTasks", 1)
            .config("spark.rapids.memory.gpu.minAllocFraction", 0.1)
            .config("spark.rapids.memory.pinnedPool.size", "2g")
            # Parallelism Configuration
            .config("spark.default.parallelism", repartition_num)
            .config("spark.sql.shuffle.partitions", repartition_num)
        )
        
        # Apply custom configurations if provided
        if custom_configs:
            for key, value in custom_configs.items():
                builder = builder.config(key, value)
        
        # Create the session
        spark = builder.getOrCreate()
        
        
        # Configure logging
        configure_spark_logging(spark)
        
        return spark
        
    except Exception as e:
        raise Exception(f"Failed to create Spark session: {e}")


def configure_spark_logging(spark: SparkSession) -> None:
    """
    Configure Spark logging to reduce verbosity.
    
    Args:
        spark (SparkSession): Spark session to configure logging for.
    """
    # Reduce py4j logging
    logging.getLogger("py4j").setLevel(logging.ERROR)
    
    # Reduce pyspark logging
    pyspark_log = logging.getLogger('pyspark')
    pyspark_log.setLevel(logging.ERROR)
    
    # Set Spark context log level
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    
    print("Current Spark configuration:")
    for key, value in sorted(sc._conf.getAll(), key=lambda x: x[0]):
        print(f"{key} = {value}")
    


def check_kafka_partitions(broker_address: List[str], topic: str) -> bool:
    """
    Check if the number of partitions in a Kafka topic matches the WORLD_SIZE.
    
    Args:
        broker_address (List[str]): List of Kafka broker addresses.
        topic (str): Name of the Kafka topic.
    
    Returns:
        bool: True if the number of partitions matches WORLD_SIZE, False otherwise.
    
    Raises:
        ImportError: If kafka-python is not installed.
    """
    try:
        from kafka import KafkaAdminClient
        from kafka.errors import KafkaError
    except ImportError:
        raise ImportError("kafka-python is required for Kafka partition checking")
    
    try:
        # Connect to the Kafka broker
        admin_client = KafkaAdminClient(bootstrap_servers=broker_address)
        
        # Retrieve metadata for the specified topic
        topic_metadata = admin_client.describe_topics([topic])
        topic_info = topic_metadata[0]
        
        # Get the number of partitions for the topic
        num_partitions = len(topic_info['partitions'])
        
        # Retrieve WORLD_SIZE from environment variables
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        print(f"Topic '{topic}' has {num_partitions} partitions. WORLD_SIZE is {world_size}.")
        
        # Compare the number of partitions with WORLD_SIZE
        if num_partitions == world_size:
            print("The number of partitions matches the WORLD_SIZE.")
            return True
        else:
            print("The number of partitions does not match the WORLD_SIZE.")
            return False
            
    except KafkaError as e:
        print(f"Kafka error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def load_from_kafka(
    spark: SparkSession,
    kafka_config: Dict[str, Any],
    offset_ranges: Dict[int, Dict[str, int]],
    schema: Any
) -> Any:
    """
    Load data from Kafka for specific offset ranges and return as Spark DataFrame.
    
    Args:
        spark (SparkSession): Spark session object.
        kafka_config (Dict[str, Any]): Kafka configuration dictionary.
        offset_ranges (Dict[int, Dict[str, int]]): Offset ranges returned by save_to_kafka function.
        schema (Any): Schema for deserializing data (StructType).
    
    Returns:
        Any: Spark DataFrame containing the deserialized data.
    
    Raises:
        Exception: If unable to load data from Kafka.
    """
    try:
        from pyspark.sql.functions import from_json, col
        
        # Build starting and ending offsets dictionaries
        starting_offsets = {kafka_config['topic']: {
            str(p): r['start'] for p, r in offset_ranges.items()
        }}
        ending_offsets = {kafka_config['topic']: {
            str(p): r['end'] for p, r in offset_ranges.items()
        }}
        
        # Convert to JSON strings
        starting_offsets_json = json.dumps(starting_offsets)
        ending_offsets_json = json.dumps(ending_offsets)
        
        # Load raw data from Kafka
        df_raw = (
            spark.read
            .format("kafka")
            .option("kafka.bootstrap.servers", ",".join(kafka_config['bootstrap_servers']))
            .option("subscribe", kafka_config['topic'])
            .option("startingOffsets", starting_offsets_json)
            .option("endingOffsets", ending_offsets_json)
            .load()
        )
        
        # Parse the value column and deserialize using the provided schema
        df_parsed = df_raw.select(
            from_json(col("value").cast("string"), schema).alias("parsed_value")
        ).select("parsed_value.*")
        
        return df_parsed
        
    except Exception as e:
        raise Exception(f"Failed to load data from Kafka: {e}")


def setup_working_directory(work_dir: str = "/mnt/data") -> None:
    """
    Set up the working directory and print directory contents.
    
    Args:
        work_dir (str): Working directory path. Defaults to "/mnt/data".
    
    Raises:
        OSError: If unable to change directory.
    """
    try:
        os.chdir(work_dir)
        print("FILES IN THIS DIRECTORY")
        print(os.listdir(os.getcwd()))
    except OSError as e:
        raise OSError(f"Failed to change to working directory {work_dir}: {e}")


def initialize_distributed_training(use_gpu: bool = True) -> Dict[str, Any]:
    """
    Initialize the distributed training environment and set up the appropriate device.
    
    Args:
        use_gpu (bool): Whether to use GPU if available. Defaults to True.
    
    Returns:
        Dict[str, Any]: Dictionary containing distributed training information:
            - 'backend': Backend used for distributed training ('nccl' or 'gloo')
            - 'global_rank': Global rank of the current process
            - 'local_rank': Local rank of the current process
            - 'world_size': Total number of processes in distributed training
            - 'device': torch.device object to be used for training
            - 'device_ids': List of device IDs for GPU training or None for CPU
            - 'env_dict': Dictionary of environment variables for distributed training
    
    Raises:
        ImportError: If torch or torch.distributed is not available.
        RuntimeError: If distributed training initialization fails.
    """
    try:
        import torch
        import torch.distributed as dist
        import datetime
    except ImportError:
        raise ImportError("PyTorch is required for distributed training initialization")
    
    try:
        # Set the appropriate backend based on whether GPU is used
        backend = "nccl" if use_gpu and torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, timeout=datetime.timedelta(seconds=5400))
        
        # Gather environment variables needed for distributed training
        env_vars = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]
        env_dict = {key: os.getenv(key) for key in env_vars}
        
        # Validate required environment variables
        for key in env_vars:
            if env_dict[key] is None:
                raise ValueError(f"Required environment variable {key} is not set")
        
        global_rank = int(env_dict["RANK"])
        local_rank = int(env_dict["LOCAL_RANK"])
        world_size = int(env_dict["WORLD_SIZE"])
        
        # Determine the device to use based on local rank
        if use_gpu and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
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
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize distributed training: {e}")