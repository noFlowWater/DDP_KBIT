"""
Model checkpoint management utilities for distributed deep learning.

This module provides utilities for saving and loading model checkpoints,
both from local filesystem and Kafka message queues.
"""

import os
import json
import pickle
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path


logger = logging.getLogger(__name__)


def load_checkpoint(
    unique_key: Optional[str] = None,
    load_directory: str = "/mnt/data/spark_DL_checkpoints",
    kafka_config: Optional[Dict[str, Any]] = None,
    timeout_ms: int = 10000
) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint from either Kafka or local filesystem based on configuration.
    
    Args:
        unique_key (Optional[str]): Unique key (UUID) for the checkpoint to load.
        load_directory (str): Directory path for local checkpoint files.
                            Defaults to "/mnt/data/spark_DL_checkpoints".
        kafka_config (Optional[Dict[str, Any]]): Kafka configuration dictionary.
                                               If None, loads from filesystem.
        timeout_ms (int): Kafka poll timeout in milliseconds. Defaults to 10000ms.
    
    Returns:
        Optional[Dict[str, Any]]: Loaded checkpoint state dictionary, or None if not found.
        
    Raises:
        ImportError: If required dependencies are not available.
        FileNotFoundError: If checkpoint file is not found in filesystem mode.
        Exception: For other loading errors.
    """
    if kafka_config is None:
        return _load_checkpoint_from_filesystem(unique_key, load_directory)
    else:
        return _load_checkpoint_from_kafka(unique_key, kafka_config, timeout_ms)


def _load_checkpoint_from_filesystem(
    unique_key: Optional[str],
    load_directory: str
) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint from local filesystem.
    
    Args:
        unique_key (Optional[str]): Unique key for the checkpoint.
        load_directory (str): Directory containing checkpoint files.
    
    Returns:
        Optional[Dict[str, Any]]: Loaded checkpoint state, or None if not found.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for checkpoint loading")
    
    try:
        # Ensure directory exists
        if not os.path.exists(load_directory):
            logger.warning(f"Checkpoint directory {load_directory} does not exist.")
            return None
        
        # Get all checkpoint files in the directory
        all_files = os.listdir(load_directory)
        checkpoint_files = [f for f in all_files if f.endswith('.pt')]
        
        if not checkpoint_files:
            logger.info("No checkpoint files found in the directory.")
            return None
        
        # Find the most recent checkpoint file based on unique_key (timestamp)
        if unique_key:
            # Look for specific checkpoint with unique_key
            matching_files = [f for f in checkpoint_files if unique_key in f]
            if matching_files:
                latest_file = max(matching_files, 
                                key=lambda x: '_'.join(x.split('_')[-2:]).split('.')[0])
            else:
                logger.warning(f"No checkpoint found with unique_key: {unique_key}")
                return None
        else:
            # Get the most recent checkpoint file based on filename timestamp
            latest_file = max(checkpoint_files, 
                            key=lambda x: '_'.join(x.split('_')[-2:]).split('.')[0])
        
        # Full path to the checkpoint file
        load_path = os.path.join(load_directory, latest_file)
        
        # Load the checkpoint using torch.load()
        loaded_state = torch.load(load_path, map_location='cpu')
        
        epoch = loaded_state.get('epoch', 'unknown')
        
        # Log successful loading
        logger.info(f"Checkpoint loaded from {load_path}, starting at epoch {epoch}")
        print(f"Checkpoint loaded from {load_path}, starting at epoch {epoch}")
        
        return loaded_state
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint from filesystem: {e}")
        raise Exception(f"Failed to load checkpoint from filesystem: {e}")


def _load_checkpoint_from_kafka(
    unique_key: str,
    kafka_config: Dict[str, Any],
    timeout_ms: int
) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint from Kafka message queue.
    
    Args:
        unique_key (str): Unique key for the checkpoint to load.
        kafka_config (Dict[str, Any]): Kafka configuration dictionary.
        timeout_ms (int): Kafka poll timeout in milliseconds.
    
    Returns:
        Optional[Dict[str, Any]]: Loaded checkpoint state, or None if not found.
    """
    try:
        from kafka import KafkaConsumer
        from kafka.errors import KafkaError
    except ImportError:
        raise ImportError("kafka-python is required for Kafka checkpoint loading")
    
    if not unique_key:
        raise ValueError("unique_key is required for Kafka checkpoint loading")
    
    try:
        # Set default Kafka configuration
        kafka_config = kafka_config.copy()
        kafka_config.setdefault('bootstrap_servers', [
            '155.230.34.51:32100',
            '155.230.34.52:32100', 
            '155.230.34.53:32100'
        ])
        kafka_config.setdefault('model_save_topic', 'model-checkpoints')
        
        # Create Kafka Consumer
        consumer = KafkaConsumer(
            kafka_config['model_save_topic'],
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_deserializer=lambda x: pickle.loads(x),
            key_deserializer=lambda x: x.decode('utf-8'),
            auto_offset_reset='earliest',
            enable_auto_commit=False
        )
        
        checkpoint_state = None
        
        try:
            while True:
                records = consumer.poll(timeout_ms)
                if not records:
                    logger.info(f"No checkpoint found for unique key {unique_key} within the timeout period.")
                    break
                
                for tp, messages in records.items():
                    for message in messages:
                        if message.key == unique_key:
                            checkpoint_state = message.value
                            logger.info(f"Checkpoint found for unique key {unique_key}")
                            break
                    if checkpoint_state is not None:
                        break
                if checkpoint_state is not None:
                    break
                    
        finally:
            consumer.close()
        
        if checkpoint_state is None:
            logger.info(f"No checkpoint found for unique key {unique_key}")
            return None
        
        logger.info(f"Checkpoint for unique key {unique_key} loaded successfully from Kafka topic '{kafka_config['model_save_topic']}'")
        print(f"Checkpoint for unique key {unique_key} loaded successfully from Kafka topic '{kafka_config['model_save_topic']}'")
        
        return checkpoint_state
        
    except KafkaError as e:
        logger.error(f"Kafka error while loading checkpoint: {e}")
        raise Exception(f"Kafka error while loading checkpoint: {e}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint from Kafka: {e}")
        raise Exception(f"Failed to load checkpoint from Kafka: {e}")


def save_checkpoint(
    state_dict: Dict[str, Any],
    unique_key: str,
    save_directory: str = "/mnt/data/spark_DL_checkpoints",
    kafka_config: Optional[Dict[str, Any]] = None,
    epoch: Optional[int] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Save checkpoint to either Kafka or local filesystem.
    
    Args:
        state_dict (Dict[str, Any]): Model state dictionary to save.
        unique_key (str): Unique identifier for the checkpoint.
        save_directory (str): Directory for saving checkpoint files.
                            Defaults to "/mnt/data/spark_DL_checkpoints".
        kafka_config (Optional[Dict[str, Any]]): Kafka configuration. 
                                               If None, saves to filesystem.
        epoch (Optional[int]): Current training epoch.
        additional_info (Optional[Dict[str, Any]]): Additional information to save.
    
    Returns:
        bool: True if save was successful, False otherwise.
        
    Raises:
        Exception: If saving fails.
    """
    # Prepare checkpoint data
    checkpoint_data = {
        'model': state_dict,
        'unique_key': unique_key,
        'epoch': epoch
    }
    
    if additional_info:
        checkpoint_data.update(additional_info)
    
    if kafka_config is None:
        return _save_checkpoint_to_filesystem(checkpoint_data, unique_key, save_directory)
    else:
        return _save_checkpoint_to_kafka(checkpoint_data, unique_key, kafka_config)


def _save_checkpoint_to_filesystem(
    checkpoint_data: Dict[str, Any],
    unique_key: str,
    save_directory: str
) -> bool:
    """
    Save checkpoint to local filesystem.
    
    Args:
        checkpoint_data (Dict[str, Any]): Checkpoint data to save.
        unique_key (str): Unique identifier for the checkpoint.
        save_directory (str): Directory for saving checkpoint files.
    
    Returns:
        bool: True if save was successful, False otherwise.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for checkpoint saving")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Generate filename with unique key
        filename = f"checkpoint_{unique_key}.pt"
        save_path = os.path.join(save_directory, filename)
        
        # Save the checkpoint
        torch.save(checkpoint_data, save_path)
        
        logger.info(f"Checkpoint saved to {save_path}")
        print(f"Checkpoint saved to {save_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint to filesystem: {e}")
        return False


def _save_checkpoint_to_kafka(
    checkpoint_data: Dict[str, Any],
    unique_key: str,
    kafka_config: Dict[str, Any]
) -> bool:
    """
    Save checkpoint to Kafka message queue.
    
    Args:
        checkpoint_data (Dict[str, Any]): Checkpoint data to save.
        unique_key (str): Unique identifier for the checkpoint.
        kafka_config (Dict[str, Any]): Kafka configuration dictionary.
    
    Returns:
        bool: True if save was successful, False otherwise.
    """
    try:
        from kafka import KafkaProducer
        from kafka.errors import KafkaError
    except ImportError:
        raise ImportError("kafka-python is required for Kafka checkpoint saving")
    
    try:
        # Set default Kafka configuration
        kafka_config = kafka_config.copy()
        kafka_config.setdefault('bootstrap_servers', [
            '155.230.34.51:32100',
            '155.230.34.52:32100', 
            '155.230.34.53:32100'
        ])
        kafka_config.setdefault('model_save_topic', 'model-checkpoints')
        
        # Create Kafka Producer
        producer = KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda x: pickle.dumps(x),
            key_serializer=lambda x: x.encode('utf-8')
        )
        
        try:
            # Send checkpoint to Kafka
            future = producer.send(
                kafka_config['model_save_topic'],
                key=unique_key,
                value=checkpoint_data
            )
            
            # Wait for message to be sent
            record_metadata = future.get(timeout=10)
            
            logger.info(f"Checkpoint saved to Kafka topic '{kafka_config['model_save_topic']}' "
                       f"with key '{unique_key}' at offset {record_metadata.offset}")
            print(f"Checkpoint saved to Kafka topic '{kafka_config['model_save_topic']}' "
                  f"with key '{unique_key}' at offset {record_metadata.offset}")
            
            return True
            
        finally:
            producer.flush()
            producer.close()
            
    except KafkaError as e:
        logger.error(f"Kafka error while saving checkpoint: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to save checkpoint to Kafka: {e}")
        return False


def list_available_checkpoints(
    load_directory: str = "/mnt/data/spark_DL_checkpoints",
    kafka_config: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    List available checkpoints from filesystem or Kafka.
    
    Args:
        load_directory (str): Directory containing checkpoint files.
        kafka_config (Optional[Dict[str, Any]]): Kafka configuration.
                                               If None, lists filesystem checkpoints.
    
    Returns:
        List[str]: List of available checkpoint identifiers.
    """
    if kafka_config is None:
        return _list_filesystem_checkpoints(load_directory)
    else:
        logger.warning("Listing Kafka checkpoints is not implemented yet")
        return []


def _list_filesystem_checkpoints(load_directory: str) -> List[str]:
    """
    List available checkpoint files in the filesystem.
    
    Args:
        load_directory (str): Directory containing checkpoint files.
    
    Returns:
        List[str]: List of checkpoint filenames.
    """
    try:
        if not os.path.exists(load_directory):
            logger.warning(f"Checkpoint directory {load_directory} does not exist.")
            return []
        
        all_files = os.listdir(load_directory)
        checkpoint_files = [f for f in all_files if f.endswith('.pt')]
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(load_directory, x)),
            reverse=True
        )
        
        return checkpoint_files
        
    except Exception as e:
        logger.error(f"Failed to list checkpoints: {e}")
        return []


def cleanup_old_checkpoints(
    load_directory: str = "/mnt/data/spark_DL_checkpoints",
    keep_count: int = 5
) -> int:
    """
    Remove old checkpoint files, keeping only the most recent ones.
    
    Args:
        load_directory (str): Directory containing checkpoint files.
        keep_count (int): Number of recent checkpoints to keep. Defaults to 5.
    
    Returns:
        int: Number of checkpoint files removed.
    """
    try:
        checkpoint_files = _list_filesystem_checkpoints(load_directory)
        
        if len(checkpoint_files) <= keep_count:
            logger.info(f"Found {len(checkpoint_files)} checkpoints, no cleanup needed")
            return 0
        
        # Remove old checkpoints
        files_to_remove = checkpoint_files[keep_count:]
        removed_count = 0
        
        for filename in files_to_remove:
            try:
                file_path = os.path.join(load_directory, filename)
                os.remove(file_path)
                logger.info(f"Removed old checkpoint: {filename}")
                removed_count += 1
            except Exception as e:
                logger.error(f"Failed to remove checkpoint {filename}: {e}")
        
        logger.info(f"Cleaned up {removed_count} old checkpoint files")
        return removed_count
        
    except Exception as e:
        logger.error(f"Failed to cleanup old checkpoints: {e}")
        return 0


def get_checkpoint_info(
    checkpoint_path: str
) -> Dict[str, Any]:
    """
    Get information about a checkpoint file.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
    
    Returns:
        Dict[str, Any]: Dictionary containing checkpoint information.
        
    Raises:
        FileNotFoundError: If checkpoint file is not found.
        Exception: If unable to load checkpoint information.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for checkpoint information")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    try:
        # Load checkpoint (map to CPU to avoid GPU requirements)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract information
        info = {
            'file_path': checkpoint_path,
            'file_size': os.path.getsize(checkpoint_path),
            'modified_time': os.path.getmtime(checkpoint_path),
            'keys': list(checkpoint.keys()) if isinstance(checkpoint, dict) else ['unknown']
        }
        
        # Add specific checkpoint information if available
        if isinstance(checkpoint, dict):
            info['epoch'] = checkpoint.get('epoch', 'unknown')
            info['unique_key'] = checkpoint.get('unique_key', 'unknown')
            
            # Add model parameter count if model state is available
            if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                param_count = sum(
                    param.numel() for param in checkpoint['model'].values()
                    if hasattr(param, 'numel')
                )
                info['model_parameters'] = param_count
        
        return info
        
    except Exception as e:
        raise Exception(f"Failed to get checkpoint information: {e}")