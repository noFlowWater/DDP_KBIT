"""
Dataset classes for distributed data processing and MNIST data handling.

This module contains dataset classes extracted from the main notebook:
- DistributedDataset: Handles distributed Kafka data consumption
- ProcessedMNISTDataset: Custom dataset for pre-processed MNIST data
- create_distributed_dataloader: Utility for creating distributed data loaders
"""

import os
import json
import pickle
import random
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import tqdm
from kafka import KafkaConsumer, TopicPartition
from kafka.errors import KafkaError
import fastavro
from io import BytesIO
import base64
import numpy as np


class DistributedDataset(Dataset):
    """
    Distributed dataset that consumes data from Kafka topics.
    
    Supports multiple message formats:
    - protobuf: Protocol Buffer messages
    - avro: Apache Avro binary format
    - mongodb_avro: MongoDB Binary data containing Avro
    - none: Plain JSON messages
    """
    
    def __init__(self, topic, partition_ranges, consumer_params, payload_config, is_last_split, last_fillup):
        """
        Initialize the distributed dataset.
        
        Args:
            topic (str): Kafka topic name
            partition_ranges (list): List of (partition_id, start_offset, end_offset) tuples
            consumer_params (dict): Kafka consumer configuration parameters
            payload_config (dict): Message format and transformation configuration
            is_last_split (bool): Whether this is the last data split
            last_fillup (bool): Whether to fill up data to match batch sizes
        """
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.topic = topic
        self.partition_ranges = partition_ranges
        self.consumer_params = consumer_params
        self.payload_config = payload_config
        self.is_last_split = is_last_split
        self.last_fillup = last_fillup

        # Configure payload processing based on config
        if payload_config is None:
            self.message_format = "none"
            self.transform_data_fn = None
            self.transform_label_fn = None
            self.protobuf_msg_class = None
            self.avro_schema = None
            self.data_field = "data"
            self.label_field = "label"
        else:
            self.message_format = payload_config.get("message_format", "protobuf")
            self.transform_data_fn = payload_config.get("transform_data_fn")
            self.transform_label_fn = payload_config.get("transform_label_fn")
            self.protobuf_msg_class = payload_config.get("protobuf_msg_class")
            self.avro_schema = payload_config.get("avro_schema")
            self.data_field = payload_config.get("data_field", "data")
            self.label_field = payload_config.get("label_field", "label")
            self.schema_fingerprint_field = payload_config.get("schema_fingerprint_field", "schema_fingerprint")

        # Load data (transformations will be applied in _process_message)
        self.data = self._load_data()

    def _load_data(self):
        """Load data from Kafka partitions assigned to this rank."""
        data = []
        max_offset_diff = 0

        for partition_id, start_offset, end_offset in self.partition_ranges:
            if int(partition_id) == self.rank:
                partition_data = self._pull_data_from_kafka(int(partition_id), int(start_offset), int(end_offset))
                data.extend(partition_data)
                
            offset_diff = end_offset - start_offset
            if offset_diff > max_offset_diff:
                max_offset_diff = offset_diff

        # Fill up data if needed for the last split
        if self.is_last_split and self.last_fillup and len(data) < max_offset_diff + 1:
            data.extend(random.choices(data, k=max_offset_diff + 1 - len(data)))
            print(f"Rank {self.rank}: Data size after sampling: {len(data)}")

        return data

    def _pull_data_from_kafka(self, partition_id, start_offset, end_offset, timeout_ms=3000):
        """
        Pull data from a specific Kafka partition within the given offset range.
        
        Args:
            partition_id (int): Kafka partition ID
            start_offset (int): Starting offset
            end_offset (int): Ending offset
            timeout_ms (int): Polling timeout in milliseconds
            
        Returns:
            list: List of (data, label) tuples
        """
        consumer = self._create_kafka_consumer()

        consumer.assign([TopicPartition(self.topic, partition_id)])
        consumer.seek(TopicPartition(self.topic, partition_id), start_offset)

        result = []
        polling_done = False
        pbar = tqdm.tqdm(total=end_offset - start_offset + 1, desc=f'Rank {self.rank} Polling', unit='msg')
        
        try:
            print(f"Rank {self.rank} : 메세지 Polling Start {start_offset} ~ {end_offset}")
            while not polling_done:
                records = consumer.poll(timeout_ms)
                if not records:
                    print(f"Rank {self.rank} : {start_offset} ~ {end_offset} polling Timeout... try again")
                    break

                for _, messages in records.items():
                    for msg in messages:
                        try:
                            # Process data and label
                            result.append(self._process_message(msg))
                            pbar.update(1)

                        except Exception as e:
                            print(f"디코딩 또는 역직렬화 중 오류 발생: {e}")
                            polling_done = True
                            break

                        if msg.offset >= end_offset:
                            polling_done = True
                            break

                    if polling_done:
                        break
        finally:
            consumer.close()
            pbar.close()

        return result

    def _create_kafka_consumer(self):
        """Create a Kafka consumer with the configured parameters."""
        # Validate bootstrap_servers
        if "bootstrap_servers" not in self.consumer_params or self.consumer_params["bootstrap_servers"] is None:
            raise KeyError("bootstrap_servers is missing or null in the KafkaConsumer's additional_options.")
        
        # Set default options
        _options = {
            'client_id': f'{self.topic}-consumer-rank-{self.rank}',
            'group_id': f'{self.topic}-consumers',
            'auto_offset_reset': 'earliest',
            'enable_auto_commit': False,
        }

        # Merge additional options
        _options.update(self.consumer_params)

        # Create KafkaConsumer object
        _consumer = KafkaConsumer(**_options)
        
        return _consumer

    def _process_message(self, msg):
        """
        Process messages in various formats.
        Supported formats: protobuf, avro, mongodb_avro, none(JSON)
        """
        if self.message_format == "protobuf":
            return self._process_protobuf_message(msg)
        elif self.message_format == "avro":
            return self._process_avro_message(msg)
        elif self.message_format == "mongodb_avro":
            return self._process_mongodb_avro_message(msg)
        elif self.message_format == "none":
            return self._process_json_message(msg)
        else:
            raise ValueError(f"지원하지 않는 메시지 형식: {self.message_format}")
        
    def _process_avro_message(self, msg):
        """Process Avro messages."""
        # Direct Avro deserialization
        avro_data = fastavro.schemaless_reader(BytesIO(msg.value), self.avro_schema)
        
        # Extract data
        data = avro_data[self.data_field]
        label = avro_data[self.label_field]
        
        # Apply transforms
        if self.transform_data_fn:
            data = self.transform_data_fn(data)
        if self.transform_label_fn:
            label = self.transform_label_fn(label)
        
        return data, label
    
    def _process_protobuf_message(self, msg):
        """Process protobuf messages."""
        if self.protobuf_msg_class is None:
            raise ValueError("protobuf_msg_class가 설정되지 않았습니다.")
        # Implementation would depend on specific protobuf message class
        # This is a placeholder for protobuf processing logic
        pass
    
    def _process_json_message(self, msg):
        """Process JSON messages."""
        # JSON message deserialization
        json_data = json.loads(msg.value.decode('utf-8'))
        
        # Extract data and label using configured field names
        data = json_data.get(self.data_field)
        label = json_data.get(self.label_field)
        
        # Apply transformation functions
        transformed_data = self.transform_data_fn(data) if self.transform_data_fn else data
        transformed_label = self.transform_label_fn(label) if self.transform_label_fn else label
        
        return transformed_data, transformed_label
    
    def _process_mongodb_avro_message(self, msg):
        """Process MongoDB Avro messages."""
        # JSON parsing → MongoDB Binary extraction → Base64 decoding → Avro deserialization
        json_data = json.loads(msg.value.decode('utf-8'))
        avro_binary_data = json_data[self.data_field]  # "avro_data"
        binary_data = base64.b64decode(avro_binary_data['Data'])
        avro_data = fastavro.schemaless_reader(BytesIO(binary_data), self.avro_schema)
        
        # Extract data
        data = avro_data["data"]
        label = avro_data["label"]
        
        # Apply transforms
        if self.transform_data_fn:
            data = self.transform_data_fn(data)
        if self.transform_label_fn:
            label = self.transform_label_fn(label)
        
        return data, label
        
    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """Get an item from the dataset."""
        return self.data[index]


class ProcessedMNISTDataset(torch.utils.data.Dataset):
    """Custom dataset for loading pre-processed MNIST data."""
    
    def __init__(self, data_path, split='train'):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Path where data is stored
            split (str): One of 'train', 'test', 'val'
        """
        self.data_path = data_path
        self.split = split
        
        if split not in ['train', 'test', 'val']:
            raise ValueError(f"split must be one of ['train', 'test', 'val'], got {split}")
        
        # Load metadata
        metadata_path = os.path.join(data_path, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load data
        data_file = os.path.join(data_path, f"{split}_data.pt")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        data_dict = torch.load(data_file)
        self.data = data_dict['data']
        self.labels = data_dict['labels']
        
        print(f"{split.capitalize()} 데이터 로드 완료:")
        print(f"  데이터 크기: {self.data.shape}")
        print(f"  라벨 크기: {self.labels.shape}")
    
    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get an item from the dataset."""
        return self.data[idx], self.labels[idx]


def create_distributed_dataloader(data_path, rank, world_size, batch_size=64, use_validation=True):
    """
    Create distributed data loaders for training.
    
    Args:
        data_path (str): Path where data is stored
        rank (int): Current process rank
        world_size (int): Total number of processes
        batch_size (int): Batch size
        use_validation (bool): True to use validation dataset, False to use test dataset
        
    Returns:
        tuple: (train_loader, eval_loader, test_loader)
    """
    
    print(f"Rank {rank}: 데이터 로더 생성 중...")
    
    # Create custom datasets
    train_dataset = ProcessedMNISTDataset(data_path, split='train')
    test_dataset = ProcessedMNISTDataset(data_path, split='test')
    val_dataset = ProcessedMNISTDataset(data_path, split='val')
    
    # Select evaluation dataset
    eval_dataset = val_dataset if use_validation else test_dataset
    eval_split_name = 'validation' if use_validation else 'test'
    
    # Create DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,  # Total number of processes
        rank=rank,                # Current process rank
        shuffle=True,             # Shuffle each epoch
        seed=42,                  # Seed for reproducibility
        drop_last=True            # Drop last batch
    )
    
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,            # No shuffle for evaluation
        seed=42,
        drop_last=False
    )
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=42,
        drop_last=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,     # Use DistributedSampler
        num_workers=2,             # Number of workers
        pin_memory=True,           # GPU transfer optimization
        persistent_workers=True    # Reuse workers
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=eval_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Print data distribution info for rank 0
    if rank == 0:
        print(f"\n데이터 분할 정보:")
        print(f"  전체 프로세스 수: {world_size}")
        print(f"  Train 데이터 총 크기: {len(train_dataset)}")
        print(f"  {eval_split_name.capitalize()} 데이터 총 크기: {len(eval_dataset)}")
        print(f"  Test 데이터 총 크기: {len(test_dataset)}")
        print(f"  각 프로세스별 예상 Train 배치 수: {len(train_loader)}")
        print(f"  각 프로세스별 예상 {eval_split_name.capitalize()} 배치 수: {len(eval_loader)}")
    
    return train_loader, eval_loader, test_loader