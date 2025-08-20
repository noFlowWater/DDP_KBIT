"""
Distributed data fetcher for coordinated data loading across multiple processes.

This module contains the DistributedDataFetcher class that handles:
- Coordinated data fetching from external APIs
- Data broadcasting across distributed processes
- Dataset creation and splitting logic
"""

import json
import requests
import torch
import torch.distributed as dist
from DDP_KBIT.data.datasets import DistributedDataset
from DDP_KBIT.data.kafka_utils import split_offsets, check_partitions, create_dynamic_url


class DistributedDataFetcher:
    """
    Distributed data fetcher for coordinating data loading across multiple processes.
    
    This class works after dist.init_process_group() has been called.
    It fetches data, splits it, and creates datasets for distributed training.
    """

    def __init__(self, data_loader_config: dict, device: torch.device):
        """
        Initialize the distributed data fetcher.
        
        Args:
            data_loader_config (dict): Configuration for data loading including API config,
                                     consumer parameters, and split configuration
            device (torch.device): Device to use for tensor operations
        """
        self.data_loader_config = data_loader_config
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.device = device

    def fetch_n_pull_splited_datasets(self) -> list[DistributedDataset]:
        """
        Fetch data, split it, and create distributed datasets.
        
        Returns:
            list: List of DistributedDataset objects, one for each split
        """
        self._validate_config()
        
        # Check if offsets data is provided directly in config
        if "offsets_data" in self.data_loader_config and "offsets_data_topic" in self.data_loader_config:
            offsets_data = self.data_loader_config["offsets_data"]
            send_topic = self.data_loader_config["offsets_data_topic"]
        else:
            send_topic, offsets_data = self._fetch_data()
        
        # Split offset data according to configuration
        # Example offsets_data: ['1:0:5833', '2:0:5833', '3:0:5833', ...]
        # This represents partition_id:start_offset:end_offset format
        offset_ranges = self._split_offsets(offsets_data)
        
        # Create datasets based on the split offset ranges
        datasets = self._create_datasets(offset_ranges, send_topic)
        
        return datasets
    
    def _validate_config(self):
        """Validate the configuration parameters."""
        if self.rank == 0:
            if ("bootstrap_servers" not in self.data_loader_config["consumer_params"] or 
                self.data_loader_config["consumer_params"]["bootstrap_servers"] is None):
                raise KeyError("bootstrap_servers is missing or null in the KafkaConsumer's additional_options.")
            
            if not self._check_partitions():
                raise ValueError("Mismatch between the number of Kafka Topic partitions and WORLD_SIZE.")
            
    def _fetch_data(self):
        """
        Fetch data from external API (only rank 0) and broadcast to all processes.
        
        Returns:
            tuple: (send_topic, offsets_data)
        """
        if self.rank == 0:
            url = self._create_dynamic_url()
            print(f"Rank 0: Sending request to {url}")
            send_topic, offsets_data = self._req_and_validate_res(url)
            print(f"Rank 0: Received data: send_topic={send_topic}, offsets_data={offsets_data}")
        else:
            send_topic, offsets_data = None, None

        # Synchronize all processes
        dist.barrier()
        send_topic = self._broadcast_string(send_topic if self.rank == 0 else "", src=0)
        offsets_data = self._broadcast_json(offsets_data if self.rank == 0 else [], src=0)
        dist.barrier()

        print(f"Rank {self.rank}: Received broadcasted data: send_topic={send_topic}, offsets_data={offsets_data}")
        return send_topic, offsets_data
    
    def _split_offsets(self, offsets_data):
        """Split offsets according to configuration."""
        return split_offsets(offsets_data, self.data_loader_config["dataset_split_config"])
    
    def _create_datasets(self, offset_ranges, send_topic):
        """
        Create DistributedDataset objects for each split.
        
        Args:
            offset_ranges (list): List of offset ranges for each split
            send_topic (str): Kafka topic name
            
        Returns:
            list: List of DistributedDataset objects
        """
        datasets = []
        # Get payload config or use default (None)
        payload_config = self.data_loader_config.get("payload_config", None)

        for i, partition_ranges in enumerate(offset_ranges):
            dataset = DistributedDataset(
                topic=send_topic,
                partition_ranges=partition_ranges,
                consumer_params=self.data_loader_config["consumer_params"],
                payload_config=payload_config,
                is_last_split=(i == len(offset_ranges) - 1),
                last_fillup=self.data_loader_config["dataset_split_config"][-1]["fillup"]
            )
            datasets.append(dataset)
        return datasets
    
    def _check_partitions(self):
        """Check if Kafka topic partitions match WORLD_SIZE."""
        return check_partitions(
            self.data_loader_config["consumer_params"]['bootstrap_servers'], 
            self.data_loader_config["api_config"]["params"]["send_topic"]
        )
            
    def _create_dynamic_url(self):
        """Create dynamic URL for API request."""
        return create_dynamic_url(self.data_loader_config["api_config"])
    
    def _req_and_validate_res(self, url):
        """
        Make API request and validate response.
        
        Args:
            url (str): API endpoint URL
            
        Returns:
            tuple: (send_topic, offsets_data)
            
        Raises:
            KeyError: If required fields are missing from response
            ValueError: If data validation fails
        """
        response = requests.get(url)
        data = json.loads(response.text)
        
        if "sendTopicStr" not in data or data["sendTopicStr"] is None:
            raise KeyError("sendTopicStr is missing or null in the response data.")
        
        if "offsetsData" not in data or data["offsetsData"] is None:
            raise KeyError("offsetsData is missing or null in the response data.")
        
        if len(data["offsetsData"]) != self.world_size:
            raise ValueError("Mismatch between the number of offsets_data List and WORLD_SIZE.")
        
        return data["sendTopicStr"], data["offsetsData"]

    def _broadcast_tensor(self, tensor, src=0):
        """
        Broadcast tensor to all processes.
        
        Args:
            tensor (torch.Tensor): Tensor to broadcast
            src (int): Source rank
            
        Returns:
            torch.Tensor: Broadcasted tensor
        """
        size_tensor = torch.tensor([tensor.size(0)], dtype=torch.int, device=self.device)
        dist.broadcast(size_tensor, src=src)
        
        if dist.get_rank() != src:
            tensor = torch.empty(size_tensor.item(), dtype=tensor.dtype, device=self.device)
        
        dist.broadcast(tensor, src=src)
        return tensor

    def _broadcast_string(self, string, src=0):
        """
        Broadcast string to all processes.
        
        Args:
            string (str): String to broadcast
            src (int): Source rank
            
        Returns:
            str: Broadcasted string
            
        Raises:
            ValueError: If input is not a string
        """
        # Ensure that `string` is actually a string
        if not isinstance(string, str):
            raise ValueError("The input to _broadcast_string must be a string.")

        tensor = torch.tensor(bytearray(string.encode('utf-8')), dtype=torch.uint8, device=self.device)
        broadcasted_tensor = self._broadcast_tensor(tensor, src)
        return broadcasted_tensor.cpu().numpy().tobytes().decode('utf-8')
    
    def _broadcast_json(self, data, src=0):
        """
        Broadcast JSON-serializable data to all processes.
        
        Args:
            data: JSON-serializable data to broadcast
            src (int): Source rank
            
        Returns:
            Data after broadcasting and JSON deserialization
        """
        json_string = json.dumps(data)
        return json.loads(self._broadcast_string(json_string, src))