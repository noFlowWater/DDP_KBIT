"""
Experiment runner module for distributed data loading performance testing.

This module contains the experiment orchestration logic extracted from the 
sparkDL_KBIT_gpu_lightning.ipynb notebook. It provides functions for running
distributed data loading experiments with different configurations and formats.
"""

import os
import time
import torch
import torch.distributed as dist
import ignite.distributed as idist
from typing import Dict, List, Any, Optional
from pyspark.context import SparkContext
import gc

# Import required modules from the project
try:
    from ddp_kbit.data.datasets import DistributedDataset
    from ddp_kbit.data.loaders import create_dataloaders
    from ddp_kbit.data.kafka_utils import split_offsets, create_dynamic_url
except ImportError:
    # Fallback imports for standalone usage
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.datasets import DistributedDataset
    from data.loaders import create_dataloaders
    from data.kafka_utils import split_offsets, create_dynamic_url


class DistributedDataFetcher:
    """
    Fetches and processes distributed data from Kafka topics for experiments.
    
    This class handles the creation of distributed datasets from Kafka topics,
    supporting multiple message formats and compression types.
    """
    
    def __init__(self, data_loader_config: dict, device: torch.device):
        """
        Initialize the data fetcher.
        
        Args:
            data_loader_config (dict): Configuration for data loading
            device (torch.device): Device for computation
        """
        self.data_loader_config = data_loader_config
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.device = device

    def fetch_n_pull_splited_datasets(self) -> List[DistributedDataset]:
        """
        Fetch data and create split datasets.
        
        Returns:
            List[DistributedDataset]: List of distributed datasets
        """
        self._validate_config()
        
        # Use offset data if available in config
        if "offsets_data" in self.data_loader_config and "offsets_data_topic" in self.data_loader_config:
            offsets_data = self.data_loader_config["offsets_data"]
            send_topic = self.data_loader_config["offsets_data_topic"]
        elif "api_config" in self.data_loader_config:
            api_config = self.data_loader_config["api_config"]
            url = create_dynamic_url(api_config)
            print(f"Rank {self.rank}: API URL = {url}")
        else:
            raise ValueError("offsets_data and offsets_data_topic or api_config are required in data_loader_config")
        
        # Split offsets according to configuration
        offset_ranges = self._split_offsets(offsets_data)
        
        # Create datasets for each split
        datasets = self._create_datasets(offset_ranges, send_topic)
        
        return datasets
    
    def _validate_config(self):
        """Validate the data loader configuration."""
        required_keys = ["consumer_params", "payload_config"]
        for key in required_keys:
            if key not in self.data_loader_config:
                raise ValueError(f"Missing required key '{key}' in data_loader_config")
    
    def _split_offsets(self, offsets_data):
        """Split offsets according to dataset split configuration."""
        return split_offsets(offsets_data, self.data_loader_config["dataset_split_config"])
    
    def _create_datasets(self, offset_ranges, send_topic):
        """Create distributed datasets from offset ranges."""
        datasets = []
        payload_config = self.data_loader_config.get("payload_config", {})
        consumer_params = self.data_loader_config["consumer_params"]
        
        for i, partition_ranges in enumerate(offset_ranges):
            is_last_split = (i == len(offset_ranges) - 1)
            last_fillup = True  # Default behavior
            
            dataset = DistributedDataset(
                topic=send_topic,
                partition_ranges=partition_ranges,
                consumer_params=consumer_params,
                payload_config=payload_config,
                is_last_split=is_last_split,
                last_fillup=last_fillup
            )
            datasets.append(dataset)
        
        return datasets


def initialize_distributed_training(use_gpu: bool = True) -> Dict[str, Any]:
    """
    Initialize the distributed training environment and set up the appropriate device.
    
    Args:
        use_gpu (bool): Whether to use GPU if available
        
    Returns:
        dict: Configuration dictionary containing backend, ranks, device info, etc.
    """
    import datetime
    
    # Set the appropriate backend based on whether GPU is used
    backend = "nccl" if use_gpu and torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, timeout=datetime.timedelta(seconds=5400))
    
    # Gather environment variables needed for distributed training
    env_dict = {}
    for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"]:
        env_dict[key] = os.environ.get(key, "Not Set")
    
    global_rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    
    # Set device based on GPU availability and configuration
    if use_gpu and torch.cuda.is_available():
        device_ids = [local_rank]
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device_ids = []
        device = torch.device("cpu")
    
    return {
        'backend': backend,
        'global_rank': global_rank,
        'local_rank': local_rank,
        'world_size': world_size,
        'device_ids': device_ids,
        'device': device,
        'env_dict': env_dict
    }


def exp_fn(training_config: Dict[str, Any], 
        data_loader_config: Dict[str, Any], 
        experiment_configs: List[Dict[str, Any]],
        use_gpu: bool = True) -> Dict[str, Any]:
    """
    Run distributed data loading experiments with multiple configurations.
    
    This function performs 4 different data loading experiments in a single 
    TorchDistributor call, testing AVRO (Lz4, none) and JSON (Lz4, none) 
    data loading performance.
    
    Args:
        training_config (Dict[str, Any]): Training configuration parameters
        kafka_config (Dict[str, Any]): Kafka connection configuration
        data_loader_config (Dict[str, Any]): Data loading configuration
        use_gpu (bool): Whether to use GPU acceleration
        
    Returns:
        Dict[str, Any]: Experiment results containing timing data for each configuration
    """
    print("Running data loading experiment with 4 configurations")
    
    # Initialize distributed environment
    init_config = initialize_distributed_training(use_gpu)
    device = idist.device()
    
    print(f"[PID {os.getpid()}] world_size = {init_config['world_size']}, "
        f"global_rank = {init_config['global_rank']}, local_rank = {init_config['local_rank']}")
    
    # Dictionary to store experiment results
    experiment_results = {}
    
    # Run experiments for each configuration
    for exp_config in experiment_configs:
        if init_config['global_rank'] == 0:
            print(f"\n===== {exp_config['name']} 실험 시작 =====")
        
        # Create configuration for current experiment
        current_data_loader_config = data_loader_config.copy()
        current_data_loader_config['offsets_data_topic'] = exp_config.get('topic')
        current_data_loader_config['payload_config'] = exp_config.get('payload_config')
        
        # Measure data loading time
        start_time = time.time()
        
        print(f"RANK[{init_config['global_rank']}] 실험: {exp_config['name']}, 토픽: {exp_config['topic']}")
        
        # Create data fetcher and load datasets
        if current_data_loader_config['data_loader_type'] == "kafka":
            splited_datasets = DistributedDataFetcher(current_data_loader_config, device=init_config["device"]) \
                        .fetch_n_pull_splited_datasets()
            
            if len(splited_datasets) == 0:
                raise ValueError("데이터셋 리스트가 비어 있습니다. 최소한 1개의 데이터셋이 필요합니다.")
                
            adjust_batch_size = int(training_config["batch_size"] / float(init_config['world_size']))
            print(f"Rank {init_config['global_rank']}: Creating DataLoader with batch size {adjust_batch_size}")

            train_loader, val_loader, test_loader = create_dataloaders(splited_datasets, adjust_batch_size)
        
        else:
            raise ValueError(f"Unsupported data_loader_type: {current_data_loader_config['data_loader_type']}")
        
        end_time = time.time()
        data_load_time = end_time - start_time
        
        # Gather timing data from all ranks
        all_data_load_times = [None for _ in range(init_config['world_size'])]
        dist.all_gather_object(all_data_load_times, data_load_time)
        
        # Gather batch information from all ranks
        batch_info = {
            'train_batches': len(train_loader),
            'val_batches': len(val_loader) if val_loader else 0,
            'test_batches': len(test_loader) if test_loader else 0
        }
        all_batch_info = [None for _ in range(init_config['world_size'])]
        dist.all_gather_object(all_batch_info, batch_info)
        
        if init_config['global_rank'] == 0:
            print(f"모든 Rank의 데이터 로딩 시간: {[f'{t:.3f}초' for t in all_data_load_times]}")
            print("각 Rank별 배치 정보:")
            for rank_idx, batch_info in enumerate(all_batch_info):
                print(f"Rank {rank_idx}: Train={batch_info['train_batches']}, Val={batch_info['val_batches']}, Test={batch_info['test_batches']}")
        
        # Store results (including all rank times)
        experiment_results[exp_config['name']] = {
            'all_data_load_times': all_data_load_times,  # All rank times
            'avg_data_load_time': sum(all_data_load_times) / len(all_data_load_times),  # Average
            'min_data_load_time': min(all_data_load_times),  # Minimum
            'max_data_load_time': max(all_data_load_times),  # Maximum
            'all_batch_info': all_batch_info  # All rank batch information
        }
        
        if init_config['global_rank'] == 0:
            avg_time = experiment_results[exp_config['name']]['avg_data_load_time']
            min_time = experiment_results[exp_config['name']]['min_data_load_time']
            max_time = experiment_results[exp_config['name']]['max_data_load_time']
            print(f'{exp_config["name"]} - 평균: {avg_time:.2f}초, 최소: {min_time:.2f}초, 최대: {max_time:.2f}초')
        
        # Memory cleanup
        del train_loader, val_loader, test_loader, splited_datasets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Clean up distributed environment
    dist.destroy_process_group()
    
    if init_config['global_rank'] == 0:
        print("\n===== 실험 결과 요약 =====")
        for exp_name, results in experiment_results.items():
            avg_time = results['avg_data_load_time']
            min_time = results['min_data_load_time']
            max_time = results['max_data_load_time']
            print(f"{exp_name}: 평균={avg_time:.2f}초, 최소={min_time:.2f}초, 최대={max_time:.2f}초")
    
    return experiment_results


def run_multiple_experiments(sc: SparkContext, training_config: Dict[str, Any], 
                            experiment_configs: List[Dict[str, Any]],
                            data_loader_config: Dict[str, Any], 
                            iteration_count: int = 30,
                            use_gpu: bool = True) -> Dict[str, List[float]]:
    """
    Run multiple iterations of experiments and collect results.
    
    Args:
        training_config: Training configuration
        kafka_config: Kafka configuration
        data_loader_config: Data loader configuration
        iteration_count: Number of iterations to run
        use_gpu: Whether to use GPU
        
    Returns:
        Dict containing aggregated results from all iterations
    """
    from pyspark.ml.torch.distributor import TorchDistributor

    # Initialize results storage (all rank times)
    total_results = {
        "avro_lz4": [],
        "avro_none": [],
        "json_lz4": [],
        "json_none": [],
    }
    
    print(f"Starting {iteration_count} iterations of experiments...")
    
    for i in range(iteration_count):
        print(f"\n{'='*50}")
        print(f"실험 반복 {i+1}/{iteration_count}")
        print(f"{'='*50}")
        
        # Run single experiment iteration
        result = TorchDistributor(
            num_processes=int(sc.getConf().get("spark.executor.instances")),
            local_mode=False,
            use_gpu=use_gpu
        ).run(exp_fn, training_config, data_loader_config, experiment_configs, use_gpu=use_gpu)
        
        # Add results to total collection (all rank times)
        for exp_name in total_results.keys():
            if exp_name in result:
                # Add all rank data loading times
                total_results[exp_name].extend(result[exp_name]['all_data_load_times'])
        
        print(f"반복 {i+1} 완료")
    
    print(f"\n총 {iteration_count}회 반복 실험 완료!")
    return total_results


def print_experiment_summary(results: Dict[str, List[float]]):
    """
    Print a summary of experiment results.
    
    Args:
        results: Dictionary containing experiment results for each configuration
    """
    print("\n" + "="*80)
    print("실험 결과 요약 (모든 반복, 모든 Rank 포함)")
    print("="*80)
    
    for exp_name, times in results.items():
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"\n🔍 {exp_name.upper()}:")
            print(f"  총 데이터 수: {len(times)}개")
            print(f"  평균: {avg_time:.3f}초")
            print(f"  최소: {min_time:.3f}초")
            print(f"  최대: {max_time:.3f}초")
        else:
            print(f"\n❌ {exp_name}: 결과 없음")


def validate_experiment_config(training_config: Dict[str, Any],
                             kafka_config: Dict[str, Any],
                             data_loader_config: Dict[str, Any]) -> bool:
    """
    Validate experiment configuration parameters.
    
    Args:
        training_config: Training configuration to validate
        kafka_config: Kafka configuration to validate
        data_loader_config: Data loader configuration to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
        
    Raises:
        ValueError: If configuration is invalid with detailed error message
    """
    # Validate training config
    required_training_keys = ['batch_size', 'num_epochs']
    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"Missing required training config key: {key}")
    
    # Validate kafka config
    required_kafka_keys = ['bootstrap_servers']
    for key in required_kafka_keys:
        if key not in kafka_config:
            raise ValueError(f"Missing required kafka config key: {key}")
    
    # Validate data loader config
    required_data_loader_keys = ['data_loader_type', 'consumer_params', 'payload_config']
    for key in required_data_loader_keys:
        if key not in data_loader_config:
            raise ValueError(f"Missing required data loader config key: {key}")
    
    if data_loader_config['data_loader_type'] == 'kafka':
        kafka_required_keys = ['offsets_data', 'offsets_data_topic']
        for key in kafka_required_keys:
            if key not in data_loader_config:
                raise ValueError(f"Missing required kafka data loader config key: {key}")
    
    return True


# Example usage and configuration templates
EXAMPLE_TRAINING_CONFIG = {
    "batch_size": 192,
    "num_epochs": 1,
    "perform_validation": True
}

EXAMPLE_KAFKA_CONFIG = {
    "bootstrap_servers": ["155.230.35.200:32100", "155.230.35.213:32100", "155.230.35.215:32100"]
}

EXAMPLE_DATA_LOADER_CONFIG = {
    "data_loader_type": "kafka",
    "offsets_data": ["0:0:19999", "1:0:19999", "2:0:19999"],
    "consumer_params": {
        "bootstrap_servers": ["155.230.35.200:32100", "155.230.35.213:32100", "155.230.35.215:32100"]
    },
    "dataset_split_config": [
        {"rate": 0.85715}, 
        {"rate": 0.071425}, 
        {"rate": 0.071425}
    ],
    "api_config": {
        "base_url": "http://155.230.36.25:3001",
        "endpoint": "data/export",
        "params": {
            "connection_id": "my-mongo-1",
            "mongo_database": "kbit-db",
            "collection": "mnist_train_avro",
            "send_topic": "kbit-p3r1"
        }
    }
}


if __name__ == "__main__":
    # Example of running a single experiment
    print("Running single experiment example...")
    
    try:
        # Validate configuration
        validate_experiment_config(
            EXAMPLE_TRAINING_CONFIG, 
            EXAMPLE_KAFKA_CONFIG, 
            EXAMPLE_DATA_LOADER_CONFIG
        )
        
        # Note: This would normally be run through TorchDistributor
        # results = exp_fn(EXAMPLE_TRAINING_CONFIG, EXAMPLE_KAFKA_CONFIG, EXAMPLE_DATA_LOADER_CONFIG)
        # print_experiment_summary(results)
        
        print("Configuration validation passed!")
        
    except Exception as e:
        print(f"Error: {e}")