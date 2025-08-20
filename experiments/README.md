# Experiments Module

This module contains experiment orchestration and performance benchmarking utilities extracted from the `sparkDL_KBIT_gpu_lightning.ipynb` notebook. It provides standalone Python modules for running distributed data loading experiments and analyzing their performance.

## Module Overview

### `runner.py` - Experiment Orchestration
Contains the main experiment runner logic including:
- `exp_fn()`: Core experiment function that runs 4 different data loading configurations
- `DistributedDataFetcher`: Class for managing distributed data fetching from Kafka
- `initialize_distributed_training()`: Distributed environment setup
- `run_multiple_experiments()`: Multi-iteration experiment runner
- Configuration validation and utilities

### `benchmarks.py` - Performance Analysis
Contains performance measurement and statistical analysis utilities:
- `calculate_boxplot_stats()`: Statistical analysis with boxplot metrics
- `PerformanceTimer`: Context manager for timing operations  
- `BenchmarkSuite`: Comprehensive benchmarking suite
- Visualization and reporting functions
- Data export/import capabilities

## Usage Examples

### Running a Single Experiment

```python
from DDP_KBIT.experiments import exp_fn, EXAMPLE_TRAINING_CONFIG, EXAMPLE_KAFKA_CONFIG, EXAMPLE_DATA_LOADER_CONFIG
from pyspark.ml.torch.distributor import TorchDistributor

# Run experiment through TorchDistributor
distributor = TorchDistributor(num_processes=3, local_mode=False, use_gpu=True)
results = distributor.run(
    exp_fn, 
    EXAMPLE_TRAINING_CONFIG, 
    EXAMPLE_KAFKA_CONFIG, 
    EXAMPLE_DATA_LOADER_CONFIG, 
    use_gpu=True
)

print("Experiment Results:", results)
```

### Running Multiple Iterations

```python
from DDP_KBIT.experiments import run_multiple_experiments

# Run 30 iterations of experiments
results = run_multiple_experiments(
    training_config=EXAMPLE_TRAINING_CONFIG,
    kafka_config=EXAMPLE_KAFKA_CONFIG, 
    data_loader_config=EXAMPLE_DATA_LOADER_CONFIG,
    iteration_count=30,
    use_gpu=True
)
```

### Performance Analysis

```python
from DDP_KBIT.experiments.benchmarks import BenchmarkSuite, analyze_experiment_results

# Create benchmark suite
suite = BenchmarkSuite("Data Loading Performance Test")

# Add results from experiments
sample_results = {
    "avro_lz4": [1.23, 1.45, 1.12, 1.67, 1.34],
    "avro_none": [2.34, 2.12, 2.45, 2.67, 2.23],
    "json_lz4": [1.89, 1.67, 1.98, 2.12, 1.87], 
    "json_none": [3.45, 3.23, 3.67, 3.89, 3.34]
}

for exp_name, times in sample_results.items():
    suite.add_results(exp_name, times)

# Run statistical analysis
analysis = suite.run_analysis()

# Generate performance comparison
comparison = suite.generate_comparison()

# Create visualizations
suite.create_visualizations(save_dir="./experiment_results")

# Export results
suite.export_results("experiment_results.json")
```

### Using Performance Timer

```python
from DDP_KBIT.experiments.benchmarks import PerformanceTimer, time_operation

# Using context manager
with PerformanceTimer("Data Loading") as timer:
    # Your data loading code here
    pass
print(f"Data loading took: {timer.elapsed:.3f} seconds")

# Using time_operation
with time_operation("Processing batch", print_result=True) as timing:
    # Your processing code here  
    pass
```

## Configuration Examples

### Training Configuration
```python
training_config = {
    "batch_size": 192,
    "num_epochs": 1, 
    "perform_validation": True
}
```

### Kafka Configuration
```python
kafka_config = {
    "bootstrap_servers": ["host1:9092", "host2:9092", "host3:9092"]
}
```

### Data Loader Configuration
```python
data_loader_config = {
    "data_loader_type": "kafka",
    "offsets_data": ["0:0:19999", "1:0:19999", "2:0:19999"],
    "consumer_params": {
        "bootstrap_servers": ["host1:9092", "host2:9092", "host3:9092"]
    },
    "dataset_split_config": [
        {"rate": 0.85715}, 
        {"rate": 0.071425}, 
        {"rate": 0.071425}
    ],
    "api_config": {
        "base_url": "http://api-server:3001",
        "endpoint": "data/export",
        "params": {
            "connection_id": "my-mongo-1",
            "mongo_database": "kbit-db", 
            "collection": "mnist_train_avro"
        }
    }
}
```

## Experiment Configurations

The `exp_fn` function runs 4 different experiment configurations:

1. **avro_lz4**: AVRO format with LZ4 compression
2. **avro_none**: AVRO format without compression  
3. **json_lz4**: JSON format with LZ4 compression
4. **json_none**: JSON format without compression

Each experiment measures data loading performance across all distributed processes and collects timing statistics.

## Statistical Analysis Features

The benchmarks module provides comprehensive statistical analysis:

- **Boxplot Statistics**: Q1, Q2 (median), Q3, IQR, fences
- **Descriptive Statistics**: Mean, variance, standard deviation, min/max
- **Outlier Detection**: Automatic detection using 1.5×IQR rule
- **Performance Comparison**: Rankings and relative performance ratios
- **Visualization**: Box plots and bar charts with error bars

## Dependencies

Core dependencies include:
- PyTorch and PyTorch Distributed
- Ignite Distributed
- Kafka Python client
- NumPy for statistical computations
- Matplotlib for visualizations
- PySpark for distributed execution

## Error Handling

Both modules include comprehensive error handling:
- Configuration validation
- Missing data handling
- Distributed environment error recovery  
- Memory cleanup after experiments

## Thread Safety

The modules are designed for distributed execution:
- Process-safe data collection
- Rank-aware logging and output
- Proper distributed cleanup

## Output Files

Results can be saved in multiple formats:
- JSON files with raw data and statistical analysis
- PNG visualizations with performance plots  
- Text reports with comprehensive analysis

## Performance Considerations

- Automatic memory cleanup after each experiment
- GPU memory management when available
- Efficient data structure usage for large result sets
- Configurable batch processing limits