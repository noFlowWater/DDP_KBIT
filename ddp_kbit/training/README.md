# ddp_kbit Training Module

This module contains extracted training-related functionality from the original `sparkDL_KBIT_gpu_lightning.ipynb` notebook, organized into three standalone Python modules with comprehensive documentation and error handling.

## Module Structure

```
ddp_kbit/training/
├── __init__.py          # Package initialization and exports
├── distributed.py       # Distributed training utilities
├── metrics.py          # Training metrics and evaluation
├── trainer.py          # Main training orchestration
├── example_usage.py    # Usage examples
└── README.md           # This documentation
```

## Modules Overview

### 1. `distributed.py` - Distributed Training Utilities

**Purpose**: Handles distributed training environment setup, TorchDistributor integration, and distributed utilities.

**Key Functions**:
- `initialize_distributed_training(use_gpu=True)`: Sets up distributed training environment
- `create_distributed_dataloader()`: Creates data loaders for distributed training
- `validate_distributed_environment()`: Validates distributed setup
- `cleanup_distributed_training()`: Cleans up distributed resources

**Features**:
- Automatic backend selection (NCCL for GPU, GLOO for CPU)
- Environment variable validation
- Device assignment based on local rank
- Error handling and cleanup
- TorchDistributor integration utilities

### 2. `metrics.py` - Training Metrics and Evaluation

**Purpose**: Provides comprehensive functionality for tracking, computing, and managing training metrics in distributed environments.

**Key Functions**:
- `reduce_dict()`: Reduces metrics across distributed processes
- `save_metrics()`: Saves engine metrics for tracking
- `debug_dataloader()`: Debug utility for data loaders
- `TrainingMetricsTracker`: Comprehensive metrics tracking class
- `PerformanceProfiler`: Performance measurement utilities

**Features**:
- Distributed metric aggregation
- Training progress tracking
- Performance profiling
- Custom metric implementations
- Integration with PyTorch Ignite metrics

### 3. `trainer.py` - Main Training Orchestration

**Purpose**: Contains the main training function with PyTorch Ignite integration and training loop logic.

**Key Functions**:
- `main_fn()`: Main distributed training function
- `TrainingConfig`: Configuration class for training parameters
- `DataLoaderConfig`: Configuration class for data loading
- Helper functions for common training scenarios

**Features**:
- Complete training orchestration
- PyTorch Ignite engine integration
- Flexible data loading (Kafka/local)
- Comprehensive evaluation and testing
- Error handling and cleanup

## Usage Examples

### Basic Usage

```python
from ddp_kbit.training import main_fn, create_simple_training_config, create_local_data_config
import torch.nn as nn

# Create a simple model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Create training configuration
training_config = create_simple_training_config(
    model=model,
    num_epochs=10,
    batch_size=64,
    learning_rate=0.001
)

# Create data configuration
data_loader_config = create_local_data_config("/path/to/data")

# Run training
results = main_fn(
    training_config=training_config,
    kafka_config={},
    data_loader_config=data_loader_config,
    use_gpu=True
)
```

### Advanced Configuration

```python
from ddp_kbit.training import main_fn, TrainingConfig, DataLoaderConfig
from ignite.metrics import Loss, Accuracy
import torch

# Custom training configuration
training_config = {
    'num_epochs': 20,
    'batch_size': 128,
    'perform_validation': True,
    'base_model': model,
    'optimizer_class': torch.optim.SGD,
    'optimizer_params': {
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4
    },
    'loss_fn': torch.nn.CrossEntropyLoss(),
    'metrics': {
        'accuracy': Accuracy(),
        'loss': Loss(torch.nn.CrossEntropyLoss())
    }
}

# Kafka data configuration
data_loader_config = {
    'data_loader_type': 'kafka',
    'consumer_params': {
        'bootstrap_servers': 'localhost:9092',
        'group_id': 'training-group'
    },
    'api_config': {
        'base_url': 'http://localhost:8080/api',
        'params': {'send_topic': 'training-data'}
    }
}

results = main_fn(training_config, {}, data_loader_config, use_gpu=True)
```

### Metrics Tracking

```python
from ddp_kbit.training import TrainingMetricsTracker

# Create metrics tracker
tracker = TrainingMetricsTracker()
tracker.start_training()

# During training
for epoch in range(num_epochs):
    tracker.start_epoch()
    
    # Training step
    train_metrics = {'loss': 0.5, 'accuracy': 0.85}
    tracker.save_metrics(train_metrics, 'train')
    
    # Validation step
    val_metrics = {'loss': 0.4, 'accuracy': 0.88}
    tracker.save_metrics(val_metrics, 'val')
    
    tracker.end_epoch()

# Print summary
tracker.print_summary(rank=0)
```

### Distributed Environment Setup

```python
from ddp_kbit.training import initialize_distributed_training, validate_distributed_environment

# Validate environment
if validate_distributed_environment():
    # Initialize distributed training
    config = initialize_distributed_training(use_gpu=True)
    print(f"Rank: {config['global_rank']}, Device: {config['device']}")
else:
    print("Environment not configured for distributed training")
```

## Running Distributed Training

### With torch.distributed.launch

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    your_training_script.py
```

### With TorchDistributor (Spark)

```python
from pyspark.ml.torch.distributor import TorchDistributor

distributor = TorchDistributor(
    num_processes=4,
    local_mode=False,
    use_gpu=True
)

result = distributor.run(main_fn, training_config, kafka_config, data_loader_config, True)
```

## Configuration Parameters

### Training Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_epochs` | int | Number of training epochs |
| `batch_size` | int | Batch size for training |
| `perform_validation` | bool | Whether to perform validation |
| `base_model` | torch.nn.Module | Model to train |
| `optimizer_class` | type | Optimizer class |
| `optimizer_params` | dict | Optimizer parameters |
| `loss_fn` | callable | Loss function |
| `metrics` | dict | Dictionary of Ignite metrics |

### Data Loader Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_loader_type` | str | 'kafka' or 'local' |
| `local_data_path` | str | Path for local data (if local) |
| `consumer_params` | dict | Kafka consumer parameters (if kafka) |
| `api_config` | dict | API configuration (if kafka) |

## Error Handling

The modules include comprehensive error handling:

- **Distributed Training**: Validates environment variables and handles initialization failures
- **Data Loading**: Checks data availability and validates configurations
- **Training Loop**: Catches and reports training failures with cleanup
- **Metrics**: Handles metric computation errors and missing data

## Performance Monitoring

Built-in performance monitoring includes:

- **Training Time**: Total and per-epoch timing
- **Data Loading Time**: Time spent loading data
- **Metric Computation**: Time for metric calculations
- **Memory Usage**: GPU/CPU memory monitoring
- **Throughput**: Samples per second tracking

## Integration with Original Notebook

These modules preserve all functionality from the original notebook:

- ✅ `main_fn` function with complete training logic
- ✅ `initialize_distributed_training` with environment setup
- ✅ PyTorch Ignite integration
- ✅ Distributed data loading support
- ✅ Kafka and local data loading
- ✅ Comprehensive metrics tracking
- ✅ TorchDistributor compatibility
- ✅ Error handling and cleanup

## Dependencies

Required packages:
- `torch`
- `torch.distributed`
- `pytorch-ignite`
- `kafka-python` (for Kafka data loading)
- `numpy`
- `tqdm` (optional, for progress bars)

## Best Practices

1. **Always validate the distributed environment** before training
2. **Use try-catch blocks** around training calls
3. **Monitor metrics regularly** during training
4. **Clean up resources** after training completion
5. **Test with small datasets** before full-scale training
6. **Use appropriate batch sizes** for your hardware
7. **Save checkpoints regularly** for long training runs

## Troubleshooting

### Common Issues

1. **Environment Variables Not Set**
   ```bash
   export MASTER_ADDR=localhost
   export MASTER_PORT=12355
   export RANK=0
   export WORLD_SIZE=1
   export LOCAL_RANK=0
   ```

2. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

3. **Data Loading Errors**
   - Check data paths
   - Validate Kafka connectivity
   - Ensure data format compatibility

For more examples and detailed usage, see `example_usage.py`.