# ddp_kbit

Distributed Deep Learning with Kafka-Based Intelligent Training

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Generate sample config (currently has dependency issues)
python -m ddp_kbit.main --create_sample_config

# Run training
python -m ddp_kbit.main --mode train --config_path sample_config.json

# Run experiments  
python -m ddp_kbit.main --mode experiment --experiment_type single
```

## Python Usage

```python
# Import through __init__.py exports
import ddp_kbit
from ddp_kbit import main_fn, exp_fn

# Or direct imports (when modules are fixed)
from ddp_kbit.training.trainer import main_fn
from ddp_kbit.experiments.runner import exp_fn

# Training
main_fn()

# Experiments
exp_fn()
```

## Configuration

Edit `sample_config.json`:
- `training_config`: epochs, batch_size, learning_rate
- `spark_config`: cluster settings, memory allocation  
- `data_config`: Kafka settings, data formats

## Features

- Distributed PyTorch training with PySpark
- Kafka data streaming (JSON, Avro, Protobuf, MongoDB)
- GPU acceleration support
- Multiple model architectures (CNN, feedforward)
- Experiment tracking and visualization

## Troubleshooting

**Import errors**: Use `import ddp_kbit` (lowercase with underscore)  
**CUDA memory**: Reduce batch_size in config  
**Spark errors**: Use `"master": "local[*]"` for local mode