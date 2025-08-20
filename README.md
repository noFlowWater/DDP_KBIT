# DDP_KBIT: Distributed Deep Learning with Kafka-Based Intelligent Training

A modular distributed deep learning system extracted from `sparkDL_KBIT_gpu_lightning.ipynb` for scalable machine learning with Apache Spark, PyTorch, and Apache Kafka.

## 🚀 Features

- **Distributed Training**: PyTorch distributed training with PySpark integration
- **Kafka Data Streaming**: Real-time data consumption from Apache Kafka
- **Multiple Data Formats**: Support for JSON, Avro, Protobuf, and MongoDB formats
- **GPU Acceleration**: RAPIDS GPU acceleration support
- **Performance Benchmarking**: Comprehensive experiment tracking and analysis
- **Modular Architecture**: Clean, maintainable code structure

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (optional, for GPU acceleration)
- Apache Kafka (for data streaming features)

### Install from Source
```bash
# Clone the repository
git clone <repository-url>
cd DDP_KBIT

# Install the package
pip install -e .
```

**⚠️ 중요**: 패키지 이름은 `DDP_KBIT`입니다 (대문자). `ddp_kbit`가 아닙니다.

### Install Dependencies Only
```bash
# Install all required dependencies
pip install -r requirements.txt

# Or install with extras for development and notebook support
pip install -r requirements.txt[dev,notebook]
```

### Verify Installation
```bash
# Check if package is installed correctly
python -c "import DDP_KBIT; print('DDP_KBIT installed successfully!')"

# Check available command line tools
ddp-kbit --help

# Run comprehensive installation test
python install_test.py
```

## 🎯 Quick Start

### 1. Generate Sample Configuration
```bash
# Using the command line tool
ddp-kbit --create_sample_config

# Or using Python directly
python -m DDP_KBIT.main --create_sample_config
```

This creates `sample_config.json` with default settings:
```json
{
  "spark_config": {
    "master": "local[*]",
    "app_name": "DDP_KBIT_Sample",
    "executor_instances": 2,
    "executor_cores": 2,
    "executor_memory": "4g"
  },
  "training_config": {
    "epochs": 5,
    "batch_size": 64,
    "learning_rate": 0.001
  },
  "data_config": {
    "kafka_servers": ["localhost:9092"],
    "topic": "mnist_topic",
    "batch_size": 32
  }
}
```

### 2. Run Training
```bash
# Single node training
ddp-kbit --mode train --config_path sample_config.json

# Distributed training
ddp-kbit --mode train --distributed --config_path sample_config.json

# Using Python module directly
python -m DDP_KBIT.main --mode train --config_path sample_config.json
```

### 3. Run Experiments
```bash
# Single experiment
ddp-kbit --mode experiment --experiment_type single

# Multiple iterations with statistical analysis
ddp-kbit --mode experiment --experiment_type multiple --iterations 10

# Using Python module directly
python -m DDP_KBIT.main --mode experiment --experiment_type multiple --iterations 10
```

## 📚 Usage Examples

### Python Script Usage
```python
from DDP_KBIT.main import run_training_mode, run_experiment_mode
from DDP_KBIT.config import training_config, data_config, spark_config

# Run training programmatically
run_training_mode(None)  # Uses default configuration

# Run experiments
run_experiment_mode(None)  # Uses default configuration
```

### Jupyter Notebook Usage
```python
# Import the notebook interface
from DDP_KBIT.notebook_interface import setup_module_path, run_training_mode

# Setup module path
setup_module_path()

# Run training
run_training_mode(None)
```

### Command Line Interface
```bash
# Show help
ddp-kbit --help

# Available modes
ddp-kbit --mode train --help
ddp-kbit --mode experiment --help
ddp-kbit --mode benchmark --help

# Training with custom config
ddp-kbit --mode train --config_path my_config.json --distributed

# Experiments with custom parameters
ddp-kbit --mode experiment --experiment_type multiple --iterations 20
```

## 🔧 Configuration

### Training Configuration
Located in `config/training_config.py`:
- **Hyperparameters**: Learning rate, batch size, epochs
- **Model settings**: Network architecture options
- **Optimization**: Optimizer and loss function settings

### Data Configuration
Located in `config/data_config.py`:
- **Kafka settings**: Bootstrap servers, topics, consumer parameters
- **Data formats**: JSON, Avro, Protobuf, MongoDB support
- **Transform functions**: Data preprocessing pipelines

### Spark Configuration
Located in `config/spark_config.py`:
- **Cluster settings**: Master URL, executor configuration
- **GPU settings**: RAPIDS acceleration, memory allocation
- **JAR dependencies**: Kafka and RAPIDS JAR files

## 📊 Data Formats

DDP_KBIT supports multiple data formats for flexible data ingestion:

### 1. JSON Format
```json
{
  "image": [pixel_values],
  "label": 5
}
```

### 2. Avro Format
Uses schema definitions in `data_config.py`:
- MNIST v1: Basic image and label schema
- MNIST v2: Extended schema with metadata

### 3. MongoDB Format
```json
{
  "images": [pixel_values],
  "label": 5,
  "metadata": {...}
}
```

### 4. Protobuf Format
Binary protocol buffer format for efficient serialization.

## 🧠 Models

### CNN Model (Net)
- 2 Convolutional layers with ReLU and MaxPool
- 2 Fully connected layers with dropout
- Designed for 28x28 grayscale images (MNIST)

### Feedforward Model (NeuralNetwork)
- Single hidden layer (784 → 128 → 10)
- ReLU activation with softmax output
- Simpler alternative to CNN

## 🚄 Distributed Training

### Setup Distributed Environment
```python
from DDP_KBIT.training import initialize_distributed_training

# Initialize distributed training
initialize_distributed_training()
```

### TorchDistributor Integration
```python
from pyspark.ml.torch.distributor import TorchDistributor
from DDP_KBIT.training import main_fn

distributor = TorchDistributor(
    num_processes=4,
    local_mode=False,
    use_gpu=True
)

distributor.run(main_fn)
```

## 📈 Experiments and Benchmarking

### Running Experiments Programmatically
```python
from DDP_KBIT.experiments import exp_fn, run_multiple_experiments

# Single experiment
exp_fn()

# Multiple experiments with statistics
results = run_multiple_experiments(iterations=10)
```

## 🔍 Monitoring and Visualization

### Performance Metrics
```python
from DDP_KBIT.utils import plot_training_metrics, calculate_boxplot_stats

# Plot training curves
plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

# Statistical analysis
stats = calculate_boxplot_stats(performance_data)
```

### Checkpoint Management
```python
from DDP_KBIT.utils import save_checkpoint, load_checkpoint

# Save model checkpoint
save_checkpoint(model, optimizer, epoch, "model_v1")

# Load checkpoint
checkpoint = load_checkpoint("model_v1")
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Use gradient accumulation
   - Enable mixed precision training

2. **Kafka Connection Issues**
   - Verify Kafka broker addresses
   - Check topic existence and partitions
   - Validate consumer group settings

3. **Spark Configuration Issues**
   - Ensure JAR files are accessible
   - Check memory allocation settings
   - Verify GPU discovery scripts

4. **Import Errors**
   - Ensure package is installed correctly: `pip install -e .`
   - Check Python path and working directory
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - **Important**: Use `import DDP_KBIT` (uppercase), not `import ddp_kbit`
   - Run `python install_test.py` for comprehensive testing

### Logging
Enable debug logging for detailed information:
```bash
ddp-kbit --mode train --log_level DEBUG
```

## 🔧 Advanced Usage

### Custom Model Integration
```python
from DDP_KBIT.models import get_model_by_name
from DDP_KBIT.training import main_fn

# Use custom model
model = get_model_by_name("cnn")  # or "feedforward"
```

### Custom Data Transformations
```python
from DDP_KBIT.data.transforms import transform_RawMNISTData

# Apply custom transforms
transformed_data = transform_RawMNISTData(raw_image_bytes)
```

### Spark Session Customization
```python
from DDP_KBIT.utils import create_spark_session

# Custom Spark configuration
spark = create_spark_session(
    app_name="Custom_App"
)
```

## 📝 Development

### Adding New Models
1. Define model class in `models/networks.py`
2. Add factory function
3. Update `__init__.py` exports

### Adding New Experiments
1. Implement experiment function in `experiments/runner.py`
2. Add configuration options
3. Update CLI interface in `main.py`

### Contributing
1. Follow existing code structure and naming conventions
2. Add comprehensive docstrings and type hints
3. Include error handling and logging
4. Update documentation and examples

## 📚 API Reference

### Main Functions
- `main_fn()`: Main training orchestration function
- `exp_fn()`: Experiment runner with multiple data format tests
- `initialize_distributed_training()`: Distributed environment setup

### Configuration Functions
- `get_training_config()`: Get training hyperparameters
- `get_spark_config()`: Get Spark session configuration

### Utility Functions
- `create_spark_session()`: Create configured Spark session
- `calculate_boxplot_stats()`: Statistical analysis
- `plot_training_metrics()`: Visualization utilities

## 📄 License

This project is derived from the `sparkDL_KBIT_gpu_lightning.ipynb` notebook and maintains the same licensing terms.

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section
2. Review configuration examples
3. Enable debug logging for detailed diagnostics
4. Consult the original notebook for reference implementation
5. Ensure proper installation with `pip install -e .`