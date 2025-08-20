#!/usr/bin/env python3
"""
DDP_KBIT Main Entry Point

This is the main entry point for the DDP_KBIT distributed deep learning system.
It provides a command-line interface to run various experiments and training tasks
extracted from the sparkDL_KBIT_gpu_lightning.ipynb notebook.

Usage:
    python main.py --mode train --config_path config.json
    python main.py --mode benchmark --iterations 5
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, Optional

# Add the current directory to the Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 임포트 시도
_IMPORTS_SUCCESSFUL = False

try:
    # 절대 임포트 시도
    try:
        from DDP_KBIT.config import training_config,data_config, spark_config
        from DDP_KBIT.models.networks import create_cnn_model, create_feedforward_model
        from DDP_KBIT.training.trainer import main_fn
        from DDP_KBIT.experiments.runner import exp_fn, run_multiple_experiments
        from DDP_KBIT.utils.spark_utils import create_spark_session, setup_working_directory
        from DDP_KBIT.utils.visualization import print_statistical_analysis
        _IMPORTS_SUCCESSFUL = True
    except ImportError:
        # 상대 임포트 시도
        from config import training_config, data_config, spark_config
        from models.networks import create_cnn_model, create_feedforward_model
        from training.trainer import main_fn
        from experiments.runner import exp_fn, run_multiple_experiments
        from utils.spark_utils import create_spark_session, setup_working_directory
        from utils.visualization import print_statistical_analysis
        _IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Error importing DDP_KBIT modules: {e}")
    print("Please ensure you're running from the correct directory and all dependencies are installed.")
    
    # 노트북 환경에서는 ImportError를 raise하지 않고 warning만 출력
    import warnings
    warnings.warn(f"Failed to import some DDP_KBIT modules: {e}")
    _IMPORTS_SUCCESSFUL = False


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_external_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from external JSON file if provided."""
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
            return {}
    return {}


def run_training_mode(args: argparse.Namespace) -> None:
    """Run training mode using the main_fn from the original notebook."""
    if not _IMPORTS_SUCCESSFUL:
        print("❌ Training mode not available - missing dependencies")
        return
        
    logging.info("Starting training mode...")
    
    # Load external configuration if provided
    external_config = load_external_config(args.config_path)
    
    # Setup working directory
    setup_working_directory()
    
    # Create Spark session
    spark = create_spark_session(
        app_name="DDP_KBIT_Training",
    )
    
    try:
        from pyspark.ml.torch.distributor import TorchDistributor
        
        # Get configurations
        train_config = training_config.get_complete_training_config()
        kafka_config = data_config.KAFKA_CONFIG
        data_loader_config = data_config.DATA_LOADER_CONFIG
        
        # Get number of processes from Spark configuration
        num_processes = int(spark.conf.get("spark.executor.instances"))
        
        # Run the main training function using TorchDistributor
        result = TorchDistributor(
            num_processes=num_processes,
            local_mode=False,
            use_gpu=train_config["use_gpu"]
        ).run(main_fn, train_config, kafka_config, data_loader_config, use_gpu=train_config["use_gpu"])
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
    finally:
        if spark:
            spark.stop()


def run_experiment_mode(args: argparse.Namespace) -> None:
    """Run experiment mode using the exp_fn from the original notebook."""
    if not _IMPORTS_SUCCESSFUL:
        print("❌ Experiment mode not available - missing dependencies")
        return
        
    logging.info(f"Starting experiment mode: {args.experiment_type}")
    
    # Load external configuration if provided
    external_config = load_external_config(args.config_path)
    
    # Setup working directory
    setup_working_directory()
    
    # Create Spark session
    spark = create_spark_session(
        app_name="DDP_KBIT_Experiments",
    )
    
    try:
        from pyspark.ml.torch.distributor import TorchDistributor
        
        # Get configurations
        train_config = training_config.get_complete_training_config()
        kafka_config = data_config.KAFKA_CONFIG
        data_loader_config = data_config.DATA_LOADER_CONFIG
        
        # Get number of processes from Spark configuration
        num_processes = int(spark.conf.get("spark.executor.instances"))
        
        if args.experiment_type == "single":
            # Run single experiment using TorchDistributor
            result = TorchDistributor(
                num_processes=num_processes,
                local_mode=False,
                use_gpu=True
            ).run(exp_fn, train_config, kafka_config, data_loader_config, use_gpu=True)
        elif args.experiment_type == "multiple":
            # Run multiple iterations using TorchDistributor
            results = TorchDistributor(
                num_processes=num_processes,
                local_mode=False,
                use_gpu=True
            ).run(run_multiple_experiments, iterations=args.iterations)
            print_statistical_analysis(results)
        else:
            logging.error(f"Unknown experiment type: {args.experiment_type}")
            
        logging.info("Experiments completed successfully!")
        
    except Exception as e:
        logging.error(f"Experiments failed: {e}")
        raise
    finally:
        if spark:
            spark.stop()


def create_sample_config() -> None:
    """Create a sample configuration file using existing config modules dynamically."""
    if not _IMPORTS_SUCCESSFUL:
        print("❌ Create sample config not available - missing dependencies")
        return
        
    # Import configuration constants and functions
    from config.training_config import get_complete_training_config
    from config.data_config import KAFKA_CONFIG, MONGO_CONFIG, DATA_LOADER_CONFIG, PAYLOAD_CONFIG
    from config.spark_config import SPARK_CONFIG
    
    # Get configurations from modules
    training_config = get_complete_training_config()
    
    # Convert non-serializable objects to strings for JSON compatibility
    serializable_training_config = {
        "base_model_type": training_config["base_model"].__class__.__name__,
        "optimizer_class": f"{training_config['optimizer_class'].__module__}.{training_config['optimizer_class'].__name__}",
        "optimizer_params": training_config["optimizer_params"],
        "loss_fn": f"{training_config['loss_fn'].__class__.__module__}.{training_config['loss_fn'].__class__.__name__}",
        "perform_validation": training_config["perform_validation"],
        "num_epochs": training_config["num_epochs"],
        "batch_size": training_config["batch_size"],
        "metrics": {k: v.__class__.__name__ for k, v in training_config["metrics"].items()}
    }
    
    # Create serializable data loader config
    serializable_data_loader_config = DATA_LOADER_CONFIG.copy()
    # Convert transform function references to strings
    if "payload_config" in serializable_data_loader_config:
        payload_config = serializable_data_loader_config["payload_config"].copy()
        if "transform_data_fn" in payload_config and callable(payload_config["transform_data_fn"]):
            payload_config["transform_data_fn"] = payload_config["transform_data_fn"].__name__
        if "transform_label_fn" in payload_config and callable(payload_config["transform_label_fn"]):
            payload_config["transform_label_fn"] = payload_config["transform_label_fn"].__name__
        serializable_data_loader_config["payload_config"] = payload_config
    
    # Create serializable payload config
    serializable_payload_config = PAYLOAD_CONFIG.copy()
    if "transform_data_fn" in serializable_payload_config and callable(serializable_payload_config["transform_data_fn"]):
        serializable_payload_config["transform_data_fn"] = serializable_payload_config["transform_data_fn"].__name__
    if "transform_label_fn" in serializable_payload_config and callable(serializable_payload_config["transform_label_fn"]):
        serializable_payload_config["transform_label_fn"] = serializable_payload_config["transform_label_fn"].__name__
    
    # Create the complete configuration using existing modules
    sample_config = {
        # "spark_config": {
        #     "master": "local[*]",  # Override for local development
        #     "app_name": "DDP_KBIT_Sample",
        #     "executor_instances": 2,  # Reduced for sample
        #     "executor_cores": 2,      # Reduced for sample  
        #     "executor_memory": "4g",  # Reduced for sample
        #     # Include some key configs from SPARK_CONFIG
        #     "driver_host_resolution": "dynamic",
        #     "gpu_enabled": True,
        #     "rapids_enabled": True
        # },
        "training_config": serializable_training_config,
        "mongo_config": MONGO_CONFIG,
        "kafka_config": KAFKA_CONFIG,
        "data_loader_config": serializable_data_loader_config,
        "payload_config": serializable_payload_config,
        "metadata": {
            "generated_from": "DDP_KBIT.config modules",
            "description": "Sample configuration dynamically generated from existing config modules",
            "usage_note": "This config uses the same settings as defined in the original notebook cell 24"
        }
    }
    
    with open("sample_config.json", "w") as f:
        json.dump(sample_config, f, indent=2)
    
    print("Created sample_config.json - customize this file for your needs.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DDP_KBIT Distributed Deep Learning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --mode train --distributed
    %(prog)s --mode experiment --experiment_type multiple --iterations 10
    %(prog)s --mode benchmark --iterations 5
    %(prog)s --create_sample_config
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["train", "experiment"],
        help="Execution mode"
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to external JSON configuration file"
    )
    
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training (for train mode)"
    )
    
    parser.add_argument(
        "--experiment_type",
        choices=["single", "multiple"],
        default="single",
        help="Type of experiment to run (for experiment mode)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations to run (for experiment/benchmark modes)"
    )
    
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--create_sample_config",
        action="store_true",
        help="Create a sample configuration file and exit"
    )
    
    args = parser.parse_args()
    
    # Handle special case for creating sample config
    if args.create_sample_config:
        create_sample_config()
        return
    
    # Validate arguments
    if not args.mode:
        parser.error("--mode is required unless using --create_sample_config")
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Log startup information
    logging.info("="*60)
    logging.info("DDP_KBIT Distributed Deep Learning System")
    logging.info("="*60)
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Config path: {args.config_path}")
    logging.info(f"Log level: {args.log_level}")
    
    try:
        # Route to appropriate mode handler
        if args.mode == "train":
            run_training_mode(args)
        elif args.mode == "experiment":
            run_experiment_mode(args)
            
    except KeyboardInterrupt:
        logging.info("Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Execution failed: {e}")
        sys.exit(1)
    
    logging.info("Execution completed successfully!")


if __name__ == "__main__":
    main()