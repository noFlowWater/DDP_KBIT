#!/usr/bin/env python3
"""
ddp_kbit Main Entry Point

This is the main entry point for the ddp_kbit distributed deep learning system.
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
    from ddp_kbit.models.networks import create_cnn_model, create_feedforward_model
    from ddp_kbit.training.trainer import main_fn
    from ddp_kbit.experiments.runner import exp_fn, run_multiple_experiments
    from ddp_kbit.utils.spark_utils import create_spark_context, setup_working_directory
    from ddp_kbit.utils.visualization import print_statistical_analysis
    _IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Error importing ddp_kbit modules: {e}")
    print("Please ensure you're running from the correct directory and all dependencies are installed.")
    
    # 노트북 환경에서는 ImportError를 raise하지 않고 warning만 출력
    import warnings
    warnings.warn(f"Failed to import some ddp_kbit modules: {e}")
    _IMPORTS_SUCCESSFUL = False


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_external_config(config_path: Optional[str] = "config.json") -> Dict[str, Any]:
    """Load configuration from external JSON file if provided."""
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
            return {}
    return {}


def run_training_mode(config_dict: Optional[Dict[str, Any]] = None) -> None:
    """Run training mode using the main_fn from the original notebook."""
    if config_dict is None:
        # error handling
        raise ValueError("config_dict is required")

    if not _IMPORTS_SUCCESSFUL:
        print("❌ Training mode not available - missing dependencies")
        return
        
    logging.info("Starting training mode...")
    
    # Setup working directory
    setup_working_directory()
    
    sc = create_spark_context()
    
    try:
        from pyspark.ml.torch.distributor import TorchDistributor
        
        # Get configurations directly from config_dict
        train_config = config_dict["training_config"].copy()
        kafka_config = config_dict["kafka_config"].copy()
        data_loader_config = config_dict["data_loader_config"].copy()
        
        # Run the main training function using TorchDistributor
        result = TorchDistributor(
            num_processes=int(sc.getConf().get("spark.executor.instances")),
            local_mode=False,
            use_gpu=config_dict["training_config"]["use_gpu"]
        ).run(main_fn, train_config, kafka_config, data_loader_config, use_gpu=config_dict["training_config"]["use_gpu"])
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
    finally:
        if spark:
            spark.stop()


def run_experiment_mode(args: argparse.Namespace, config_dict: Optional[Dict[str, Any]] = None) -> None:
    """Run experiment mode using the exp_fn from the original notebook."""
    if not _IMPORTS_SUCCESSFUL:
        print("❌ Experiment mode not available - missing dependencies")
        return
        
    logging.info(f"Starting experiment mode: {args.experiment_type}")
    
    if config_dict is None:
        # error handling
        raise ValueError("config_dict is required")

    logging.info("Using provided config dictionary")
    
    
    # Setup working directory
    setup_working_directory()

    sc = create_spark_context()
    
    try:
        from pyspark.ml.torch.distributor import TorchDistributor
        
        # Get configurations directly from config_dict
        train_config = config_dict["training_config"].copy()
        kafka_config = config_dict["kafka_config"].copy()
        data_loader_config = config_dict["data_loader_config"].copy()
        
        if args.experiment_type == "single":
            # Run single experiment using TorchDistributor
            result = TorchDistributor(
                num_processes=int(sc.getConf().get("spark.executor.instances")),
                local_mode=False,
                use_gpu=config_dict["training_config"]["use_gpu"]
            ).run(exp_fn, train_config, kafka_config, data_loader_config, use_gpu=config_dict["training_config"]["use_gpu"])
        elif args.experiment_type == "multiple":
            results = run_multiple_experiments(sc, train_config, 
                                                kafka_config, data_loader_config, 
                                                iteration_count=args.iterations, 
                                                use_gpu=config_dict["training_config"]["use_gpu"])
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

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ddp_kbit Distributed Deep Learning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --mode train --distributed
    %(prog)s --mode experiment --experiment_type multiple --iterations 10
    %(prog)s --mode benchmark --iterations 5
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
        default="config.json",
        help="Path to the configuration file"
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
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.mode:
        parser.error("--mode is required")
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Log startup information
    logging.info("="*60)
    logging.info("ddp_kbit Distributed Deep Learning System")
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