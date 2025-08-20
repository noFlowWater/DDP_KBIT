"""
Utils Module for DDP_KBIT

This module contains utility functions for Spark session management,
model checkpointing, and visualization for the DDP_KBIT distributed
deep learning system.
"""

from ddp_kbit.utils.spark_utils import (
    get_first_ip,
    load_config,
    create_spark_session,
    configure_spark_logging,
    check_kafka_partitions,
    load_from_kafka,
    setup_working_directory,
    initialize_distributed_training
)
from ddp_kbit.utils.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    list_available_checkpoints,
    cleanup_old_checkpoints,
    get_checkpoint_info
)
from ddp_kbit.utils.visualization import (
    calculate_boxplot_stats,
    print_statistical_analysis,
    plot_training_metrics,
    plot_predictions_grid,
    create_performance_comparison_boxplot,
    create_performance_bar_chart,
    save_results_to_json,
    load_results_from_json
)

__all__ = [
    # Spark utilities
    'get_first_ip',
    'load_config',
    'create_spark_session',
    'configure_spark_logging',
    'check_kafka_partitions',
    'load_from_kafka',
    'setup_working_directory',
    'initialize_distributed_training',
    
    # Checkpoint utilities
    'load_checkpoint',
    'save_checkpoint',
    'list_available_checkpoints',
    'cleanup_old_checkpoints',
    'get_checkpoint_info',
    
    # Visualization utilities
    'calculate_boxplot_stats',
    'print_statistical_analysis',
    'plot_training_metrics',
    'plot_predictions_grid',
    'create_performance_comparison_boxplot',
    'create_performance_bar_chart',
    'save_results_to_json',
    'load_results_from_json',
]