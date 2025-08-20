"""
Visualization utilities for distributed deep learning experiments.

This module provides utilities for plotting training metrics, creating visualizations,
and calculating statistical summaries including boxplot statistics.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


def calculate_boxplot_stats(data: List[float]) -> Optional[Dict[str, Any]]:
    """
    Calculate boxplot statistics and additional statistical measures.
    
    Args:
        data (List[float]): List of numerical data points.
    
    Returns:
        Optional[Dict[str, Any]]: Dictionary containing statistical measures:
            - count: Number of data points
            - q1: First quartile (25th percentile)
            - median: Second quartile (50th percentile)
            - q3: Third quartile (75th percentile)
            - iqr: Interquartile range
            - lower_fence: Lower fence for outlier detection
            - upper_fence: Upper fence for outlier detection
            - lower_outliers: List of lower outliers
            - upper_outliers: List of upper outliers
            - mean: Arithmetic mean
            - variance: Variance
            - std_dev: Standard deviation
            - min: Minimum value
            - max: Maximum value
        
        Returns None if data is empty.
    """
    if not data:
        logger.warning("Empty data provided to calculate_boxplot_stats")
        return None
    
    try:
        data_array = np.array(data)
        
        # Boxplot 5-number summary statistics
        q1 = np.percentile(data_array, 25)
        q2 = np.percentile(data_array, 50)  # median
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        
        # Calculate fences for outlier detection
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        
        # Find outliers
        lower_outliers = [x for x in data if x < lower_fence]
        upper_outliers = [x for x in data if x > upper_fence]
        
        # Additional statistical measures
        mean_val = np.mean(data_array)
        variance_val = np.var(data_array, ddof=1) if len(data_array) > 1 else 0
        std_dev_val = np.std(data_array, ddof=1) if len(data_array) > 1 else 0
        min_val = np.min(data_array)
        max_val = np.max(data_array)
        
        return {
            'count': len(data_array),
            'q1': float(q1),
            'median': float(q2),
            'q3': float(q3),
            'iqr': float(iqr),
            'lower_fence': float(lower_fence),
            'upper_fence': float(upper_fence),
            'lower_outliers': lower_outliers,
            'upper_outliers': upper_outliers,
            'mean': float(mean_val),
            'variance': float(variance_val),
            'std_dev': float(std_dev_val),
            'min': float(min_val),
            'max': float(max_val)
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate boxplot statistics: {e}")
        return None


def print_statistical_analysis(
    results_dict: Dict[str, List[float]],
    title: str = "Statistical Analysis"
) -> None:
    """
    Print detailed statistical analysis for multiple datasets.
    
    Args:
        results_dict (Dict[str, List[float]]): Dictionary mapping dataset names to data lists.
        title (str): Title for the analysis output. Defaults to "Statistical Analysis".
    """
    print(title)
    print("=" * 80)
    
    for key, results in results_dict.items():
        if results:
            stats = calculate_boxplot_stats(results)
            if stats:
                print(f"\n🔍 {key.upper()} Statistical Analysis:")
                print(f"   Total data points: {stats['count']}")
                
                print(f"   \n📊 Boxplot 5-number summary:")
                print(f"      Q1 (25th): {stats['q1']:.3f}s")
                print(f"      Q2 (median): {stats['median']:.3f}s")
                print(f"      Q3 (75th): {stats['q3']:.3f}s")
                print(f"      IQR: {stats['iqr']:.3f}s")
                print(f"      Lower Fence: {stats['lower_fence']:.3f}s")
                print(f"      Upper Fence: {stats['upper_fence']:.3f}s")
                
                print(f"   \n📈 Basic statistics:")
                print(f"      Mean: {stats['mean']:.3f}s")
                print(f"      Variance: {stats['variance']:.6f}")
                print(f"      Standard deviation: {stats['std_dev']:.3f}s")
                print(f"      Minimum: {stats['min']:.3f}s")
                print(f"      Maximum: {stats['max']:.3f}s")
                
                print(f"   \n⚠️  Outlier analysis:")
                if stats['lower_outliers']:
                    print(f"      Lower outliers ({len(stats['lower_outliers'])}): {stats['lower_outliers'][:5]}{'...' if len(stats['lower_outliers']) > 5 else ''}")
                else:
                    print("      No lower outliers found")
                    
                if stats['upper_outliers']:
                    print(f"      Upper outliers ({len(stats['upper_outliers'])}): {stats['upper_outliers'][:5]}{'...' if len(stats['upper_outliers']) > 5 else ''}")
                else:
                    print("      No upper outliers found")
            else:
                print(f"\n❌ Failed to calculate statistics for {key}")
        else:
            print(f"\n⚠️  No data available for {key}")


def plot_training_metrics(
    results: Dict[str, Dict[str, Dict[str, List[float]]]],
    batch_sizes: List[int],
    epochs: List[int],
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot training metrics for different batch sizes across epochs.
    
    Args:
        results (Dict): Nested dictionary containing training results.
                       Structure: {batch_size: {'train_metrics': {'accuracy': [...], 'loss': [...]}, 
                                              'val_metrics': {'accuracy': [...], 'loss': [...]}}}
        batch_sizes (List[int]): List of batch sizes to plot.
        epochs (List[int]): List of epoch numbers.
        figsize (Tuple[int, int]): Figure size. Defaults to (15, 10).
        save_path (Optional[str]): Path to save the plot. If None, displays the plot.
        
    Raises:
        ImportError: If matplotlib is not available.
        Exception: If plotting fails.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting training metrics")
    
    try:
        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Train Accuracy
        for batch_size in batch_sizes:
            if (batch_size in results and 
                'train_metrics' in results[batch_size] and 
                'accuracy' in results[batch_size]['train_metrics']):
                axs[0, 0].plot(
                    epochs, 
                    results[batch_size]['train_metrics']['accuracy'], 
                    label=f'Batch Size {batch_size}'
                )
        axs[0, 0].set_title('Train Accuracy')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Accuracy')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)
        
        # Train Loss
        for batch_size in batch_sizes:
            if (batch_size in results and 
                'train_metrics' in results[batch_size] and 
                'loss' in results[batch_size]['train_metrics']):
                axs[0, 1].plot(
                    epochs, 
                    results[batch_size]['train_metrics']['loss'], 
                    label=f'Batch Size {batch_size}'
                )
        axs[0, 1].set_title('Train Loss')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
        
        # Validation Accuracy
        for batch_size in batch_sizes:
            if (batch_size in results and 
                'val_metrics' in results[batch_size] and 
                'accuracy' in results[batch_size]['val_metrics']):
                axs[1, 0].plot(
                    epochs, 
                    results[batch_size]['val_metrics']['accuracy'], 
                    label=f'Batch Size {batch_size}'
                )
        axs[1, 0].set_title('Validation Accuracy')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Accuracy')
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)
        
        # Validation Loss
        for batch_size in batch_sizes:
            if (batch_size in results and 
                'val_metrics' in results[batch_size] and 
                'loss' in results[batch_size]['val_metrics']):
                axs[1, 1].plot(
                    epochs, 
                    results[batch_size]['val_metrics']['loss'], 
                    label=f'Batch Size {batch_size}'
                )
        axs[1, 1].set_title('Validation Loss')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Loss')
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training metrics plot saved to {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"Failed to plot training metrics: {e}")
        raise Exception(f"Failed to plot training metrics: {e}")


def plot_predictions_grid(
    predictions: List[Any],
    grid_size: Tuple[int, int] = (5, 8),
    figsize: Tuple[int, int] = (12, 7),
    image_shape: Tuple[int, int] = (28, 28),
    save_path: Optional[str] = None
) -> None:
    """
    Plot a grid of predictions with their predicted labels.
    
    Args:
        predictions (List[Any]): List of prediction objects with 'data' and 'preds' attributes.
        grid_size (Tuple[int, int]): Grid dimensions (rows, cols). Defaults to (5, 8).
        figsize (Tuple[int, int]): Figure size. Defaults to (12, 7).
        image_shape (Tuple[int, int]): Shape to reshape image data. Defaults to (28, 28).
        save_path (Optional[str]): Path to save the plot. If None, displays the plot.
        
    Raises:
        ImportError: If matplotlib or numpy is not available.
        Exception: If plotting fails.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib and numpy are required for plotting predictions")
    
    try:
        rows, cols = grid_size
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Flatten axes if it's multidimensional
        if rows > 1 or cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            if i < len(predictions):
                # Get image data and reshape
                img_data = np.array(predictions[i].data).reshape(image_shape)
                
                # Get prediction
                prediction = np.argmax(predictions[i].preds)
                
                # Display image
                ax.imshow(img_data, cmap='gray')
                ax.set_title(f"Pred: {prediction}")
                ax.axis('off')
            else:
                # Hide empty subplots
                ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions grid saved to {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"Failed to plot predictions grid: {e}")
        raise Exception(f"Failed to plot predictions grid: {e}")


def create_performance_comparison_boxplot(
    data_dict: Dict[str, List[float]],
    title: str = "Performance Comparison",
    ylabel: str = "Time (seconds)",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Create a boxplot for comparing performance across different configurations.
    
    Args:
        data_dict (Dict[str, List[float]]): Dictionary mapping configuration names to performance data.
        title (str): Plot title. Defaults to "Performance Comparison".
        ylabel (str): Y-axis label. Defaults to "Time (seconds)".
        figsize (Tuple[int, int]): Figure size. Defaults to (12, 8).
        save_path (Optional[str]): Path to save the plot. If None, displays the plot.
        
    Raises:
        ImportError: If matplotlib is not available.
        Exception: If plotting fails.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for creating boxplots")
    
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data and labels
        data_list = []
        labels = []
        
        for config_name, values in data_dict.items():
            if values:  # Only include non-empty datasets
                data_list.append(values)
                labels.append(config_name)
        
        if not data_list:
            logger.warning("No data available for boxplot")
            return
        
        # Create boxplot
        bp = ax.boxplot(data_list, labels=labels, patch_artist=True)
        
        # Customize appearance
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
        
        ax.set_title(title, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('Configuration', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if they're long
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Boxplot saved to {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"Failed to create boxplot: {e}")
        raise Exception(f"Failed to create boxplot: {e}")


def create_performance_bar_chart(
    data_dict: Dict[str, List[float]],
    title: str = "Average Performance Comparison",
    ylabel: str = "Average Time (seconds)",
    figsize: Tuple[int, int] = (10, 6),
    include_error_bars: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    Create a bar chart showing average performance with optional error bars.
    
    Args:
        data_dict (Dict[str, List[float]]): Dictionary mapping configuration names to performance data.
        title (str): Plot title. Defaults to "Average Performance Comparison".
        ylabel (str): Y-axis label. Defaults to "Average Time (seconds)".
        figsize (Tuple[int, int]): Figure size. Defaults to (10, 6).
        include_error_bars (bool): Whether to include standard deviation error bars. Defaults to True.
        save_path (Optional[str]): Path to save the plot. If None, displays the plot.
        
    Raises:
        ImportError: If matplotlib or numpy is not available.
        Exception: If plotting fails.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib and numpy are required for creating bar charts")
    
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate means and standard deviations
        labels = []
        means = []
        stds = []
        
        for config_name, values in data_dict.items():
            if values:  # Only include non-empty datasets
                labels.append(config_name)
                means.append(np.mean(values))
                stds.append(np.std(values) if include_error_bars else 0)
        
        if not labels:
            logger.warning("No data available for bar chart")
            return
        
        # Create bar chart
        x_pos = np.arange(len(labels))
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'plum', 'lightgray']
        
        bars = ax.bar(x_pos, means, 
                     yerr=stds if include_error_bars else None,
                     capsize=5 if include_error_bars else 0,
                     color=[colors[i % len(colors)] for i in range(len(labels))],
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=0.5)
        
        # Add value labels on top of bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (stds[i] if include_error_bars else 0),
                   f'{mean:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_title(title, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Bar chart saved to {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"Failed to create bar chart: {e}")
        raise Exception(f"Failed to create bar chart: {e}")


def save_results_to_json(
    results_dict: Dict[str, Any],
    filepath: str,
    include_statistics: bool = True
) -> bool:
    """
    Save results dictionary to JSON file with optional statistical analysis.
    
    Args:
        results_dict (Dict[str, Any]): Dictionary containing results to save.
        filepath (str): Path where to save the JSON file.
        include_statistics (bool): Whether to include calculated statistics. Defaults to True.
    
    Returns:
        bool: True if save was successful, False otherwise.
    """
    try:
        import json
        from pathlib import Path
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving
        save_data = results_dict.copy()
        
        # Add statistical analysis if requested
        if include_statistics:
            save_data['statistics'] = {}
            for key, values in results_dict.items():
                if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
                    stats = calculate_boxplot_stats(values)
                    if stats:
                        save_data['statistics'][key] = stats
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save results to JSON: {e}")
        return False


def load_results_from_json(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load results dictionary from JSON file.
    
    Args:
        filepath (str): Path to the JSON file to load.
    
    Returns:
        Optional[Dict[str, Any]]: Loaded results dictionary, or None if loading failed.
    """
    try:
        import json
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Results loaded from {filepath}")
        return results
        
    except FileNotFoundError:
        logger.error(f"Results file not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in results file: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load results from JSON: {e}")
        return None