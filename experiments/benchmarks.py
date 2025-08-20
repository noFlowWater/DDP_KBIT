"""
Performance benchmarking utilities for distributed data loading experiments.

This module contains performance measurement and statistical analysis utilities
extracted from the sparkDL_KBIT_gpu_lightning.ipynb notebook. It provides
functions for timing data loading operations, calculating statistical metrics,
and analyzing experimental results.
"""

import time
import numpy as np
import statistics
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
from contextlib import contextmanager
import json
import os
from datetime import datetime


class PerformanceTimer:
    """
    Context manager for measuring execution time with high precision.
    
    Example:
        with PerformanceTimer() as timer:
            # Your code here
            pass
        print(f"Execution time: {timer.elapsed:.3f} seconds")
    """
    
    def __init__(self, description: str = "Operation"):
        """
        Initialize the performance timer.
        
        Args:
            description (str): Description of the operation being timed
        """
        self.description = description
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0
    
    def __enter__(self):
        """Start timing when entering the context."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing when exiting the context."""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        
    def get_elapsed_time(self) -> float:
        """
        Get the elapsed time in seconds.
        
        Returns:
            float: Elapsed time in seconds
        """
        return self.elapsed


@contextmanager
def time_operation(description: str = "Operation", print_result: bool = True):
    """
    Context manager for timing operations with optional printing.
    
    Args:
        description (str): Description of the operation
        print_result (bool): Whether to print the timing result
        
    Yields:
        dict: Dictionary containing timing information
    """
    start_time = time.time()
    timing_info = {'start_time': start_time, 'description': description}
    
    try:
        yield timing_info
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        timing_info.update({
            'end_time': end_time,
            'elapsed_time': elapsed
        })
        
        if print_result:
            print(f"{description}: {elapsed:.3f} seconds")


def calculate_boxplot_stats(data: List[float]) -> Optional[Dict[str, Union[float, int, List[float]]]]:
    """
    Calculate boxplot statistics and other statistical values.
    
    This function computes the five-number summary (boxplot statistics) along with
    additional statistical measures including mean, variance, standard deviation,
    and outlier detection.
    
    Args:
        data (List[float]): List of numerical data points
        
    Returns:
        Optional[Dict]: Dictionary containing statistical measures, None if data is empty
            - count: Number of data points
            - q1: First quartile (25th percentile)
            - median: Second quartile (50th percentile)  
            - q3: Third quartile (75th percentile)
            - iqr: Interquartile range (Q3 - Q1)
            - lower_fence: Lower fence for outlier detection (Q1 - 1.5*IQR)
            - upper_fence: Upper fence for outlier detection (Q3 + 1.5*IQR)
            - lower_outliers: Data points below lower fence
            - upper_outliers: Data points above upper fence
            - mean: Arithmetic mean
            - variance: Sample variance (ddof=1)
            - std_dev: Sample standard deviation (ddof=1)
            - min: Minimum value
            - max: Maximum value
    """
    if not data:
        return None
    
    data_array = np.array(data)
    
    # Boxplot five-number summary
    q1 = np.percentile(data_array, 25)
    q2 = np.percentile(data_array, 50)  # median
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1
    
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    # Outlier detection
    lower_outliers = [x for x in data if x < lower_fence]
    upper_outliers = [x for x in data if x > upper_fence]
    
    # Additional statistical measures
    mean = np.mean(data_array)
    variance = np.var(data_array, ddof=1)  # Sample variance
    std_dev = np.std(data_array, ddof=1)   # Sample standard deviation
    
    return {
        'count': len(data),
        'q1': q1,
        'median': q2,
        'q3': q3,
        'iqr': iqr,
        'lower_fence': lower_fence,
        'upper_fence': upper_fence,
        'lower_outliers': lower_outliers,
        'upper_outliers': upper_outliers,
        'mean': mean,
        'variance': variance,
        'std_dev': std_dev,
        'min': np.min(data_array),
        'max': np.max(data_array)
    }


def print_detailed_stats(stats: Dict[str, Union[float, int, List[float]]], 
                        experiment_name: str = "Experiment") -> None:
    """
    Print detailed statistical analysis in a formatted manner.
    
    Args:
        stats (Dict): Statistical measures from calculate_boxplot_stats
        experiment_name (str): Name of the experiment for display
    """
    if not stats:
        print(f"\n❌ {experiment_name}: No results available")
        return
    
    print(f"\n🔍 {experiment_name.upper()} Statistical Analysis:")
    print(f"   Total data points: {stats['count']}")
    
    print(f"\n📊 Boxplot Five-Number Summary:")
    print(f"      Q1 (25th percentile): {stats['q1']:.3f} seconds")
    print(f"      Q2 (median): {stats['median']:.3f} seconds")
    print(f"      Q3 (75th percentile): {stats['q3']:.3f} seconds")
    print(f"      IQR: {stats['iqr']:.3f} seconds")
    print(f"      Lower Fence: {stats['lower_fence']:.3f} seconds")
    print(f"      Upper Fence: {stats['upper_fence']:.3f} seconds")
    
    print(f"\n📈 Basic Statistical Measures:")
    print(f"      Mean: {stats['mean']:.3f} seconds")
    print(f"      Variance: {stats['variance']:.6f}")
    print(f"      Standard Deviation: {stats['std_dev']:.3f} seconds")
    print(f"      Minimum: {stats['min']:.3f} seconds")
    print(f"      Maximum: {stats['max']:.3f} seconds")
    
    print(f"\n⚠️  Outlier Analysis:")
    if stats['lower_outliers']:
        formatted_lower = [f'{x:.3f}' for x in stats['lower_outliers']]
        print(f"      Lower outliers ({len(stats['lower_outliers'])}): {formatted_lower}")
    else:
        print(f"      Lower outliers: None")
    
    if stats['upper_outliers']:
        formatted_upper = [f'{x:.3f}' for x in stats['upper_outliers']]
        print(f"      Upper outliers ({len(stats['upper_outliers'])}): {formatted_upper}")
    else:
        print(f"      Upper outliers: None")


def analyze_experiment_results(results: Dict[str, List[float]], 
                              print_results: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Analyze results from multiple experiments and generate statistical summaries.
    
    Args:
        results (Dict[str, List[float]]): Dictionary mapping experiment names to timing data
        print_results (bool): Whether to print detailed results
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary containing statistical analysis for each experiment
    """
    analysis_results = {}
    
    if print_results:
        print("\n" + "="*80)
        print("Detailed Statistical Analysis (All Iterations, All Ranks)")
        print("="*80)
    
    for experiment_name, timing_data in results.items():
        if timing_data:
            stats = calculate_boxplot_stats(timing_data)
            analysis_results[experiment_name] = stats
            
            if print_results and stats:
                print_detailed_stats(stats, experiment_name)
        else:
            analysis_results[experiment_name] = None
            if print_results:
                print(f"\n❌ {experiment_name}: No results available")
    
    if print_results:
        print("\n" + "="*80)
    
    return analysis_results


def compare_experiments(results: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Compare performance across different experiments.
    
    Args:
        results (Dict[str, List[float]]): Results from multiple experiments
        
    Returns:
        Dict[str, Any]: Comparison statistics and rankings
    """
    comparison = {
        'rankings': {},
        'performance_ratios': {},
        'summary': {}
    }
    
    # Calculate means for comparison
    experiment_means = {}
    for exp_name, timing_data in results.items():
        if timing_data:
            experiment_means[exp_name] = np.mean(timing_data)
    
    if not experiment_means:
        return comparison
    
    # Sort experiments by performance (lower is better)
    sorted_experiments = sorted(experiment_means.items(), key=lambda x: x[1])
    
    # Create rankings
    for rank, (exp_name, mean_time) in enumerate(sorted_experiments, 1):
        comparison['rankings'][exp_name] = {
            'rank': rank,
            'mean_time': mean_time,
            'rank_suffix': get_ordinal_suffix(rank)
        }
    
    # Calculate performance ratios relative to best performer
    best_time = sorted_experiments[0][1]
    for exp_name, mean_time in experiment_means.items():
        comparison['performance_ratios'][exp_name] = mean_time / best_time
    
    # Generate summary
    comparison['summary'] = {
        'best_performer': sorted_experiments[0][0],
        'worst_performer': sorted_experiments[-1][0],
        'performance_spread': sorted_experiments[-1][1] - sorted_experiments[0][1],
        'relative_improvement': (sorted_experiments[-1][1] - sorted_experiments[0][1]) / sorted_experiments[-1][1] * 100
    }
    
    return comparison


def print_comparison_results(comparison: Dict[str, Any]) -> None:
    """
    Print experiment comparison results in a formatted manner.
    
    Args:
        comparison (Dict[str, Any]): Comparison results from compare_experiments
    """
    if not comparison['rankings']:
        print("No comparison data available.")
        return
    
    print("\n" + "="*80)
    print("EXPERIMENT PERFORMANCE COMPARISON")
    print("="*80)
    
    print("\n🏆 Performance Rankings (Lower time is better):")
    for exp_name, rank_info in comparison['rankings'].items():
        ratio = comparison['performance_ratios'][exp_name]
        print(f"  {rank_info['rank']}{rank_info['rank_suffix']} place: {exp_name.upper()}")
        print(f"    Mean time: {rank_info['mean_time']:.3f} seconds")
        print(f"    Performance ratio: {ratio:.2f}x")
        if ratio > 1:
            slowdown = (ratio - 1) * 100
            print(f"    Slowdown: {slowdown:.1f}% vs best")
        print()
    
    summary = comparison['summary']
    print(f"📊 Summary:")
    print(f"  Best performer: {summary['best_performer'].upper()}")
    print(f"  Worst performer: {summary['worst_performer'].upper()}")
    print(f"  Performance spread: {summary['performance_spread']:.3f} seconds")
    print(f"  Relative improvement: {summary['relative_improvement']:.1f}%")
    
    print("\n" + "="*80)


def get_ordinal_suffix(number: int) -> str:
    """
    Get the ordinal suffix for a number (e.g., 1st, 2nd, 3rd, 4th).
    
    Args:
        number (int): The number to get the suffix for
        
    Returns:
        str: The ordinal suffix
    """
    if 10 <= number % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')
    return suffix


def create_performance_visualization(results: Dict[str, List[float]], 
                                   save_path: Optional[str] = None,
                                   show_plot: bool = True) -> plt.Figure:
    """
    Create visualization of experiment performance results.
    
    Args:
        results (Dict[str, List[float]]): Experiment results
        save_path (Optional[str]): Path to save the plot (if None, won't save)
        show_plot (bool): Whether to display the plot
        
    Returns:
        plt.Figure: The matplotlib figure object
    """
    # Filter out empty results
    filtered_results = {k: v for k, v in results.items() if v}
    
    if not filtered_results:
        print("No data available for visualization.")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot
    data_for_boxplot = list(filtered_results.values())
    labels = list(filtered_results.keys())
    
    ax1.boxplot(data_for_boxplot, labels=labels)
    ax1.set_title('Performance Distribution by Experiment')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Bar plot with means and error bars
    means = [np.mean(data) for data in data_for_boxplot]
    stds = [np.std(data, ddof=1) for data in data_for_boxplot]
    
    bars = ax2.bar(labels, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    ax2.set_title('Mean Performance with Error Bars')
    ax2.set_ylabel('Mean Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean_val:.3f}s',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def save_results_to_file(results: Dict[str, List[float]], 
                        analysis: Dict[str, Dict[str, Any]],
                        filepath: str,
                        include_metadata: bool = True) -> None:
    """
    Save experiment results and analysis to a JSON file.
    
    Args:
        results (Dict[str, List[float]]): Raw experiment results
        analysis (Dict[str, Dict[str, Any]]): Statistical analysis results
        filepath (str): Path to save the file
        include_metadata (bool): Whether to include metadata
    """
    output_data = {
        'raw_results': results,
        'statistical_analysis': {}
    }
    
    # Convert numpy types to Python native types for JSON serialization
    for exp_name, stats in analysis.items():
        if stats:
            output_data['statistical_analysis'][exp_name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else 
                   [float(x) for x in v] if isinstance(v, list) and v and isinstance(v[0], (np.floating, float)) else
                   float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in stats.items()
            }
        else:
            output_data['statistical_analysis'][exp_name] = None
    
    if include_metadata:
        output_data['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(results),
            'total_data_points': sum(len(v) for v in results.values() if v),
            'generated_by': 'DDP_KBIT.experiments.benchmarks'
        }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {filepath}")


def load_results_from_file(filepath: str) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, Any]]]:
    """
    Load experiment results and analysis from a JSON file.
    
    Args:
        filepath (str): Path to the results file
        
    Returns:
        Tuple[Dict, Dict]: Raw results and analysis data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data['raw_results'], data['statistical_analysis']


# Utility functions for data loading timing
def time_dataloader_iteration(dataloader, max_batches: Optional[int] = None) -> Dict[str, float]:
    """
    Time a complete iteration through a dataloader.
    
    Args:
        dataloader: PyTorch DataLoader object
        max_batches (Optional[int]): Maximum number of batches to process
        
    Returns:
        Dict[str, float]: Timing statistics
    """
    batch_times = []
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        batch_start = time.time()
        # Simulate some processing (just accessing the data)
        _ = batch
        batch_end = time.time()
        batch_times.append(batch_end - batch_start)
        
        if max_batches and i >= max_batches - 1:
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return {
        'total_time': total_time,
        'mean_batch_time': np.mean(batch_times) if batch_times else 0.0,
        'median_batch_time': np.median(batch_times) if batch_times else 0.0,
        'min_batch_time': np.min(batch_times) if batch_times else 0.0,
        'max_batch_time': np.max(batch_times) if batch_times else 0.0,
        'num_batches': len(batch_times)
    }


if __name__ == "__main__":
    # Example usage
    print("Benchmarking utilities example...")