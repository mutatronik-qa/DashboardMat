"""
Example usage of the mat_processor package.

This script demonstrates how to use the mat_processor package to load, analyze,
and visualize data from a .mat file.
"""
import os
import numpy as np
from pathlib import Path

# Import the mat_processor package
from mat_processor.io import load_mat, explore_mat_structure
from mat_processor.analysis import parameterize_data, convert_to_dataframe
from mat_processor.visualization import MATVisualizer

def main():
    # Create a sample .mat file for demonstration
    # In a real scenario, you would load your own .mat file
    sample_data = {
        'time': np.linspace(0, 10, 1000),
        'signal1': np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000),
        'signal2': np.cos(np.linspace(0, 10, 1000)) * np.exp(-0.1 * np.linspace(0, 10, 1000)),
        'matrix': np.random.rand(20, 30),
        'metadata': {
            'sample_rate': 1000,
            'units': 'volts',
            'experiment_date': '2023-01-01'
        }
    }
    
    # Save the sample data to a .mat file
    import scipy.io
    os.makedirs('data', exist_ok=True)
    mat_file_path = 'data/example_data.mat'
    scipy.io.savemat(mat_file_path, sample_data)
    print(f"Created sample .mat file at: {os.path.abspath(mat_file_path)}")
    
    # Example 1: Load and explore the .mat file
    print("\n--- Example 1: Loading and exploring the .mat file ---")
    data = load_mat(mat_file_path)
    print("\nFile structure:")
    print(explore_mat_structure(data))
    
    # Example 2: Get basic statistics
    print("\n--- Example 2: Basic statistics ---")
    stats = parameterize_data(data)
    for var_name, var_stats in stats.items():
        print(f"\nStatistics for {var_name}:")
        for stat_name, value in var_stats.items():
            print(f"  {stat_name}: {value:.4f}")
    
    # Example 3: Convert to pandas DataFrames
    print("\n--- Example 3: Converting to pandas DataFrames ---")
    dfs = convert_to_dataframe(data)
    for name, df in dfs.items():
        print(f"\nDataFrame for {name} (shape: {df.shape}):")
        print(df.head())
    
    # Example 4: Create visualizations
    print("\n--- Example 4: Creating visualizations ---")
    visualizer = MATVisualizer(data)
    
    # Create output directory for plots
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Create a histogram
    if 'signal1' in data:
        fig_hist = visualizer.plot_histogram(
            'signal1',
            bins=30,
            title='Distribution of Signal 1',
            xlabel='Amplitude',
            ylabel='Frequency',
            figsize=(10, 6)
        )
        hist_path = output_dir / 'signal1_histogram.png'
        visualizer.save_figure('hist_signal1', hist_path)
        print(f"Saved histogram to: {hist_path.absolute()}")
    
    # Create a time series plot
    if 'time' in data and 'signal1' in data:
        fig_ts = visualizer.plot_time_series(
            'time', 'signal1',
            title='Signal 1 over Time',
            xlabel='Time (s)',
            ylabel='Amplitude',
            figsize=(12, 5),
            color='blue',
            linewidth=1.5
        )
        ts_path = output_dir / 'signal1_timeseries.png'
        visualizer.save_figure('ts_time_signal1', ts_path)
        print(f"Saved time series plot to: {ts_path.absolute()}")
    
    # Create a heatmap for 2D data
    if 'matrix' in data and len(data['matrix'].shape) == 2:
        fig_heat = visualizer.plot_heatmap(
            'matrix',
            title='Sample Matrix Heatmap',
            xlabel='X',
            ylabel='Y',
            figsize=(10, 8),
            cmap='viridis',
            annot=False,
            fmt='.2f'
        )
        heat_path = output_dir / 'matrix_heatmap.png'
        visualizer.save_figure('heatmap_matrix', heat_path)
        print(f"Saved heatmap to: {heat_path.absolute()}")
    
    print("\nExample script completed successfully!")
    print(f"Check the 'data' directory for the sample .mat file and 'output' directory for generated plots.")

if __name__ == "__main__":
    main()
