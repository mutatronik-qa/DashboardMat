# MAT File Processor

A Python package for reading, analyzing, and visualizing data from MATLAB .mat files.

## Features

- Load and explore .mat files with support for both older and newer MATLAB formats
- Extract basic statistics and parameterize data
- Create various visualizations (histograms, time series, heatmaps)
- Convert .mat files to other formats (CSV, JSON)
- Command-line interface for easy usage
- Programmatic API for integration into other Python scripts

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/mat-processor.git
   cd mat-processor
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Usage

### Command Line Interface

#### Get information about a .mat file
```bash
python -m mat_processor.cli info your_file.mat
```

#### Create visualizations from .mat file data
```bash
# Creates plots in the current directory
python -m mat_processor.cli plot your_file.mat

# Save plots to a specific directory
python -m mat_processor.cli plot your_file.mat --output-dir ./output_plots
```

#### Convert .mat file to other formats
```bash
# Convert to CSV (creates a directory with CSV files)
python -m mat_processor.cli convert your_file.mat --format csv

# Convert to JSON
python -m mat_processor.cli convert your_file.mat --format json

# Specify output file/directory
python -m mat_processor.cli convert your_file.mat --format csv --output ./output_data
```

### Python API

```python
from mat_processor.io import load_mat
from mat_processor.analysis import parameterize_data, convert_to_dataframe
from mat_processor.visualization import MATVisualizer

# Load a .mat file
data = load_mat('your_file.mat')

# Get basic statistics about the data
stats = parameterize_data(data)
print(stats)

# Convert data to pandas DataFrames
dataframes = convert_to_dataframe(data)
for name, df in dataframes.items():
    print(f"\n{name}:")
    print(df.head())

# Create visualizations
visualizer = MATVisualizer(data)

# Create a histogram of a 1D array
fig = visualizer.plot_histogram('your_1d_array')

# Create a heatmap of a 2D array
fig = visualizer.plot_heatmap('your_2d_array')

# Show all figures
visualizer.show_all()

# Save a specific figure
visualizer.save_figure('hist_your_1d_array', 'histogram.png')
```

## Examples

### Loading and Exploring Data

```python
from mat_processor.io import load_mat, explore_mat_structure

# Load the .mat file
data = load_mat('example.mat')

# Print the structure of the file
print(explore_mat_structure(data))
```

### Analyzing Data

```python
from mat_processor.analysis import parameterize_data

# Get statistics for all numeric arrays
stats = parameterize_data(data)

# Print statistics for a specific array
print("Statistics for 'signal1':")
for key, value in stats.get('signal1', {}).items():
    print(f"{key}: {value}")
```

### Creating Visualizations

```python
from mat_processor.visualization import MATVisualizer

# Create a visualizer instance
visualizer = MATVisualizer(data)

# Create a histogram
fig_hist = visualizer.plot_histogram('signal1', bins=50, 
                                   title='Distribution of Signal 1',
                                   xlabel='Amplitude')

# Create a time series plot (if you have time and signal data)
fig_ts = visualizer.plot_time_series('time', 'signal1',
                                   title='Signal 1 over Time',
                                   xlabel='Time (s)',
                                   ylabel='Amplitude')

# Show all figures
visualizer.show_all()
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
