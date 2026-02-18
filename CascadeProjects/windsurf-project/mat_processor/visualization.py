"""
Visualization utilities for .mat file data.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, Any, Optional, List, Union
import numpy.typing as npt
from pathlib import Path

# Set a nice style for plots
plt.style.use('seaborn')
sns.set_palette('colorblind')

class MATVisualizer:
    """Class for creating visualizations from .mat file data."""
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize with data from load_mat().
        
        Args:
            data: Dictionary containing the loaded .mat data
        """
        self.data = data
        self.figures = {}
    
    def plot_histogram(self, key: str, bins: int = 30, 
                      title: Optional[str] = None,
                      xlabel: Optional[str] = None,
                      ylabel: str = 'Frequency',
                      figsize: tuple = (10, 6),
                      **kwargs) -> plt.Figure:
        """
        Create a histogram of a numeric array.
        
        Args:
            key: Key of the data to plot
            bins: Number of bins for the histogram
            title: Plot title (defaults to key if None)
            xlabel: X-axis label (defaults to key if None)
            ylabel: Y-axis label
            figsize: Figure size (width, height)
            **kwargs: Additional arguments to pass to plt.hist()
            
        Returns:
            Matplotlib Figure object
        """
        if key not in self.data:
            raise KeyError(f"Key '{key}' not found in data")
            
        data = self.data[key]
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError(f"Data for key '{key}' is not numeric")
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(data.ravel(), bins=bins, edgecolor='black', **kwargs)
        
        title = title or f'Histogram of {key}'
        xlabel = xlabel or key
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        self.figures[f"hist_{key}"] = fig
        return fig
    
    def plot_time_series(self, x_key: str, y_key: str,
                        title: Optional[str] = None,
                        xlabel: Optional[str] = None,
                        ylabel: Optional[str] = None,
                        figsize: tuple = (12, 6),
                        **kwargs) -> plt.Figure:
        """
        Plot time series data.
        
        Args:
            x_key: Key for x-axis data
            y_key: Key for y-axis data
            title: Plot title
            xlabel: X-axis label (defaults to x_key if None)
            ylabel: Y-axis label (defaults to y_key if None)
            figsize: Figure size (width, height)
            **kwargs: Additional arguments to pass to plt.plot()
            
        Returns:
            Matplotlib Figure object
        """
        for k in [x_key, y_key]:
            if k not in self.data:
                raise KeyError(f"Key '{k}' not found in data")
        
        x = self.data[x_key]
        y = self.data[y_key]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, **kwargs)
        
        title = title or f"{y_key} vs {x_key}"
        xlabel = xlabel or x_key
        ylabel = ylabel or y_key
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        self.figures[f"ts_{x_key}_{y_key}"] = fig
        return fig
    
    def plot_heatmap(self, key: str, 
                    title: Optional[str] = None,
                    xlabel: str = 'X',
                    ylabel: str = 'Y',
                    figsize: tuple = (10, 8),
                    **kwargs) -> plt.Figure:
        """
        Create a heatmap of 2D array data.
        
        Args:
            key: Key of the 2D array to plot
            title: Plot title (defaults to key if None)
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size (width, height)
            **kwargs: Additional arguments to pass to sns.heatmap()
            
        Returns:
            Matplotlib Figure object
        """
        if key not in self.data:
            raise KeyError(f"Key '{key}' not found in data")
            
        data = self.data[key]
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError(f"Data for key '{key}' must be a 2D array")
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(data, ax=ax, **kwargs)
        
        title = title or f'Heatmap of {key}'
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        self.figures[f"heatmap_{key}"] = fig
        return fig
    
    def save_figure(self, fig_key: str, file_path: Union[str, Path], 
                   dpi: int = 300, **kwargs):
        """
        Save a figure to a file.
        
        Args:
            fig_key: Key of the figure in self.figures
            file_path: Path to save the figure to
            dpi: DPI for the output figure
            **kwargs: Additional arguments to pass to savefig()
        """
        if fig_key not in self.figures:
            raise KeyError(f"Figure with key '{fig_key}' not found")
            
        self.figures[fig_key].savefig(file_path, dpi=dpi, bbox_inches='tight', 
                                     **kwargs)
    
    def show_all(self):
        """Display all created figures."""
        for fig in self.figures.values():
            plt.figure(fig.number)
            plt.show()
