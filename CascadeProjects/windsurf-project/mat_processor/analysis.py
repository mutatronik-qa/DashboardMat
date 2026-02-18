"""
Data analysis and parameterization utilities for .mat files.
"""
import numpy as np
from typing import Dict, Any, List, Union, Tuple
import numpy.typing as npt
from scipy import stats
import pandas as pd


def get_basic_stats(data: np.ndarray) -> Dict[str, Any]:
    """
    Calculate basic statistics for a numpy array.
    
    Args:
        data: Input data array
        
    Returns:
        Dictionary containing statistics (min, max, mean, std, etc.)
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    return {
        'min': np.nanmin(data),
        'max': np.nanmax(data),
        'mean': np.nanmean(data),
        'median': np.nanmedian(data),
        'std': np.nanstd(data),
        'variance': np.nanvar(data),
        'skewness': stats.skew(data, nan_policy='omit'),
        'kurtosis': stats.kurtosis(data, nan_policy='omit'),
        'q1': np.nanquantile(data, 0.25),
        'q3': np.nanquantile(data, 0.75),
        'iqr': np.nanquantile(data, 0.75) - np.nanquantile(data, 0.25),
        'count': np.count_nonzero(~np.isnan(data)),
        'nan_count': np.count_nonzero(np.isnan(data)),
    }


def parameterize_data(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate basic statistics for all numeric arrays in the data dictionary.
    
    Args:
        data: Dictionary containing data from load_mat()
        
    Returns:
        Nested dictionary with statistics for each numeric array
    """
    results = {}
    
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.number):
                results[key] = get_basic_stats(value)
            elif value.dtype.names:  # Structured array
                for field in value.dtype.names:
                    field_key = f"{key}.{field}"
                    if np.issubdtype(value[field].dtype, np.number):
                        results[field_key] = get_basic_stats(value[field])
    
    return results


def convert_to_dataframe(data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Convert numeric arrays from the .mat file into pandas DataFrames.
    
    Args:
        data: Dictionary from load_mat()
        
    Returns:
        Dictionary with variable names as keys and DataFrames as values
    """
    dfs = {}
    
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            if value.ndim <= 2:  # Convert 1D and 2D arrays to DataFrames
                dfs[key] = pd.DataFrame(value)
            elif value.ndim > 2:  # For higher dimensions, store shape info
                dfs[key] = pd.DataFrame({
                    'shape': [str(value.shape)],
                    'dtype': [str(value.dtype)]
                })
    
    return dfs
