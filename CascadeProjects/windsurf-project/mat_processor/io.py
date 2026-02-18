"""
Input/Output operations for .mat files.
"""
import h5py
import numpy as np
from scipy import io
from pathlib import Path
from typing import Dict, Union, Any
import numpy.typing as npt


def load_mat(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a .mat file and return its contents as a dictionary.
    
    Args:
        file_path: Path to the .mat file
        
    Returns:
        Dictionary containing all variables from the .mat file
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If the file is not a valid .mat file
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.suffix == '.mat':
        raise ValueError("File must have a .mat extension")
    
    try:
        # Try loading with scipy.io first (works for most .mat files)
        data = io.loadmat(str(file_path), simplify_cells=True)
        # Remove system variables (those starting with '__')
        return {k: v for k, v in data.items() if not k.startswith('__')}
    except Exception as e:
        # Fall back to h5py for newer MATLAB format
        try:
            with h5py.File(file_path, 'r') as f:
                return {k: np.array(v) for k, v in f.items()}
        except Exception as e2:
            raise ValueError(f"Could not read .mat file: {e2}")


def save_mat(data: Dict[str, Any], file_path: Union[str, Path]):
    """
    Save data to a .mat file.
    
    Args:
        data: Dictionary of variables to save
        file_path: Path where to save the .mat file
    """
    file_path = Path(file_path)
    io.savemat(str(file_path), data)


def explore_mat_structure(data: Dict[str, Any], level: int = 0) -> str:
    """
    Generate a string representation of the .mat file structure.
    
    Args:
        data: Dictionary from load_mat
        level: Current indentation level (used internally for recursion)
        
    Returns:
        String representation of the data structure
    """
    output = []
    indent = '  ' * level
    
    for key, value in data.items():
        if isinstance(value, dict):
            output.append(f"{indent}{key} (dict):")
            output.append(explore_mat_structure(value, level + 1))
        elif isinstance(value, (np.ndarray, list, tuple)):
            if hasattr(value, 'shape'):
                output.append(f"{indent}{key}: {type(value).__name__} with shape {value.shape}")
            else:
                output.append(f"{indent}{key}: {type(value).__name__} with length {len(value)}")
        else:
            output.append(f"{indent}{key}: {type(value).__name__} = {value}")
    
    return '\n'.join(output)
