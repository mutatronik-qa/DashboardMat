"""Data loading module for Ninapro database.

This module provides functionality to load and preprocess EMG data from the Ninapro database.
"""

import os
from typing import Dict, Tuple, Optional, Union
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def cargar_datos_ninapro(file_path: str, subject_id: int = 1, dataset: str = 'DB1') -> Dict[str, np.ndarray]:
    """Load EMG data from Ninapro database.

    Args:
        file_path (str): Path to the .mat file containing the Ninapro data.
        subject_id (int, optional): Subject ID. Defaults to 1.
        dataset (str, optional): Dataset name (e.g., 'DB1', 'DB2'). Defaults to 'DB1'.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing the following keys:
            - 'emg': Raw EMG signals (n_samples, n_channels)
            - 'stimulus': Movement labels (n_samples,)
            - 'repetition': Repetition labels (n_samples,)
            - 'restimulus': Restimulus labels (n_samples,)

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        KeyError: If required fields are not found in the .mat file.
        ValueError: If the dataset format is not supported.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with h5py.File(file_path, 'r') as f:
            # Load data based on dataset version
            if dataset == 'DB1':
                emg = np.array(f['emg']).T
                stimulus = np.array(f['stimulus']).flatten()
                repetition = np.array(f['repetition']).flatten()
                restimulus = np.array(f['restimulus']).flatten()
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")

        return {
            'emg': emg,
            'stimulus': stimulus,
            'repetition': repetition,
            'restimulus': restimulus
        }

    except Exception as e:
        raise RuntimeError(f"Error loading data from {file_path}: {str(e)}")


def load_multiple_subjects(data_dir: str, subject_ids: list, dataset: str = 'DB1',
                         verbose: bool = True) -> Dict[int, Dict[str, np.ndarray]]:
    """Load data for multiple subjects.

    Args:
        data_dir (str): Directory containing the data files.
        subject_ids (list): List of subject IDs to load.
        dataset (str, optional): Dataset name. Defaults to 'DB1'.
        verbose (bool, optional): Whether to show progress. Defaults to True.

    Returns:
        Dict[int, Dict[str, np.ndarray]]: Dictionary with subject IDs as keys and
            their corresponding data dictionaries as values.
    """
    data = {}
    iterable = tqdm(subject_ids, desc="Loading subjects") if verbose else subject_ids
    
    for subject_id in iterable:
        try:
            file_path = os.path.join(data_dir, f"S{subject_id:03d}_E1_A1.mat")
            data[subject_id] = cargar_datos_ninapro(file_path, subject_id, dataset)
        except Exception as e:
            print(f"Error loading subject {subject_id}: {str(e)}")
            continue
            
    return data


def split_data(data: Dict[str, np.ndarray], test_size: float = 0.2,
              random_state: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Split data into training and testing sets.

    Args:
        data (Dict[str, np.ndarray]): Dictionary containing 'emg' and 'stimulus' keys.
        test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: Training and testing data splits.
    """
    from sklearn.model_selection import train_test_split
    
    X = data['emg']
    y = data['stimulus']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    train_data = {
        'emg': X_train,
        'stimulus': y_train,
        'repetition': data['repetition'][:len(y_train)],
        'restimulus': data['restimulus'][:len(y_train)]
    }
    
    test_data = {
        'emg': X_test,
        'stimulus': y_test,
        'repetition': data['repetition'][len(y_train):],
        'restimulus': data['restimulus'][len(y_train):]
    }
    
    return train_data, test_data
