"""Signal preprocessing module for EMG data.

This module provides functions for preprocessing EMG signals, including filtering,
normalization, and segmentation.
"""

from typing import Tuple, Optional, Union, Dict, Any
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
from tqdm import tqdm


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Design a bandpass Butterworth filter.

    Args:
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int, optional): Filter order. Defaults to 4.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float,
                         order: int = 4) -> np.ndarray:
    """Apply a bandpass filter to the input data.

    Args:
        data (np.ndarray): Input signal (n_samples, n_channels).
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int, optional): Filter order. Defaults to 4.

    Returns:
        np.ndarray: Filtered signal (n_samples, n_channels).
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data, axis=0)


def notch_filter(data: np.ndarray, notch_freq: float, fs: float, Q: float = 30.0) -> np.ndarray:
    """Apply a notch filter to remove power line interference.

    Args:
        data (np.ndarray): Input signal (n_samples, n_channels).
        notch_freq (float): Frequency to notch out (e.g., 50 or 60 Hz).
        fs (float): Sampling frequency in Hz.
        Q (float, optional): Quality factor. Defaults to 30.0.

    Returns:
        np.ndarray: Filtered signal (n_samples, n_channels).
    """
    nyq = 0.5 * fs
    freq = notch_freq / nyq
    b, a = signal.iirnotch(freq, Q)
    return filtfilt(b, a, data, axis=0)


def normalize_emg(data: np.ndarray, method: str = 'standard',
                params: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Normalize EMG data.

    Args:
        data (np.ndarray): Input signal (n_samples, n_channels).
        method (str, optional): Normalization method. Options: 'standard', 'minmax', 'mvc'.
                              Defaults to 'standard'.
        params (Optional[Dict[str, Any]], optional): Normalization parameters. If None,
                                                   they will be calculated from data.
                                                   Defaults to None.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: Normalized data and parameters used.
    """
    if params is None:
        params = {}
    
    if method == 'standard':
        if 'mean' not in params or 'std' not in params:
            params['mean'] = np.mean(data, axis=0, keepdims=True)
            params['std'] = np.std(data, axis=0, keepdims=True) + 1e-8
        normalized = (data - params['mean']) / params['std']
    
    elif method == 'minmax':
        if 'min' not in params or 'max' not in params:
            params['min'] = np.min(data, axis=0, keepdims=True)
            params['max'] = np.max(data, axis=0, keepdims=True)
        normalized = (data - params['min']) / (params['max'] - params['min'] + 1e-8)
    
    elif method == 'mvc':
        if 'mvc' not in params:
            raise ValueError("MVC values must be provided in params['mvc']")
        normalized = data / (params['mvc'] + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def preprocess_pipeline(data: Dict[str, np.ndarray], fs: float = 100.0,
                      lowcut: float = 20.0, highcut: float = 450.0,
                      notch_freq: Optional[float] = 50.0,
                      normalize: bool = True,
                      norm_method: str = 'standard',
                      norm_params: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
    """Complete preprocessing pipeline for EMG data.

    Args:
        data (Dict[str, np.ndarray]): Dictionary containing 'emg' key with raw data.
        fs (float, optional): Sampling frequency in Hz. Defaults to 100.0.
        lowcut (float, optional): Low cutoff frequency in Hz. Defaults to 20.0.
        highcut (float, optional): High cutoff frequency in Hz. Defaults to 450.0.
        notch_freq (Optional[float], optional): Notch filter frequency. If None, not applied.
                                              Defaults to 50.0.
        normalize (bool, optional): Whether to normalize the data. Defaults to True.
        norm_method (str, optional): Normalization method. Defaults to 'standard'.
        norm_params (Optional[Dict[str, Any]], optional): Normalization parameters.
                                                       Defaults to None.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing preprocessed data.
    """
    processed_data = data.copy()
    emg = data['emg']
    
    # Apply bandpass filter (20-450 Hz)
    emg_filtered = apply_bandpass_filter(emg, lowcut, highcut, fs)
    
    # Apply notch filter if specified (e.g., for 50/60 Hz power line noise)
    if notch_freq is not None:
        emg_filtered = notch_filter(emg_filtered, notch_freq, fs)
    
    # Normalize the data if requested
    if normalize:
        emg_filtered, norm_params = normalize_emg(emg_filtered, method=norm_method, params=norm_params)
        processed_data['norm_params'] = norm_params
    
    processed_data['emg'] = emg_filtered
    return processed_data


def segment_signal(data: Dict[str, np.ndarray], window_size: int, overlap: float,
                  fs: float = 100.0) -> Dict[str, np.ndarray]:
    """Segment EMG signals into windows.

    Args:
        data (Dict[str, np.ndarray]): Dictionary containing 'emg' and 'stimulus'.
        window_size (int): Window size in milliseconds.
        overlap (float): Overlap between windows (0-1).
        fs (float, optional): Sampling frequency in Hz. Defaults to 100.0.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing segmented data.
    """
    emg = data['emg']
    stimulus = data['stimulus']
    
    # Convert window size from ms to samples
    window_samples = int(window_size * fs / 1000)
    step = int(window_samples * (1 - overlap))
    
    n_samples = emg.shape[0]
    n_channels = emg.shape[1]
    n_windows = (n_samples - window_samples) // step + 1
    
    # Initialize arrays for segmented data
    emg_segmented = np.zeros((n_windows, window_samples, n_channels))
    stimulus_segmented = np.zeros(n_windows, dtype=int)
    
    # Segment the data
    for i in range(n_windows):
        start = i * step
        end = start + window_samples
        emg_segmented[i] = emg[start:end]
        # Use majority vote for the label in the window
        stimulus_segmented[i] = np.argmax(np.bincount(stimulus[start:end]))
    
    return {
        'emg': emg_segmented,
        'stimulus': stimulus_segmented,
        'window_size': window_size,
        'overlap': overlap
    }
