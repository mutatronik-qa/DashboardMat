"""Feature extraction module for EMG signals.

This module provides functions to extract time and frequency domain features
from EMG signals.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import signal, fft
from scipy.stats import skew, kurtosis
from tqdm import tqdm


def extract_time_domain_features(emg: np.ndarray, window_size: int = 200, 
                              overlap: float = 0.75, fs: float = 100.0) -> Dict[str, np.ndarray]:
    """Extract time domain features from EMG signals.
    
    Args:
        emg (np.ndarray): Input EMG signal (n_samples, n_channels) or (n_windows, window_size, n_channels).
        window_size (int, optional): Window size in milliseconds. Defaults to 200.
        overlap (float, optional): Overlap between windows (0-1). Defaults to 0.75.
        fs (float, optional): Sampling frequency in Hz. Defaults to 100.0.
        
    Returns:
        Dict[str, np.ndarray]: Dictionary of extracted features.
    """
    # Convert to 3D if input is 2D (n_samples, n_channels)
    if emg.ndim == 2:
        emg = segment_emg(emg, window_size, overlap, fs)
    
    n_windows, window_samples, n_channels = emg.shape
    
    # Initialize feature arrays
    features = {
        'mav': np.zeros((n_windows, n_channels)),
        'rms': np.zeros((n_windows, n_channels)),
        'wl': np.zeros((n_windows, n_channels)),
        'var': np.zeros((n_windows, n_channels)),
        'ssi': np.zeros((n_windows, n_channels)),
        'zc': np.zeros((n_windows, n_channels)),
        'ssc': np.zeros((n_windows, n_channels)),
        'skewness': np.zeros((n_windows, n_channels)),
        'kurt': np.zeros((n_windows, n_channels))
    }
    
    for i in tqdm(range(n_windows), desc="Extracting time domain features"):
        for j in range(n_channels):
            window = emg[i, :, j]
            
            # Mean Absolute Value (MAV)
            features['mav'][i, j] = np.mean(np.abs(window))
            
            # Root Mean Square (RMS)
            features['rms'][i, j] = np.sqrt(np.mean(window ** 2))
            
            # Waveform Length (WL)
            features['wl'][i, j] = np.sum(np.abs(np.diff(window)))
            
            # Variance (VAR)
            features['var'][i, j] = np.var(window, ddof=1)
            
            # Simple Square Integral (SSI)
            features['ssi'][i, j] = np.sum(window ** 2)
            
            # Zero Crossing (ZC)
            features['zc'][i, j] = np.sum(np.abs(np.diff(np.signbit(window))))
            
            # Slope Sign Change (SSC)
            diff = np.diff(window)
            features['ssc'][i, j] = np.sum((diff[1:] * diff[:-1]) < 0)
            
            # Skewness
            features['skewness'][i, j] = skew(window)
            
            # Kurtosis
            features['kurt'][i, j] = kurtosis(window)
    
    return features


def extract_frequency_domain_features(emg: np.ndarray, window_size: int = 200,
                                    overlap: float = 0.75, fs: float = 100.0,
                                    nfft: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Extract frequency domain features from EMG signals.
    
    Args:
        emg (np.ndarray): Input EMG signal (n_samples, n_channels) or (n_windows, window_size, n_channels).
        window_size (int, optional): Window size in milliseconds. Defaults to 200.
        overlap (float, optional): Overlap between windows (0-1). Defaults to 0.75.
        fs (float, optional): Sampling frequency in Hz. Defaults to 100.0.
        nfft (Optional[int], optional): Number of FFT points. If None, uses window size. Defaults to None.
        
    Returns:
        Dict[str, np.ndarray]: Dictionary of extracted features.
    """
    # Convert to 3D if input is 2D (n_samples, n_channels)
    if emg.ndim == 2:
        emg = segment_emg(emg, window_size, overlap, fs)
    
    n_windows, window_samples, n_channels = emg.shape
    
    # Set NFFT to window size if not specified
    if nfft is None:
        nfft = window_samples
    
    # Initialize feature arrays
    features = {
        'mnf': np.zeros((n_windows, n_channels)),
        'mdf': np.zeros((n_windows, n_channels)),
        'pkf': np.zeros((n_windows, n_channels)),
        'psd': np.zeros((n_windows, nfft // 2 + 1, n_channels))
    }
    
    # Frequency vector
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    
    for i in tqdm(range(n_windows), desc="Extracting frequency domain features"):
        for j in range(n_channels):
            window = emg[i, :, j]
            
            # Compute power spectral density using Welch's method
            f, psd = signal.welch(window, fs=fs, nperseg=window_samples, nfft=nfft)
            
            # Store PSD for later use
            features['psd'][i, :, j] = psd
            
            # Mean Frequency (MNF)
            features['mnf'][i, j] = np.sum(f * psd) / (np.sum(psd) + 1e-10)
            
            # Median Frequency (MDF)
            cumsum = np.cumsum(psd)
            features['mdf'][i, j] = f[np.argmax(cumsum >= 0.5 * cumsum[-1])]
            
            # Peak Frequency (PKF)
            features['pkf'][i, j] = f[np.argmax(psd)]
    
    return features


def segment_emg(emg: np.ndarray, window_size: int, overlap: float, 
               fs: float = 100.0) -> np.ndarray:
    """Segment EMG signal into overlapping windows.
    
    Args:
        emg (np.ndarray): Input EMG signal (n_samples, n_channels).
        window_size (int): Window size in milliseconds.
        overlap (float): Overlap between windows (0-1).
        fs (float, optional): Sampling frequency in Hz. Defaults to 100.0.
        
    Returns:
        np.ndarray: Segmented EMG signal (n_windows, window_samples, n_channels).
    """
    n_samples, n_channels = emg.shape
    window_samples = int(window_size * fs / 1000)
    step = int(window_samples * (1 - overlap))
    n_windows = (n_samples - window_samples) // step + 1
    
    # Initialize array for segmented data
    emg_segmented = np.zeros((n_windows, window_samples, n_channels))
    
    # Segment the data
    for i in range(n_windows):
        start = i * step
        end = start + window_samples
        emg_segmented[i] = emg[start:end]
    
    return emg_segmented


def extract_all_features(emg: np.ndarray, window_size: int = 200,
                       overlap: float = 0.75, fs: float = 100.0,
                       nfft: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Extract both time and frequency domain features.
    
    Args:
        emg (np.ndarray): Input EMG signal (n_samples, n_channels).
        window_size (int, optional): Window size in milliseconds. Defaults to 200.
        overlap (float, optional): Overlap between windows (0-1). Defaults to 0.75.
        fs (float, optional): Sampling frequency in Hz. Defaults to 100.0.
        nfft (Optional[int], optional): Number of FFT points. If None, uses window size. Defaults to None.
        
    Returns:
        Dict[str, np.ndarray]: Dictionary containing all extracted features.
    """
    # Segment the EMG signal if needed
    if emg.ndim == 2:
        emg_segmented = segment_emg(emg, window_size, overlap, fs)
    else:
        emg_segmented = emg
    
    # Extract time domain features
    time_features = extract_time_domain_features(emg_segmented, window_size, overlap, fs)
    
    # Extract frequency domain features
    freq_features = extract_frequency_domain_features(emg_segmented, window_size, overlap, fs, nfft)
    
    # Combine all features
    all_features = {**time_features, **freq_features}
    
    # Add segmented EMG data
    all_features['emg'] = emg_segmented
    
    return all_features


def create_feature_matrix(features: Dict[str, np.ndarray], 
                         feature_list: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """Create a feature matrix from the extracted features.
    
    Args:
        features (Dict[str, np.ndarray]): Dictionary of extracted features.
        feature_list (Optional[List[str]], optional): List of feature names to include.
                                                   If None, includes all features. Defaults to None.
        
    Returns:
        Tuple[np.ndarray, List[str]]: Feature matrix (n_samples, n_features) and feature names.
    """
    if feature_list is None:
        # Exclude 'emg' and 'psd' from default features as they're not 2D
        feature_list = [k for k in features.keys() if k not in ['emg', 'psd']]
    
    # Get number of samples and channels from the first feature
    n_samples = features[feature_list[0]].shape[0]
    n_channels = features[feature_list[0]].shape[1] if features[feature_list[0]].ndim > 1 else 1
    
    # Calculate total number of features
    n_features = len(feature_list) * n_channels
    
    # Initialize feature matrix
    X = np.zeros((n_samples, n_features))
    feature_names = []
    
    # Fill the feature matrix
    for i, feat_name in enumerate(feature_list):
        feat_data = features[feat_name]
        
        # Handle both 2D and 3D features
        if feat_data.ndim == 2:  # 2D features (n_samples, n_channels)
            for ch in range(n_channels):
                X[:, i * n_channels + ch] = feat_data[:, ch]
                feature_names.append(f"{feat_name}_ch{ch+1}")
        elif feat_data.ndim == 1:  # 1D features (n_samples,)
            X[:, i] = feat_data
            feature_names.append(feat_name)
    
    return X, feature_names
