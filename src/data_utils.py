"""
Data Utilities Module for Hyperspectral Data Processing

This module provides utilities for loading, preprocessing, and managing
hyperspectral data, including support for the Xiong'an dataset and synthetic data generation.
"""

import numpy as np
import scipy.io as sio
from typing import Tuple, Optional, Union, Dict, Any
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import h5py


class HyperspectralDataLoader:
    """Loader for hyperspectral data from various formats."""
    
    def __init__(self, wavelength_range: Tuple[int, int] = (400, 1000)):
        """
        Initialize the data loader.
        
        Args:
            wavelength_range: Target wavelength range in nm (default: (400, 1000))
        """
        self.wavelength_range = wavelength_range
        self.data = None
        self.wavelengths = None
        self.ground_truth = None
        self.metadata = {}
        
    def load_xiong_an_data(self, data_path: str, gt_path: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load Xiong'an hyperspectral dataset.
        
        Args:
            data_path: Path to the main hyperspectral data file
            gt_path: Optional path to ground truth file
            
        Returns:
            Tuple of (hyperspectral_data, ground_truth)
        """
        try:
            # Load main data
            if data_path.endswith('.mat'):
                try:
                    # First try SciPy loader (non-HDF5 .mat files)
                    data_dict = sio.loadmat(data_path)
                    # Common variable names in Xiong'an dataset
                    possible_keys = ['xiongan', 'data', 'hyperspectral', 'image']
                    data_key = None
                    for key in possible_keys:
                        if key in data_dict:
                            data_key = key
                            break
                    if data_key is None:
                        # Use the largest array that's not metadata
                        data_key = max([k for k in data_dict.keys() if not k.startswith('__')],
                                       key=lambda k: data_dict[k].size)
                    self.data = data_dict[data_key]
                    print(f"Loaded data with key '{data_key}' and shape: {self.data.shape}")
                except NotImplementedError:
                    # MATLAB v7.3 (HDF5) files need h5py
                    print(f"  → Using h5py for MATLAB v7.3 file: {data_path}")
                    with h5py.File(data_path, 'r') as f:
                        print(f"  → HDF5 file opened, top-level keys: {list(f.keys())}")
                        # Collect datasets and pick the largest likely hyperspectral cube
                        def _collect_datasets(h5obj, prefix=''):
                            items = []
                            for k, v in h5obj.items():
                                path = f"{prefix}/{k}" if prefix else k
                                if isinstance(v, h5py.Dataset):
                                    items.append((path, v))
                                elif isinstance(v, h5py.Group):
                                    items.extend(_collect_datasets(v, path))
                            return items
                        datasets = _collect_datasets(f)
                        if not datasets:
                            # Try common dataset names for Xiong'an data
                            common_names = ['XiongAn', 'xiongan', 'data', 'hyperspectral', 'image', 'hsi']
                            found_dataset = None
                            for name in common_names:
                                if name in f:
                                    found_dataset = f[name]
                                    break
                            if found_dataset is None:
                                available_keys = list(f.keys())
                                raise RuntimeError(f"No datasets found inside HDF5 .mat file. Available keys: {available_keys}")
                            path, dset = name, found_dataset
                        else:
                            # Choose dataset with maximum number of elements
                            path, dset = max(datasets, key=lambda item: np.prod(item[1].shape))
                        arr = dset[()]
                        # Ensure dtype is float
                        if not np.issubdtype(arr.dtype, np.floating):
                            arr = arr.astype(np.float32)
                        # Heuristics to move spectral bands axis to the last position if needed
                        if arr.ndim == 3:
                            h, w, c = arr.shape
                            # If bands likely in first/second axis (250 typical), move to last
                            if h == 250 and c != 250:
                                arr = np.moveaxis(arr, 0, -1)
                            elif w == 250 and c != 250:
                                arr = np.moveaxis(arr, 1, -1)
                        elif arr.ndim > 3:
                            # Flatten leading dims except bands if we can identify bands=250
                            band_axis = None
                            for ax, dim in enumerate(arr.shape):
                                if dim == 250:
                                    band_axis = ax
                                    break
                            if band_axis is not None:
                                # Move bands to last, collapse the rest to 2D spatial
                                arr = np.moveaxis(arr, band_axis, -1)
                                spatial = int(np.prod(arr.shape[:-1]))
                                arr = arr.reshape(spatial, 1, arr.shape[-1])
                        if arr.ndim != 3:
                            raise RuntimeError(f"Unexpected dataset shape for hyperspectral cube: {arr.shape}")
                        self.data = np.array(arr)
                        print(f"Loaded HDF5 dataset at '{path}' with shape: {self.data.shape}")
                
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            # Load ground truth if provided
            if gt_path and os.path.exists(gt_path):
                try:
                    gt_dict = sio.loadmat(gt_path)
                    gt_key = [k for k in gt_dict.keys() if not k.startswith('__')][0]
                    self.ground_truth = gt_dict[gt_key]
                except NotImplementedError:
                    with h5py.File(gt_path, 'r') as f:
                        # pick first dataset
                        def _first_dataset(h5obj):
                            for k, v in h5obj.items():
                                if isinstance(v, h5py.Dataset):
                                    return v[()]
                                if isinstance(v, h5py.Group):
                                    found = _first_dataset(v)
                                    if found is not None:
                                        return found
                            return None
                        gt_arr = _first_dataset(f)
                        if gt_arr is None:
                            raise RuntimeError("No dataset found in ground truth .mat file")
                        self.ground_truth = np.array(gt_arr)
                print(f"Loaded ground truth with shape: {self.ground_truth.shape}")
            
            # Generate wavelengths for Xiong'an dataset (400-1000nm, 250 bands)
            if self.data.shape[-1] == 250:
                self.wavelengths = np.linspace(400, 1000, 250)
            else:
                self.wavelengths = np.linspace(self.wavelength_range[0], 
                                             self.wavelength_range[1], 
                                             self.data.shape[-1])
            
            # Store metadata
            self.metadata = {
                'original_shape': self.data.shape,
                'num_bands': self.data.shape[-1],
                'wavelength_range': (self.wavelengths[0], self.wavelengths[-1]),
                'data_source': 'xiong_an'
            }
            
            return self.data, self.ground_truth
            
        except Exception as e:
            print(f"Error loading Xiong'an data: {e}")
            raise
    
    def generate_synthetic_data(self, 
                              height: int = 100, 
                              width: int = 100, 
                              num_bands: int = 250,
                              num_classes: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic hyperspectral data for testing.
        
        Args:
            height: Image height in pixels
            width: Image width in pixels
            num_bands: Number of spectral bands
            num_classes: Number of material classes
            
        Returns:
            Tuple of (hyperspectral_data, class_labels)
        """
        # Generate wavelengths
        self.wavelengths = np.linspace(self.wavelength_range[0], 
                                     self.wavelength_range[1], 
                                     num_bands)
        
        # Create class labels
        np.random.seed(42)  # For reproducibility
        labels = np.random.randint(0, num_classes, (height, width))
        
        # Generate synthetic spectra for each class
        class_spectra = []
        for class_id in range(num_classes):
            # Create characteristic spectral curves
            base_spectrum = np.exp(-((self.wavelengths - (500 + class_id * 100)) / 80) ** 2)
            # Add some vegetation-like features
            if class_id < 3:  # Vegetation classes
                base_spectrum += 0.3 * np.exp(-((self.wavelengths - 750) / 50) ** 2)
            # Add water absorption features
            water_absorption = 1 - 0.5 * np.exp(-((self.wavelengths - 900) / 30) ** 2)
            base_spectrum *= water_absorption
            
            class_spectra.append(base_spectrum)
        
        # Generate hyperspectral image
        hyperspectral_data = np.zeros((height, width, num_bands))
        for i in range(height):
            for j in range(width):
                class_id = labels[i, j]
                # Add some noise and variation
                noise = np.random.normal(0, 0.05, num_bands)
                variation = np.random.normal(1, 0.1)
                hyperspectral_data[i, j, :] = class_spectra[class_id] * variation + noise
        
        # Normalize to [0, 1] range
        hyperspectral_data = np.clip(hyperspectral_data, 0, None)
        hyperspectral_data = hyperspectral_data / np.max(hyperspectral_data)
        
        self.data = hyperspectral_data
        self.ground_truth = labels
        
        # Store metadata
        self.metadata = {
            'original_shape': self.data.shape,
            'num_bands': num_bands,
            'wavelength_range': (self.wavelengths[0], self.wavelengths[-1]),
            'data_source': 'synthetic',
            'num_classes': num_classes
        }
        
        print(f"Generated synthetic data with shape: {self.data.shape}")
        print(f"Generated labels with shape: {labels.shape}")
        
        return self.data, labels
    
    def preprocess_data(self, 
                       normalization: str = 'minmax',
                       remove_bad_bands: bool = True,
                       noise_threshold: float = 0.01) -> np.ndarray:
        """
        Preprocess hyperspectral data.
        
        Args:
            normalization: Type of normalization ('minmax', 'standard', 'none')
            remove_bad_bands: Whether to remove noisy bands
            noise_threshold: Threshold for noise detection
            
        Returns:
            Preprocessed hyperspectral data
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        data = self.data.copy()
        original_shape = data.shape
        
        # Reshape for processing
        data_reshaped = data.reshape(-1, data.shape[-1])
        
        # Remove bad bands if requested
        if remove_bad_bands:
            band_std = np.std(data_reshaped, axis=0)
            good_bands = band_std > noise_threshold
            data_reshaped = data_reshaped[:, good_bands]
            self.wavelengths = self.wavelengths[good_bands]
            print(f"Removed {np.sum(~good_bands)} bad bands")
        
        # Normalization
        if normalization == 'minmax':
            scaler = MinMaxScaler()
            data_reshaped = scaler.fit_transform(data_reshaped)
        elif normalization == 'standard':
            scaler = StandardScaler()
            data_reshaped = scaler.fit_transform(data_reshaped)
        elif normalization != 'none':
            warnings.warn(f"Unknown normalization method: {normalization}")
        
        # Reshape back
        new_shape = original_shape[:-1] + (data_reshaped.shape[-1],)
        self.data = data_reshaped.reshape(new_shape)
        
        return self.data

    def resample_spectra_to(self, spectra: np.ndarray, target_wavelengths: np.ndarray) -> np.ndarray:
        """
        Resample spectra along the wavelength axis to target wavelengths using linear interpolation.

        Args:
            spectra: Array with shape (num_samples, num_bands)
            target_wavelengths: 1D array of target wavelengths

        Returns:
            Resampled spectra with shape (num_samples, len(target_wavelengths))
        """
        if self.wavelengths is None:
            raise ValueError("Source wavelengths are not set.")
        source_wavelengths = self.wavelengths

        # Ensure increasing order for interpolation
        if source_wavelengths[0] > source_wavelengths[-1]:
            source_wavelengths = source_wavelengths[::-1]
            spectra = spectra[:, ::-1]

        target_wavelengths = np.asarray(target_wavelengths)
        resampled = np.empty((spectra.shape[0], target_wavelengths.shape[0]), dtype=np.float32)
        for i in range(spectra.shape[0]):
            resampled[i] = np.interp(target_wavelengths, source_wavelengths, spectra[i])
        return resampled
    
    def extract_samples(self, 
                       num_samples: int = 1000, 
                       sampling_method: str = 'random') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract samples from the hyperspectral data.
        
        Args:
            num_samples: Number of samples to extract
            sampling_method: Sampling method ('random', 'uniform')
            
        Returns:
            Tuple of (sample_spectra, sample_labels)
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        height, width = self.data.shape[:2]
        total_pixels = height * width
        
        if num_samples > total_pixels:
            num_samples = total_pixels
            warnings.warn(f"Requested samples exceed available pixels. Using all {total_pixels} pixels.")
        
        # Reshape data
        data_reshaped = self.data.reshape(-1, self.data.shape[-1])
        
        if sampling_method == 'random':
            indices = np.random.choice(total_pixels, num_samples, replace=False)
        elif sampling_method == 'uniform':
            step = total_pixels // num_samples
            indices = np.arange(0, total_pixels, step)[:num_samples]
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        sample_spectra = data_reshaped[indices]
        
        sample_labels = None
        if self.ground_truth is not None:
            labels_reshaped = self.ground_truth.reshape(-1)
            sample_labels = labels_reshaped[indices]
        
        return sample_spectra, sample_labels
    
    def get_spectral_subset(self, wavelength_indices: np.ndarray) -> np.ndarray:
        """
        Extract a subset of spectral bands.
        
        Args:
            wavelength_indices: Indices of wavelengths to extract
            
        Returns:
            Subset of hyperspectral data
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        return self.data[..., wavelength_indices]
    
    def save_processed_data(self, file_path: str) -> None:
        """
        Save processed data to file.
        
        Args:
            file_path: Path to save the data
        """
        if self.data is None:
            raise ValueError("No data to save.")
        
        save_dict = {
            'data': self.data,
            'wavelengths': self.wavelengths,
            'metadata': self.metadata
        }
        
        if self.ground_truth is not None:
            save_dict['ground_truth'] = self.ground_truth
        
        np.savez_compressed(file_path, **save_dict)
        print(f"Data saved to {file_path}")


class SpectralDataManager:
    """Manager for handling multiple hyperspectral datasets."""
    
    def __init__(self):
        self.datasets = {}
        
    def add_dataset(self, name: str, loader: HyperspectralDataLoader) -> None:
        """Add a dataset to the manager."""
        self.datasets[name] = loader
        
    def get_dataset(self, name: str) -> HyperspectralDataLoader:
        """Get a dataset by name."""
        if name not in self.datasets:
            raise KeyError(f"Dataset '{name}' not found.")
        return self.datasets[name]
    
    def list_datasets(self) -> list:
        """List all available datasets."""
        return list(self.datasets.keys())
    
    def get_combined_samples(self, 
                           dataset_names: list, 
                           samples_per_dataset: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine samples from multiple datasets.
        
        Args:
            dataset_names: List of dataset names to combine
            samples_per_dataset: Number of samples per dataset
            
        Returns:
            Tuple of (combined_spectra, dataset_labels)
        """
        all_spectra = []
        all_labels = []
        
        for i, name in enumerate(dataset_names):
            loader = self.get_dataset(name)
            spectra, _ = loader.extract_samples(samples_per_dataset)
            all_spectra.append(spectra)
            all_labels.extend([i] * len(spectra))
        
        combined_spectra = np.vstack(all_spectra)
        combined_labels = np.array(all_labels)
        
        return combined_spectra, combined_labels


if __name__ == "__main__":
    # Example usage
    print("Hyperspectral Data Loader Test")
    print("=" * 40)
    
    # Test synthetic data generation
    loader = HyperspectralDataLoader()
    data, labels = loader.generate_synthetic_data(height=50, width=50, num_bands=250)
    
    print(f"Synthetic data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Wavelength range: {loader.wavelengths[0]:.1f} - {loader.wavelengths[-1]:.1f} nm")
    
    # Test preprocessing
    processed_data = loader.preprocess_data(normalization='minmax')
    print(f"Processed data range: {np.min(processed_data):.3f} - {np.max(processed_data):.3f}")
    
    # Test sample extraction
    samples, sample_labels = loader.extract_samples(num_samples=100)
    print(f"Extracted {len(samples)} samples")