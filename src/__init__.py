"""
Hyperspectral Reconstruction Package

A comprehensive toolkit for hyperspectral data reconstruction using
limited detector arrays and ridge regression with cross-validation.
"""

__version__ = "1.0.0"
__author__ = "Hyperspectral Reconstruction Team"

from .detector_response import DetectorResponseGenerator
from .data_utils import HyperspectralDataLoader, SpectralDataManager
from .spectral_reconstruction import SpectralReconstructor
from .visualization import SpectralVisualizer

__all__ = [
    'DetectorResponseGenerator',
    'HyperspectralDataLoader', 
    'SpectralDataManager',
    'SpectralReconstructor',
    'SpectralVisualizer'
]