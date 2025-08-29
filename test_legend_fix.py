#!/usr/bin/env python3
"""
Quick test to demonstrate improved legend positioning
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add src to path
sys.path.insert(0, 'src')

from detector_response import DetectorResponseGenerator
from data_utils import HyperspectralDataLoader
from visualization import SpectralVisualizer

print("Testing improved legend positioning...")

# Generate test data
detector_gen = DetectorResponseGenerator(num_detectors=15, num_bands=250)
responses = detector_gen.generate_detector_responses()

data_loader = HyperspectralDataLoader()
data, _ = data_loader.generate_synthetic_data(height=20, width=20, num_bands=250)
samples, _ = data_loader.extract_samples(num_samples=10)

# Simulate measurements and reconstruction
measurements = detector_gen.simulate_measurements(samples)
reconstructed = samples + np.random.normal(0, 0.02, samples.shape)  # Add small error

# Create visualizer and test plots
visualizer = SpectralVisualizer(data_loader.wavelengths)

print("✓ Testing detector response plot with improved legend...")
detector_gen.plot_detector_responses(save_path="test_detector_legend.png")

print("✓ Testing spectral comparison plot with improved legend...")
visualizer.plot_spectral_comparison(
    samples, reconstructed, 
    num_samples=3, 
    save_path="test_spectral_legend.png"
)

print("✓ Legend positioning improvements tested successfully!")
print("Check the generated PNG files for improved legend layout:")
print("  - test_detector_legend.png")
print("  - test_spectral_legend.png")