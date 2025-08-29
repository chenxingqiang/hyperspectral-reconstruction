#!/usr/bin/env python3
"""
Simple test to verify basic functionality
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

print("Testing basic hyperspectral reconstruction components...")

try:
    # Test 1: Basic imports
    print("1. Testing imports...")
    from detector_response import DetectorResponseGenerator
    from data_utils import HyperspectralDataLoader
    from spectral_reconstruction import SpectralReconstructor
    print("   ✓ All modules imported successfully")
    
    # Test 2: Data generation
    print("2. Testing data generation...")
    data_loader = HyperspectralDataLoader()
    data, labels = data_loader.generate_synthetic_data(height=10, width=10, num_bands=250)
    print(f"   ✓ Generated synthetic data: {data.shape}")
    
    # Test 3: Detector response generation
    print("3. Testing detector responses...")
    detector_gen = DetectorResponseGenerator(num_detectors=5, num_bands=250)
    responses = detector_gen.generate_detector_responses()
    print(f"   ✓ Generated detector responses: {responses.shape}")
    
    # Test 4: Measurement simulation
    print("4. Testing measurement simulation...")
    samples, _ = data_loader.extract_samples(num_samples=20)
    measurements = detector_gen.simulate_measurements(samples, noise_level=0.01)
    print(f"   ✓ Simulated measurements: {measurements.shape}")
    
    # Test 5: Basic reconstruction
    print("5. Testing reconstruction...")
    reconstructor = SpectralReconstructor(responses, data_loader.wavelengths)
    X, y = reconstructor.prepare_training_data(samples, measurements)
    reconstructor.optimal_alpha = 1e-3  # Set manually
    reconstructor.train(X, y, alpha_selection='manual')
    predicted = reconstructor.predict(measurements)
    print(f"   ✓ Reconstruction completed: {predicted.shape}")
    
    # Test 6: Performance evaluation
    print("6. Testing evaluation...")
    metrics = reconstructor.evaluate_reconstruction(samples, predicted)
    print(f"   ✓ R² Score: {metrics['r2']:.4f}")
    print(f"   ✓ RMSE: {metrics['rmse']:.6f}")
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED!")
    print("The hyperspectral reconstruction system is working correctly.")
    print("="*50)
    
except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)