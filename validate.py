#!/usr/bin/env python3
"""
Quick validation script for hyperspectral reconstruction system
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """Test basic functionality of all modules."""
    print("Testing Hyperspectral Reconstruction System...")
    
    try:
        # Test detector response
        from detector_response import DetectorResponseGenerator
        from data_utils import HyperspectralDataLoader
        
        data_loader = HyperspectralDataLoader()
        data, labels = data_loader.generate_synthetic_data(height=20, width=20, num_bands=250)
        print(f"✓ Data module: Generated data shape {data.shape}")
        
        detector_gen = DetectorResponseGenerator(num_detectors=5, wavelength_range=(400, 1000), num_bands=250)
        responses = detector_gen.generate_detector_responses()
        print(f"✓ Detector module: Generated {responses.shape[1]} detectors")
        
        # Test reconstruction
        from spectral_reconstruction import SpectralReconstructor
        
        # Get samples
        samples, _ = data_loader.extract_samples(num_samples=50)
        measurements = detector_gen.simulate_measurements(samples)
        
        reconstructor = SpectralReconstructor(responses, data_loader.wavelengths)
        X, y = reconstructor.prepare_training_data(samples, measurements)
        reconstructor.train(X, y, alpha_selection='manual')
        
        predicted = reconstructor.predict(measurements)
        metrics = reconstructor.evaluate_reconstruction(samples, predicted)
        print(f"✓ Reconstruction module: R² = {metrics['r2']:.3f}")
        
        # Test visualization (basic)
        from visualization import SpectralVisualizer
        
        visualizer = SpectralVisualizer(data_loader.wavelengths)
        print(f"✓ Visualization module: Initialized successfully")
        
        print("\n" + "="*50)
        print("VALIDATION SUCCESSFUL!")
        print("All core modules are working correctly.")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)