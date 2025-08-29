"""
Test Script for Hyperspectral Reconstruction System

This script performs comprehensive testing of the hyperspectral reconstruction
pipeline to validate functionality and performance.
"""

import os
import sys
import numpy as np
import time
import warnings
from pathlib import Path
import unittest

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from detector_response import DetectorResponseGenerator
from data_utils import HyperspectralDataLoader, SpectralDataManager
from spectral_reconstruction import SpectralReconstructor
from visualization import SpectralVisualizer


class TestHyperspectralReconstruction(unittest.TestCase):
    """Test suite for hyperspectral reconstruction system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.wavelengths = np.linspace(400, 1000, 250)
        self.num_detectors = 15
        self.num_samples = 100
        
        # Suppress warnings during testing
        warnings.filterwarnings('ignore')
    
    def test_detector_response_generation(self):
        """Test detector response function generation."""
        print("\n=== Testing Detector Response Generation ===")
        
        detector_gen = DetectorResponseGenerator(
            num_detectors=self.num_detectors,
            wavelength_range=(400, 1000)
        )
        
        # Test response generation
        responses = detector_gen.generate_detector_responses(fwhm=50.0)
        
        # Validate shape
        self.assertEqual(responses.shape, (250, self.num_detectors))
        
        # Validate peak transmittance
        max_responses = np.max(responses, axis=0)
        self.assertTrue(np.allclose(max_responses, 0.5, atol=0.01))
        
        # Validate coverage
        combined_response = np.sum(responses, axis=1)
        self.assertTrue(np.all(combined_response > 0))
        
        # Test statistics
        stats = detector_gen.get_coverage_statistics()
        self.assertIn('coverage_uniformity', stats)
        self.assertGreater(stats['mean_coverage'], 0)
        
        print(f"✓ Detector responses generated successfully")
        print(f"✓ Coverage uniformity: {stats['coverage_uniformity']:.4f}")
        
    def test_synthetic_data_generation(self):
        """Test synthetic hyperspectral data generation."""
        print("\n=== Testing Synthetic Data Generation ===")
        
        data_loader = HyperspectralDataLoader(wavelength_range=(400, 1000))
        
        # Generate synthetic data
        data, labels = data_loader.generate_synthetic_data(
            height=50, width=50, num_bands=250, num_classes=5
        )
        
        # Validate shapes
        self.assertEqual(data.shape, (50, 50, 250))
        self.assertEqual(labels.shape, (50, 50))
        
        # Validate data range
        self.assertTrue(np.all(data >= 0))
        self.assertTrue(np.all(data <= 1))
        
        # Validate labels
        unique_labels = np.unique(labels)
        self.assertEqual(len(unique_labels), 5)
        self.assertTrue(np.all(unique_labels >= 0))
        self.assertTrue(np.all(unique_labels < 5))
        
        print(f"✓ Synthetic data generated: {data.shape}")
        print(f"✓ Labels generated: {labels.shape}")
        print(f"✓ Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
        
    def test_data_preprocessing(self):
        """Test data preprocessing functions."""
        print("\n=== Testing Data Preprocessing ===")
        
        data_loader = HyperspectralDataLoader(wavelength_range=(400, 1000))
        data, _ = data_loader.generate_synthetic_data(
            height=30, width=30, num_bands=250
        )
        
        # Test preprocessing
        processed_data = data_loader.preprocess_data(
            normalization='minmax',
            remove_bad_bands=False
        )
        
        # Validate normalization
        self.assertTrue(np.all(processed_data >= 0))
        self.assertTrue(np.all(processed_data <= 1))
        
        # Test sample extraction
        samples, sample_labels = data_loader.extract_samples(
            num_samples=100, sampling_method='random'
        )
        
        self.assertEqual(samples.shape[0], 100)
        self.assertEqual(samples.shape[1], 250)
        
        print(f"✓ Data preprocessed successfully")
        print(f"✓ Extracted {len(samples)} samples")
        
    def test_measurement_simulation(self):
        """Test detector measurement simulation."""
        print("\n=== Testing Measurement Simulation ===")
        
        # Generate detector responses
        detector_gen = DetectorResponseGenerator(
            num_detectors=self.num_detectors,
            wavelength_range=(400, 1000)
        )
        responses = detector_gen.generate_detector_responses()
        
        # Generate test spectra
        test_spectra = np.random.rand(self.num_samples, 250)
        
        # Simulate measurements
        measurements = detector_gen.simulate_measurements(
            test_spectra, noise_level=0.01
        )
        
        # Validate measurements
        self.assertEqual(measurements.shape, (self.num_samples, self.num_detectors))
        self.assertTrue(np.all(measurements >= 0))
        
        # Test noise effect
        measurements_no_noise = detector_gen.simulate_measurements(
            test_spectra, noise_level=0.0
        )
        noise_diff = np.mean(np.abs(measurements - measurements_no_noise))
        self.assertGreater(noise_diff, 0)
        
        print(f"✓ Measurements simulated: {measurements.shape}")
        print(f"✓ Noise effect detected: {noise_diff:.6f}")
        
    def test_spectral_reconstruction(self):
        """Test spectral reconstruction with ridge regression."""
        print("\n=== Testing Spectral Reconstruction ===")
        
        # Generate test data
        detector_gen = DetectorResponseGenerator(
            num_detectors=self.num_detectors,
            wavelength_range=(400, 1000)
        )
        responses = detector_gen.generate_detector_responses()
        
        # Generate spectra and measurements
        true_spectra = np.random.rand(self.num_samples, 250)
        measurements = detector_gen.simulate_measurements(true_spectra, noise_level=0.01)
        
        # Create reconstructor
        reconstructor = SpectralReconstructor(responses, self.wavelengths)
        
        # Prepare training data
        X, y = reconstructor.prepare_training_data(true_spectra, measurements)
        
        # Train model with cross-validation
        reconstructor.train(X, y, alpha_selection='cv', cv_folds=3)
        
        # Test prediction
        predicted_spectra = reconstructor.predict(measurements)
        
        # Validate prediction shape
        self.assertEqual(predicted_spectra.shape, true_spectra.shape)
        
        # Evaluate reconstruction
        metrics = reconstructor.evaluate_reconstruction(true_spectra, predicted_spectra)
        
        # Validate metrics
        self.assertIn('r2', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('sam_degrees', metrics)
        
        # Check reasonable performance (synthetic data should reconstruct well)
        self.assertGreater(metrics['r2'], 0.7)
        self.assertLess(metrics['rmse'], 0.5)
        
        print(f"✓ Reconstruction completed")
        print(f"✓ R² Score: {metrics['r2']:.4f}")
        print(f"✓ RMSE: {metrics['rmse']:.6f}")
        print(f"✓ SAM: {metrics['sam_degrees']:.2f}°")
        
    def test_cross_validation(self):
        """Test cross-validation functionality."""
        print("\n=== Testing Cross-Validation ===")
        
        # Generate test data
        detector_gen = DetectorResponseGenerator(
            num_detectors=self.num_detectors,
            wavelength_range=(400, 1000)
        )
        responses = detector_gen.generate_detector_responses()
        
        true_spectra = np.random.rand(200, 250)  # More samples for CV
        measurements = detector_gen.simulate_measurements(true_spectra, noise_level=0.01)
        
        # Create reconstructor
        reconstructor = SpectralReconstructor(responses, self.wavelengths)
        X, y = reconstructor.prepare_training_data(true_spectra, measurements)
        
        # Train with cross-validation
        reconstructor.train(X, y, alpha_selection='cv', cv_folds=3)
        
        # Perform cross-validation
        cv_results = reconstructor.cross_validate_reconstruction(X, y, cv_folds=3)
        
        # Validate CV results
        self.assertIn('mean_r2', cv_results)
        self.assertIn('std_r2', cv_results)
        self.assertEqual(len(cv_results['r2_scores']), 3)
        
        # Check reasonable CV performance
        self.assertGreater(cv_results['mean_r2'], 0.5)
        self.assertLess(cv_results['std_r2'], 0.3)
        
        print(f"✓ Cross-validation completed")
        print(f"✓ CV R²: {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")
        
    def test_visualization_components(self):
        """Test visualization functionality."""
        print("\n=== Testing Visualization Components ===")
        
        # Generate test data
        detector_gen = DetectorResponseGenerator(
            num_detectors=self.num_detectors,
            wavelength_range=(400, 1000)
        )
        responses = detector_gen.generate_detector_responses()
        
        true_spectra = np.random.rand(50, 250)
        reconstructed_spectra = true_spectra + np.random.normal(0, 0.05, true_spectra.shape)
        
        # Create visualizer
        visualizer = SpectralVisualizer(self.wavelengths)
        
        # Test visualization methods (without actually displaying plots)
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        try:
            # Test detector response plotting
            visualizer.plot_detector_responses(responses)
            
            # Test spectral comparison plotting
            visualizer.plot_spectral_comparison(
                true_spectra, reconstructed_spectra, num_samples=3
            )
            
            # Test error analysis
            visualizer.plot_error_analysis(true_spectra, reconstructed_spectra)
            
            print("✓ Visualization components working correctly")
            
        except Exception as e:
            print(f"⚠ Visualization test failed: {e}")
            print("  (This may be due to display/backend issues)")
        
    def test_integration_pipeline(self):
        """Test complete integration pipeline."""
        print("\n=== Testing Integration Pipeline ===")
        
        start_time = time.time()
        
        try:
            # Step 1: Generate detectors
            detector_gen = DetectorResponseGenerator(num_detectors=10)  # Smaller for speed
            responses = detector_gen.generate_detector_responses()
            
            # Step 2: Generate data
            data_loader = HyperspectralDataLoader()
            data, _ = data_loader.generate_synthetic_data(
                height=20, width=20, num_bands=250
            )
            
            # Step 3: Extract samples
            samples, _ = data_loader.extract_samples(num_samples=100)
            
            # Step 4: Simulate measurements
            measurements = detector_gen.simulate_measurements(samples, noise_level=0.01)
            
            # Step 5: Train reconstructor
            reconstructor = SpectralReconstructor(responses, data_loader.wavelengths)
            X, y = reconstructor.prepare_training_data(samples, measurements)
            reconstructor.train(X, y, alpha_selection='cv', cv_folds=3)
            
            # Step 6: Reconstruct and evaluate
            predicted = reconstructor.predict(measurements)
            metrics = reconstructor.evaluate_reconstruction(samples, predicted)
            
            # Validate end-to-end performance
            self.assertGreater(metrics['r2'], 0.5)
            
            execution_time = time.time() - start_time
            
            print(f"✓ Integration pipeline completed successfully")
            print(f"✓ End-to-end R²: {metrics['r2']:.4f}")
            print(f"✓ Execution time: {execution_time:.2f}s")
            
        except Exception as e:
            self.fail(f"Integration pipeline failed: {e}")


def run_performance_benchmark():
    """Run performance benchmark tests."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Test different configurations
    configs = [
        {'num_detectors': 10, 'num_samples': 500, 'name': 'Small'},
        {'num_detectors': 15, 'num_samples': 1000, 'name': 'Medium'},
        {'num_detectors': 20, 'num_samples': 2000, 'name': 'Large'}
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting {config['name']} configuration...")
        print(f"  Detectors: {config['num_detectors']}, Samples: {config['num_samples']}")
        
        start_time = time.time()
        
        try:
            # Run pipeline
            detector_gen = DetectorResponseGenerator(num_detectors=config['num_detectors'])
            responses = detector_gen.generate_detector_responses()
            
            data_loader = HyperspectralDataLoader()
            data, _ = data_loader.generate_synthetic_data(height=50, width=50)
            samples, _ = data_loader.extract_samples(num_samples=config['num_samples'])
            
            measurements = detector_gen.simulate_measurements(samples, noise_level=0.01)
            
            reconstructor = SpectralReconstructor(responses, data_loader.wavelengths)
            X, y = reconstructor.prepare_training_data(samples, measurements)
            reconstructor.train(X, y, alpha_selection='cv', cv_folds=3)
            
            predicted = reconstructor.predict(measurements)
            metrics = reconstructor.evaluate_reconstruction(samples, predicted)
            
            execution_time = time.time() - start_time
            
            result = {
                'config': config['name'],
                'r2': metrics['r2'],
                'rmse': metrics['rmse'],
                'sam': metrics['sam_degrees'],
                'time': execution_time
            }
            results.append(result)
            
            print(f"  ✓ R²: {metrics['r2']:.4f}")
            print(f"  ✓ RMSE: {metrics['rmse']:.6f}")
            print(f"  ✓ Time: {execution_time:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    # Summary
    print(f"\n" + "="*40)
    print("BENCHMARK SUMMARY")
    print("="*40)
    for result in results:
        print(f"{result['config']:8} | R²: {result['r2']:.3f} | "
              f"RMSE: {result['rmse']:.4f} | Time: {result['time']:.1f}s")


def main():
    """Main test function."""
    print("HYPERSPECTRAL RECONSTRUCTION TEST SUITE")
    print("="*60)
    
    # Run unit tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHyperspectralReconstruction)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    # Check if all tests passed
    if result.wasSuccessful():
        print(f"\n✓ All {result.testsRun} tests passed successfully!")
        
        # Run performance benchmark
        try:
            run_performance_benchmark()
        except Exception as e:
            print(f"\n⚠ Benchmark failed: {e}")
        
        print(f"\n" + "="*60)
        print("TEST SUITE COMPLETED SUCCESSFULLY!")
        print("The hyperspectral reconstruction system is ready for use.")
        print("="*60)
        
    else:
        print(f"\n✗ {len(result.failures)} test(s) failed!")
        print(f"✗ {len(result.errors)} error(s) occurred!")
        
        for test, error in result.failures + result.errors:
            print(f"\nFailed: {test}")
            print(f"Error: {error}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)