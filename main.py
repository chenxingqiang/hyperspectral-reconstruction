"""
Main Execution Script for Hyperspectral Reconstruction

This script provides a complete workflow for hyperspectral reconstruction using
15 Gaussian detectors, ridge regression with cross-validation, and comprehensive analysis.
"""

import os
import sys
import numpy as np
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import warnings

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from detector_response import DetectorResponseGenerator
from data_utils import HyperspectralDataLoader, SpectralDataManager
from spectral_reconstruction import SpectralReconstructor
from visualization import SpectralVisualizer


class HyperspectralReconstructionPipeline:
    """Complete pipeline for hyperspectral reconstruction experiments."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the reconstruction pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.results_dir = self.setup_results_directory()
        
        # Initialize components
        self.detector_generator = None
        self.data_loader = None
        self.reconstructor = None
        self.visualizer = None
        
        # Results storage
        self.results = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'config': self.config
            },
            'detector_config': {},
            'data_info': {},
            'reconstruction_metrics': {},
            'cross_validation_results': {},
            'execution_time': {}
        }
    
    def load_config(self, config_path: str = None) -> dict:
        """Load configuration from file or use defaults."""
        default_config = {
            # Detector configuration
            'num_detectors': 15,
            'wavelength_range': [400, 1000],
            'detector_fwhm': 50.0,
            'peak_transmittance': 0.5,
            'overlap_factor': 0.8,
            
            # Data configuration
            'data_source': 'synthetic',  # 'synthetic' or 'xiong_an'
            'xiong_an_data_path': '../2.xiongan雄安新区航空高光谱遥感影像分类数据集/xiongan.mat',
            'xiong_an_gt_path': '../2.xiongan雄安新区航空高光谱遥感影像分类数据集/xiongan_gt.mat',
            'synthetic_height': 100,
            'synthetic_width': 100,
            'synthetic_bands': 250,
            'synthetic_classes': 5,
            'num_samples': 2000,
            'sampling_method': 'random',
            
            # Preprocessing
            'normalization': 'minmax',
            'remove_bad_bands': True,
            'noise_threshold': 0.01,
            
            # Noise simulation
            'noise_level': 0.01,  # 1% noise
            
            # Reconstruction configuration
            'alpha_selection': 'cv',  # 'cv', 'gcv', 'manual'
            'cv_folds': 5,
            'alpha_range': [1e-6, 1e2],
            'n_alphas': 50,
            
            # Visualization
            'generate_plots': True,
            'plot_num_samples': 8,
            'save_plots': True,
            
            # Output
            'save_results': True,
            'save_model': True,
            'save_detector_config': True,
            'verbose': True
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
            
        return default_config
    
    def setup_results_directory(self) -> str:
        """Setup results directory with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results/experiment_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(f"{results_dir}/plots", exist_ok=True)
        os.makedirs(f"{results_dir}/models", exist_ok=True)
        os.makedirs(f"{results_dir}/data", exist_ok=True)
        
        return results_dir
    
    def step1_generate_detectors(self):
        """Step 1: Generate detector response functions."""
        if self.config['verbose']:
            print("Step 1: Generating detector response functions...")
        
        start_time = time.time()
        
        self.detector_generator = DetectorResponseGenerator(
            num_detectors=self.config['num_detectors'],
            wavelength_range=tuple(self.config['wavelength_range'])
        )
        
        detector_responses = self.detector_generator.generate_detector_responses(
            fwhm=self.config['detector_fwhm'],
            peak_transmittance=self.config['peak_transmittance'],
            overlap_factor=self.config['overlap_factor']
        )
        
        # Store detector configuration
        detector_stats = self.detector_generator.get_coverage_statistics()
        self.results['detector_config'] = detector_stats
        
        # Save detector configuration
        if self.config['save_detector_config']:
            detector_path = f"{self.results_dir}/data/detector_config.npz"
            self.detector_generator.save_detector_config(detector_path)
        
        # Visualize detector responses
        if self.config['generate_plots']:
            plot_path = f"{self.results_dir}/plots/detector_responses.png" if self.config['save_plots'] else None
            self.detector_generator.plot_detector_responses(save_path=plot_path)
        
        execution_time = time.time() - start_time
        self.results['execution_time']['detector_generation'] = execution_time
        
        if self.config['verbose']:
            print(f"  ✓ Generated {self.config['num_detectors']} detectors")
            print(f"  ✓ Coverage uniformity: {detector_stats['coverage_uniformity']:.4f}")
            print(f"  ✓ Execution time: {execution_time:.2f}s\n")
        
        return detector_responses
    
    def step2_load_data(self):
        """Step 2: Load and preprocess hyperspectral data."""
        if self.config['verbose']:
            print("Step 2: Loading and preprocessing data...")
        
        start_time = time.time()
        
        self.data_loader = HyperspectralDataLoader(
            wavelength_range=tuple(self.config['wavelength_range'])
        )
        
        # Load data based on configuration
        if self.config['data_source'] == 'xiong_an':
            try:
                data, gt = self.data_loader.load_xiong_an_data(
                    self.config['xiong_an_data_path'],
                    self.config['xiong_an_gt_path']
                )
                if self.config['verbose']:
                    print(f"  ✓ Loaded Xiong'an dataset: {data.shape}")
            except Exception as e:
                print(f"  ⚠ Failed to load Xiong'an data: {e}")
                print("  → Falling back to synthetic data")
                self.config['data_source'] = 'synthetic'
        
        if self.config['data_source'] == 'synthetic':
            data, gt = self.data_loader.generate_synthetic_data(
                height=self.config['synthetic_height'],
                width=self.config['synthetic_width'],
                num_bands=self.config['synthetic_bands'],
                num_classes=self.config['synthetic_classes']
            )
            if self.config['verbose']:
                print(f"  ✓ Generated synthetic data: {data.shape}")
        
        # Preprocess data
        processed_data = self.data_loader.preprocess_data(
            normalization=self.config['normalization'],
            remove_bad_bands=self.config['remove_bad_bands'],
            noise_threshold=self.config['noise_threshold']
        )
        
        # Extract samples
        sample_spectra, sample_labels = self.data_loader.extract_samples(
            num_samples=self.config['num_samples'],
            sampling_method=self.config['sampling_method']
        )
        
        # Store data information
        self.results['data_info'] = {
            'source': self.config['data_source'],
            'original_shape': data.shape,
            'processed_shape': processed_data.shape,
            'num_samples': len(sample_spectra),
            'wavelength_range': [self.data_loader.wavelengths[0], self.data_loader.wavelengths[-1]],
            'num_bands': len(self.data_loader.wavelengths)
        }
        
        execution_time = time.time() - start_time
        self.results['execution_time']['data_loading'] = execution_time
        
        if self.config['verbose']:
            print(f"  ✓ Extracted {len(sample_spectra)} samples")
            print(f"  ✓ Wavelength range: {self.data_loader.wavelengths[0]:.1f}-{self.data_loader.wavelengths[-1]:.1f} nm")
            print(f"  ✓ Execution time: {execution_time:.2f}s\n")
        
        return sample_spectra, sample_labels
    
    def step3_simulate_measurements(self, sample_spectra, detector_responses):
        """Step 3: Simulate detector measurements with noise."""
        if self.config['verbose']:
            print("Step 3: Simulating detector measurements...")
        
        start_time = time.time()
        
        # Simulate measurements
        detector_measurements = self.detector_generator.simulate_measurements(
            sample_spectra, 
            noise_level=self.config['noise_level']
        )
        
        # Calculate SNR
        signal_power = np.mean(detector_measurements ** 2)
        noise_power = (self.config['noise_level'] ** 2) * signal_power
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
        
        self.results['data_info']['measurements_shape'] = detector_measurements.shape
        self.results['data_info']['noise_level'] = self.config['noise_level']
        self.results['data_info']['snr_db'] = snr_db
        
        execution_time = time.time() - start_time
        self.results['execution_time']['measurement_simulation'] = execution_time
        
        if self.config['verbose']:
            print(f"  ✓ Simulated measurements: {detector_measurements.shape}")
            print(f"  ✓ Noise level: {self.config['noise_level']*100:.1f}%")
            print(f"  ✓ SNR: {snr_db:.1f} dB")
            print(f"  ✓ Execution time: {execution_time:.2f}s\n")
        
        return detector_measurements
    
    def step4_train_reconstructor(self, detector_measurements, sample_spectra, detector_responses):
        """Step 4: Train the spectral reconstructor."""
        if self.config['verbose']:
            print("Step 4: Training spectral reconstructor...")
        
        start_time = time.time()
        
        # Initialize reconstructor
        self.reconstructor = SpectralReconstructor(
            detector_responses, 
            self.data_loader.wavelengths
        )
        
        # Prepare training data
        X, y = self.reconstructor.prepare_training_data(
            sample_spectra, 
            detector_measurements, 
            normalize=True
        )
        
        # Train model
        self.reconstructor.train(
            X, y,
            alpha_selection=self.config['alpha_selection'],
            cv_folds=self.config['cv_folds'],
            alpha_range=tuple(self.config['alpha_range'])
        )
        
        # Store training results
        self.results['reconstruction_metrics']['optimal_alpha'] = self.reconstructor.optimal_alpha
        self.results['reconstruction_metrics']['training_mse'] = self.reconstructor.training_history['train_mse']
        self.results['reconstruction_metrics']['training_r2'] = self.reconstructor.training_history['train_r2']
        
        # Save model
        if self.config['save_model']:
            model_path = f"{self.results_dir}/models/reconstructor_model.npz"
            self.reconstructor.save_model(model_path)
        
        execution_time = time.time() - start_time
        self.results['execution_time']['training'] = execution_time
        
        if self.config['verbose']:
            print(f"  ✓ Optimal α: {self.reconstructor.optimal_alpha:.2e}")
            print(f"  ✓ Training R²: {self.reconstructor.training_history['train_r2']:.4f}")
            print(f"  ✓ Execution time: {execution_time:.2f}s\n")
        
        return X, y
    
    def step5_evaluate_reconstruction(self, X, y, sample_spectra, detector_measurements):
        """Step 5: Evaluate reconstruction performance."""
        if self.config['verbose']:
            print("Step 5: Evaluating reconstruction performance...")
        
        start_time = time.time()
        
        # Predict reconstructed spectra
        reconstructed_spectra = self.reconstructor.predict(detector_measurements)
        
        # Calculate reconstruction metrics
        metrics = self.reconstructor.evaluate_reconstruction(sample_spectra, reconstructed_spectra)
        self.results['reconstruction_metrics'].update(metrics)
        
        # Perform cross-validation
        cv_results = self.reconstructor.cross_validate_reconstruction(X, y, self.config['cv_folds'])
        self.results['cross_validation_results'] = cv_results
        
        execution_time = time.time() - start_time
        self.results['execution_time']['evaluation'] = execution_time
        
        if self.config['verbose']:
            print(f"  ✓ Reconstruction R²: {metrics['r2']:.4f}")
            print(f"  ✓ RMSE: {metrics['rmse']:.6f}")
            print(f"  ✓ SAM: {metrics['sam_degrees']:.2f}°")
            print(f"  ✓ CV R²: {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")
            print(f"  ✓ Execution time: {execution_time:.2f}s\n")
        
        return reconstructed_spectra, metrics, cv_results
    
    def step6_generate_visualizations(self, detector_responses, sample_spectra, reconstructed_spectra, metrics, cv_results):
        """Step 6: Generate comprehensive visualizations."""
        if not self.config['generate_plots']:
            return
        
        if self.config['verbose']:
            print("Step 6: Generating visualizations...")
        
        start_time = time.time()
        
        # Initialize visualizer
        self.visualizer = SpectralVisualizer(self.data_loader.wavelengths)
        
        # Generate comprehensive report
        save_dir = f"{self.results_dir}/plots" if self.config['save_plots'] else None
        
        self.visualizer.create_comprehensive_report(
            detector_responses=detector_responses,
            true_spectra=sample_spectra,
            reconstructed_spectra=reconstructed_spectra,
            metrics_dict=metrics,
            training_history=self.reconstructor.training_history,
            cv_results=cv_results,
            save_dir=save_dir
        )
        
        execution_time = time.time() - start_time
        self.results['execution_time']['visualization'] = execution_time
        
        if self.config['verbose']:
            print(f"  ✓ Generated comprehensive visualization report")
            print(f"  ✓ Execution time: {execution_time:.2f}s\n")
    
    def save_results(self):
        """Save experiment results."""
        if not self.config['save_results']:
            return
        
        # Calculate total execution time
        total_time = sum(self.results['execution_time'].values())
        self.results['execution_time']['total'] = total_time
        
        # Save results to JSON
        results_path = f"{self.results_dir}/experiment_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save configuration
        config_path = f"{self.results_dir}/experiment_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        if self.config['verbose']:
            print(f"Results saved to: {self.results_dir}")
            print(f"Total execution time: {total_time:.2f}s")
    
    def run_complete_pipeline(self):
        """Run the complete hyperspectral reconstruction pipeline."""
        print("=" * 60)
        print("HYPERSPECTRAL RECONSTRUCTION PIPELINE")
        print("=" * 60)
        print(f"Experiment timestamp: {self.results['experiment_info']['timestamp']}")
        print(f"Results directory: {self.results_dir}\n")
        
        try:
            # Step 1: Generate detectors
            detector_responses = self.step1_generate_detectors()
            
            # Step 2: Load data
            sample_spectra, sample_labels = self.step2_load_data()
            
            # Step 3: Simulate measurements
            detector_measurements = self.step3_simulate_measurements(sample_spectra, detector_responses)
            
            # Step 4: Train reconstructor
            X, y = self.step4_train_reconstructor(detector_measurements, sample_spectra, detector_responses)
            
            # Step 5: Evaluate reconstruction
            reconstructed_spectra, metrics, cv_results = self.step5_evaluate_reconstruction(
                X, y, sample_spectra, detector_measurements
            )
            
            # Step 6: Generate visualizations
            self.step6_generate_visualizations(
                detector_responses, sample_spectra, reconstructed_spectra, metrics, cv_results
            )
            
            # Save results
            self.save_results()
            
            print("=" * 60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Final Results:")
            print(f"  - Reconstruction R²: {metrics['r2']:.4f}")
            print(f"  - RMSE: {metrics['rmse']:.6f}")
            print(f"  - SAM: {metrics['sam_degrees']:.2f}°")
            print(f"  - Cross-validation R²: {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")
            print(f"  - Results saved to: {self.results_dir}")
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Hyperspectral Reconstruction Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data-source', type=str, choices=['synthetic', 'xiong_an'], 
                       default='synthetic', help='Data source to use')
    parser.add_argument('--num-samples', type=int, default=2000, help='Number of samples to use')
    parser.add_argument('--noise-level', type=float, default=0.01, help='Noise level (0.01 = 1%)')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--quiet', action='store_true', help='Disable verbose output')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = HyperspectralReconstructionPipeline(args.config)
    
    # Override config with command line arguments
    if args.data_source:
        pipeline.config['data_source'] = args.data_source
    if args.num_samples:
        pipeline.config['num_samples'] = args.num_samples
    if args.noise_level:
        pipeline.config['noise_level'] = args.noise_level
    if args.cv_folds:
        pipeline.config['cv_folds'] = args.cv_folds
    if args.no_plots:
        pipeline.config['generate_plots'] = False
    if args.quiet:
        pipeline.config['verbose'] = False
    
    # Run pipeline
    success = pipeline.run_complete_pipeline()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())