"""
Detector Response Module for Hyperspectral Reconstruction

This module generates 15 Gaussian-shaped detector response functions that cover
the wavelength range of 400-1000nm, with 50% peak transmittance at center wavelengths.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import scipy.stats as stats


class DetectorResponseGenerator:
    """Generator for Gaussian-shaped detector response functions."""

    def __init__(self, num_detectors: int = 15, wavelength_range: Tuple[int, int] = (400, 1000), num_bands: int = 250):
        """
        Initialize the detector response generator.

        Args:
            num_detectors: Number of detectors to generate (default: 15)
            wavelength_range: Wavelength range in nm (default: (400, 1000))
            num_bands: Number of wavelength bands (default: 250)
        """
        self.num_detectors = num_detectors
        self.wavelength_range = wavelength_range
        self.num_bands = num_bands
        self.wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num_bands)
        self.detector_responses = None
        self.center_wavelengths = None

    def generate_detector_responses(self,
                                  fwhm: float = 50.0,
                                  peak_transmittance: float = 0.5,
                                  overlap_factor: float = 0.8) -> np.ndarray:
        """
        Generate Gaussian detector response functions.

        Args:
            fwhm: Full Width at Half Maximum in nm (default: 50.0)
            peak_transmittance: Peak transmittance value (default: 0.5)
            overlap_factor: Overlap factor for detector spacing (default: 0.8)

        Returns:
            Array of detector responses with shape (num_wavelengths, num_detectors)
        """
        # Calculate center wavelengths with overlapping coverage
        total_range = self.wavelength_range[1] - self.wavelength_range[0]
        spacing = total_range / (self.num_detectors - 1 + overlap_factor)

        self.center_wavelengths = np.linspace(
            self.wavelength_range[0] + spacing * overlap_factor / 2,
            self.wavelength_range[1] - spacing * overlap_factor / 2,
            self.num_detectors
        )

        # Convert FWHM to standard deviation
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

        # Generate detector responses
        responses = np.zeros((len(self.wavelengths), self.num_detectors))

        for i, center_wl in enumerate(self.center_wavelengths):
            # Generate Gaussian response
            gaussian_response = stats.norm.pdf(self.wavelengths, center_wl, sigma)
            # Normalize to peak transmittance
            gaussian_response = gaussian_response / np.max(gaussian_response) * peak_transmittance
            responses[:, i] = gaussian_response

        self.detector_responses = responses
        return responses

    def plot_detector_responses(self, save_path: Optional[str] = None) -> None:
        """
        Plot the detector response functions.

        Args:
            save_path: Optional path to save the plot
        """
        if self.detector_responses is None:
            raise ValueError("Detector responses not generated. Call generate_detector_responses() first.")

        plt.figure(figsize=(12, 8))

        # Plot individual detector responses
        for i in range(self.num_detectors):
            plt.plot(self.wavelengths, self.detector_responses[:, i],
                    label=f'Detector {i+1} ({self.center_wavelengths[i]:.1f}nm)',
                    linewidth=2, alpha=0.7)

        # Plot combined response
        combined_response = np.sum(self.detector_responses, axis=1)
        plt.plot(self.wavelengths, combined_response, 'k--',
                linewidth=3, label='Combined Response', alpha=0.8)

        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Transmittance')
        plt.title('15 Gaussian Detector Response Functions (400-1000nm)')
        plt.grid(True, alpha=0.3)
        # Position legend outside plot area with better formatting
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True,
                  fancybox=True, shadow=True, fontsize=9)
        plt.tight_layout(rect=[0, 0, 0.75, 1])  # Leave space for external legend

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def get_coverage_statistics(self) -> dict:
        """
        Calculate coverage statistics for the detector array.

        Returns:
            Dictionary containing coverage statistics
        """
        if self.detector_responses is None:
            raise ValueError("Detector responses not generated. Call generate_detector_responses() first.")

        combined_response = np.sum(self.detector_responses, axis=1)
        min_coverage = np.min(combined_response)
        max_coverage = np.max(combined_response)
        mean_coverage = np.mean(combined_response)
        std_coverage = np.std(combined_response)

        # Coverage uniformity (coefficient of variation)
        coverage_uniformity = std_coverage / mean_coverage if mean_coverage > 0 else np.inf

        return {
            'min_coverage': min_coverage,
            'max_coverage': max_coverage,
            'mean_coverage': mean_coverage,
            'std_coverage': std_coverage,
            'coverage_uniformity': coverage_uniformity,
            'wavelength_range': self.wavelength_range,
            'num_detectors': self.num_detectors,
            'center_wavelengths': self.center_wavelengths.tolist()
        }

    def simulate_measurements(self, hyperspectral_data: np.ndarray,
                            noise_level: float = 0.01) -> np.ndarray:
        """
        Simulate detector measurements from hyperspectral data.

        Args:
            hyperspectral_data: Hyperspectral data with shape (..., num_wavelengths)
            noise_level: Relative noise level (default: 0.01 for 1%)

        Returns:
            Simulated detector measurements with shape (..., num_detectors)
        """
        if self.detector_responses is None:
            raise ValueError("Detector responses not generated. Call generate_detector_responses() first.")

        # Ensure hyperspectral data has correct wavelength dimension
        original_shape = hyperspectral_data.shape
        if hyperspectral_data.shape[-1] != len(self.wavelengths):
            raise ValueError(f"Expected {len(self.wavelengths)} wavelengths, got {hyperspectral_data.shape[-1]}")

        # Reshape for matrix multiplication
        reshaped_data = hyperspectral_data.reshape(-1, hyperspectral_data.shape[-1])

        # Simulate measurements by integrating with detector responses
        measurements = np.dot(reshaped_data, self.detector_responses)

        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * np.abs(measurements))
            measurements += noise

        # Reshape back to original dimensions
        output_shape = original_shape[:-1] + (self.num_detectors,)
        measurements = measurements.reshape(output_shape)

        return measurements

    def save_detector_config(self, file_path: str) -> None:
        """
        Save detector configuration to file.

        Args:
            file_path: Path to save the configuration
        """
        if self.detector_responses is None:
            raise ValueError("Detector responses not generated. Call generate_detector_responses() first.")

        np.savez(file_path,
                wavelengths=self.wavelengths,
                detector_responses=self.detector_responses,
                center_wavelengths=self.center_wavelengths,
                num_detectors=self.num_detectors,
                wavelength_range=self.wavelength_range)

    def load_detector_config(self, file_path: str) -> None:
        """
        Load detector configuration from file.

        Args:
            file_path: Path to load the configuration from
        """
        data = np.load(file_path)
        self.wavelengths = data['wavelengths']
        self.detector_responses = data['detector_responses']
        self.center_wavelengths = data['center_wavelengths']
        self.num_detectors = int(data['num_detectors'])
        self.wavelength_range = tuple(data['wavelength_range'])


if __name__ == "__main__":
    # Example usage
    detector_gen = DetectorResponseGenerator(num_detectors=15)
    responses = detector_gen.generate_detector_responses(fwhm=50.0)

    print("Detector Response Generator Test")
    print("=" * 40)

    # Print statistics
    stats = detector_gen.get_coverage_statistics()
    print(f"Coverage Statistics:")
    print(f"  Wavelength Range: {stats['wavelength_range']} nm")
    print(f"  Number of Detectors: {stats['num_detectors']}")
    print(f"  Coverage Uniformity: {stats['coverage_uniformity']:.4f}")
    print(f"  Mean Coverage: {stats['mean_coverage']:.4f}")

    # Plot responses
    detector_gen.plot_detector_responses()