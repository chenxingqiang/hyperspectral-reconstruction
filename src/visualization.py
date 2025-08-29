"""
Visualization Module for Hyperspectral Reconstruction Analysis

This module provides comprehensive visualization tools for analyzing detector responses,
spectral reconstruction results, and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict, Any
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

# Set style
plt.style.use('default')
sns.set_palette("husl")


class SpectralVisualizer:
    """Comprehensive visualization tool for hyperspectral reconstruction analysis."""
    
    def __init__(self, wavelengths: np.ndarray):
        """
        Initialize the visualizer.
        
        Args:
            wavelengths: Wavelength array in nm
        """
        self.wavelengths = wavelengths
        self.figure_size = (12, 8)
        self.dpi = 300
        
    def plot_detector_responses(self, 
                              detector_responses: np.ndarray,
                              center_wavelengths: Optional[np.ndarray] = None,
                              title: str = "Detector Response Functions",
                              compact_legend: bool = True,
                              save_path: Optional[str] = None) -> None:
        """
        Plot detector response functions.
        
        Args:
            detector_responses: Detector response matrix (num_wavelengths, num_detectors)
            center_wavelengths: Center wavelengths for each detector
            title: Plot title
            compact_legend: Use compact legend formatting
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        num_detectors = detector_responses.shape[1]
        colors = plt.cm.tab20(np.linspace(0, 1, num_detectors))
        
        # Plot individual responses
        for i in range(num_detectors):
            center_label = f" ({center_wavelengths[i]:.1f}nm)" if center_wavelengths is not None else ""
            if compact_legend and num_detectors > 10:
                # Use shorter labels for many detectors
                label = f'D{i+1}{center_label}'
            else:
                label = f'Detector {i+1}{center_label}'
            
            ax1.plot(self.wavelengths, detector_responses[:, i], 
                    color=colors[i], linewidth=2, alpha=0.8,
                    label=label)
        
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Transmittance')
        ax1.set_title(f'{title} - Individual Responses')
        ax1.grid(True, alpha=0.3)
        # Place legend outside the plot area
        legend_fontsize = 7 if (compact_legend and num_detectors > 10) else 8
        legend_ncol = 2 if (compact_legend and num_detectors > 10) else 1
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, 
                  fancybox=True, shadow=True, fontsize=legend_fontsize, ncol=legend_ncol)
        
        # Plot combined response and coverage analysis
        combined_response = np.sum(detector_responses, axis=1)
        ax2.plot(self.wavelengths, combined_response, 'k-', linewidth=3, label='Combined Response')
        ax2.fill_between(self.wavelengths, combined_response, alpha=0.3)
        
        # Add coverage statistics
        min_coverage = np.min(combined_response)
        max_coverage = np.max(combined_response)
        mean_coverage = np.mean(combined_response)
        
        ax2.axhline(y=mean_coverage, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_coverage:.3f}')
        ax2.axhline(y=min_coverage, color='orange', linestyle='--', alpha=0.7, label=f'Min: {min_coverage:.3f}')
        
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Combined Transmittance')
        ax2.set_title('Combined Detector Coverage')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout(rect=[0, 0, 0.75, 1])  # Leave space for external legend
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_spectral_comparison(self, 
                               true_spectra: np.ndarray,
                               reconstructed_spectra: np.ndarray,
                               sample_indices: Optional[List[int]] = None,
                               num_samples: int = 5,
                               title: str = "Spectral Reconstruction Comparison",
                               save_path: Optional[str] = None) -> None:
        """
        Plot comparison between true and reconstructed spectra.
        
        Args:
            true_spectra: Ground truth spectra (num_samples, num_wavelengths)
            reconstructed_spectra: Reconstructed spectra (num_samples, num_wavelengths)
            sample_indices: Specific sample indices to plot
            num_samples: Number of random samples to plot if sample_indices not provided
            title: Plot title
            save_path: Optional path to save the plot
        """
        if sample_indices is None:
            sample_indices = np.random.choice(len(true_spectra), 
                                            min(num_samples, len(true_spectra)), 
                                            replace=False)
        
        num_plots = len(sample_indices)
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3*num_plots))
        if num_plots == 1:
            axes = [axes]
        
        for i, (ax, idx) in enumerate(zip(axes, sample_indices)):
            ax.plot(self.wavelengths, true_spectra[idx], 'b-', linewidth=2, 
                   label='True Spectrum', alpha=0.8)
            ax.plot(self.wavelengths, reconstructed_spectra[idx], 'r--', linewidth=2, 
                   label='Reconstructed', alpha=0.8)
            
            # Calculate error
            error = np.abs(true_spectra[idx] - reconstructed_spectra[idx])
            ax.fill_between(self.wavelengths, 
                          np.minimum(true_spectra[idx], reconstructed_spectra[idx]),
                          np.maximum(true_spectra[idx], reconstructed_spectra[idx]),
                          alpha=0.2, color='gray', label='Error')
            
            # Add metrics
            mse = np.mean(error ** 2)
            r2 = 1 - np.sum(error ** 2) / np.sum((true_spectra[idx] - np.mean(true_spectra[idx])) ** 2)
            
            ax.set_title(f'Sample {idx} - MSE: {mse:.6f}, R²: {r2:.3f}')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Reflectance')
            ax.grid(True, alpha=0.3)
            # Position legend outside the plot area
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        plt.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout(rect=[0, 0, 0.85, 0.96])  # Leave space for external legends
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_reconstruction_statistics(self, 
                                     metrics_dict: Dict[str, float],
                                     cv_results: Optional[Dict[str, Any]] = None,
                                     title: str = "Reconstruction Performance",
                                     save_path: Optional[str] = None) -> None:
        """
        Plot reconstruction performance statistics.
        
        Args:
            metrics_dict: Dictionary of performance metrics
            cv_results: Optional cross-validation results
            title: Plot title
            save_path: Optional path to save the plot
        """
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 3, figure=fig)
        
        # Main metrics display
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        # Create metrics table
        metrics_text = []
        for key, value in metrics_dict.items():
            if 'r2' in key.lower():
                metrics_text.append(f"{key.upper()}: {value:.4f}")
            elif 'sam' in key.lower():
                metrics_text.append(f"{key.upper()}: {value:.3f}°")
            else:
                metrics_text.append(f"{key.upper()}: {value:.6f}")
        
        metrics_table = "\n".join(metrics_text)
        ax1.text(0.5, 0.5, metrics_table, transform=ax1.transAxes, 
                fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        ax1.set_title(title, fontsize=16, pad=20)
        
        if cv_results is not None:
            # Cross-validation box plots
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.boxplot([cv_results['mse_scores']], labels=['MSE'])
            ax2.set_title('CV MSE Distribution')
            ax2.grid(True, alpha=0.3)
            
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.boxplot([cv_results['r2_scores']], labels=['R²'])
            ax3.set_title('CV R² Distribution')
            ax3.grid(True, alpha=0.3)
            
            ax4 = fig.add_subplot(gs[1, 2])
            ax4.boxplot([cv_results['sam_scores']], labels=['SAM (°)'])
            ax4.set_title('CV SAM Distribution')
            ax4.grid(True, alpha=0.3)
            
            # CV fold comparison
            ax5 = fig.add_subplot(gs[2, :])
            folds = range(1, len(cv_results['mse_scores']) + 1)
            
            ax5_twin = ax5.twinx()
            
            line1 = ax5.plot(folds, cv_results['mse_scores'], 'bo-', label='MSE', alpha=0.7)
            line2 = ax5_twin.plot(folds, cv_results['r2_scores'], 'ro-', label='R²', alpha=0.7)
            
            ax5.set_xlabel('CV Fold')
            ax5.set_ylabel('MSE', color='b')
            ax5_twin.set_ylabel('R²', color='r')
            ax5.set_title('Cross-Validation Performance by Fold')
            ax5.grid(True, alpha=0.3)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax5.legend(lines, labels, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_error_analysis(self, 
                          true_spectra: np.ndarray,
                          reconstructed_spectra: np.ndarray,
                          title: str = "Error Analysis",
                          save_path: Optional[str] = None) -> None:
        """
        Plot detailed error analysis.
        
        Args:
            true_spectra: Ground truth spectra
            reconstructed_spectra: Reconstructed spectra
            title: Plot title
            save_path: Optional path to save the plot
        """
        # Calculate errors
        absolute_error = np.abs(true_spectra - reconstructed_spectra)
        relative_error = absolute_error / (np.abs(true_spectra) + 1e-8)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Mean absolute error by wavelength
        mean_abs_error = np.mean(absolute_error, axis=0)
        std_abs_error = np.std(absolute_error, axis=0)
        
        axes[0, 0].plot(self.wavelengths, mean_abs_error, 'b-', linewidth=2)
        axes[0, 0].fill_between(self.wavelengths, 
                               mean_abs_error - std_abs_error,
                               mean_abs_error + std_abs_error,
                               alpha=0.3)
        axes[0, 0].set_xlabel('Wavelength (nm)')
        axes[0, 0].set_ylabel('Mean Absolute Error')
        axes[0, 0].set_title('Wavelength-wise Error Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Error histogram
        axes[0, 1].hist(absolute_error.flatten(), bins=50, alpha=0.7, density=True)
        axes[0, 1].set_xlabel('Absolute Error')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. True vs Predicted scatter plot
        sample_size = min(1000, len(true_spectra))
        sample_indices = np.random.choice(len(true_spectra), sample_size, replace=False)
        true_sample = true_spectra[sample_indices].flatten()
        pred_sample = reconstructed_spectra[sample_indices].flatten()
        
        axes[1, 0].scatter(true_sample, pred_sample, alpha=0.5, s=1)
        min_val = min(np.min(true_sample), np.min(pred_sample))
        max_val = max(np.max(true_sample), np.max(pred_sample))
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[1, 0].set_xlabel('True Values')
        axes[1, 0].set_ylabel('Predicted Values')
        axes[1, 0].set_title('True vs Predicted Scatter Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Relative error by wavelength
        mean_rel_error = np.mean(relative_error, axis=0)
        axes[1, 1].plot(self.wavelengths, mean_rel_error * 100, 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Wavelength (nm)')
        axes[1, 1].set_ylabel('Mean Relative Error (%)')
        axes[1, 1].set_title('Relative Error by Wavelength')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_pca_analysis(self, 
                         true_spectra: np.ndarray,
                         reconstructed_spectra: np.ndarray,
                         n_components: int = 3,
                         title: str = "PCA Analysis",
                         save_path: Optional[str] = None) -> None:
        """
        Plot PCA analysis of spectral data.
        
        Args:
            true_spectra: Ground truth spectra
            reconstructed_spectra: Reconstructed spectra
            n_components: Number of PCA components
            title: Plot title
            save_path: Optional path to save the plot
        """
        # Combine data for PCA
        combined_data = np.vstack([true_spectra, reconstructed_spectra])
        labels = np.hstack([np.zeros(len(true_spectra)), np.ones(len(reconstructed_spectra))])
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(combined_data)
        
        fig = plt.figure(figsize=(15, 10))
        
        if n_components >= 3:
            # 3D scatter plot
            ax1 = fig.add_subplot(221, projection='3d')
            true_idx = labels == 0
            recon_idx = labels == 1
            
            ax1.scatter(pca_data[true_idx, 0], pca_data[true_idx, 1], pca_data[true_idx, 2],
                       c='blue', alpha=0.6, label='True Spectra', s=20)
            ax1.scatter(pca_data[recon_idx, 0], pca_data[recon_idx, 1], pca_data[recon_idx, 2],
                       c='red', alpha=0.6, label='Reconstructed', s=20)
            
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax1.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
            ax1.set_title('3D PCA Visualization')
            ax1.legend()
        
        # 2D scatter plot
        ax2 = fig.add_subplot(222)
        true_idx = labels == 0
        recon_idx = labels == 1
        
        ax2.scatter(pca_data[true_idx, 0], pca_data[true_idx, 1],
                   c='blue', alpha=0.6, label='True Spectra', s=20)
        ax2.scatter(pca_data[recon_idx, 0], pca_data[recon_idx, 1],
                   c='red', alpha=0.6, label='Reconstructed', s=20)
        
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax2.set_title('2D PCA Visualization')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Explained variance plot
        ax3 = fig.add_subplot(223)
        ax3.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
               pca.explained_variance_ratio_ * 100)
        ax3.set_xlabel('Principal Component')
        ax3.set_ylabel('Explained Variance (%)')
        ax3.set_title('Explained Variance by Component')
        ax3.grid(True, alpha=0.3)
        
        # Cumulative explained variance
        ax4 = fig.add_subplot(224)
        cumulative_var = np.cumsum(pca.explained_variance_ratio_) * 100
        ax4.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'bo-')
        ax4.set_xlabel('Number of Components')
        ax4.set_ylabel('Cumulative Explained Variance (%)')
        ax4.set_title('Cumulative Explained Variance')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def plot_alpha_selection_analysis(self, 
                                    training_history: Dict[str, Any],
                                    title: str = "Regularization Parameter Analysis",
                                    save_path: Optional[str] = None) -> None:
        """
        Plot alpha selection analysis from ridge regression.
        
        Args:
            training_history: Training history containing alpha selection results
            title: Plot title
            save_path: Optional path to save the plot
        """
        if 'alphas' not in training_history:
            print("No alpha selection history available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        alphas = training_history['alphas']
        cv_scores = training_history['cv_scores']
        optimal_alpha = training_history['optimal_alpha']
        
        # Mean CV score vs alpha
        mean_scores = np.mean(cv_scores, axis=1)
        std_scores = np.std(cv_scores, axis=1)
        
        axes[0, 0].semilogx(alphas, -mean_scores, 'b-', linewidth=2)
        axes[0, 0].fill_between(alphas, -mean_scores - std_scores, 
                               -mean_scores + std_scores, alpha=0.3)
        axes[0, 0].axvline(x=optimal_alpha, color='r', linestyle='--', 
                          label=f'Optimal α = {optimal_alpha:.2e}')
        axes[0, 0].set_xlabel('Regularization Parameter (α)')
        axes[0, 0].set_ylabel('Cross-Validation Score (MSE)')
        axes[0, 0].set_title('Alpha Selection via Cross-Validation')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # CV score distribution for optimal alpha
        optimal_idx = np.argmin(np.abs(alphas - optimal_alpha))
        axes[0, 1].hist(cv_scores[optimal_idx], bins=10, alpha=0.7)
        axes[0, 1].set_xlabel('CV Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'CV Score Distribution (α = {optimal_alpha:.2e})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Model coefficients analysis
        if 'model_coefficients' in training_history:
            coef = training_history['model_coefficients']
            if coef.ndim == 2:
                coef_norm = np.linalg.norm(coef, axis=1)
                axes[1, 0].plot(coef_norm)
                axes[1, 0].set_xlabel('Output Dimension')
                axes[1, 0].set_ylabel('Coefficient Norm')
                axes[1, 0].set_title('Model Coefficient Norms')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].plot(coef)
                axes[1, 0].set_xlabel('Feature Index')
                axes[1, 0].set_ylabel('Coefficient Value')
                axes[1, 0].set_title('Model Coefficients')
                axes[1, 0].grid(True, alpha=0.3)
        
        # Training performance
        if 'train_mse' in training_history and 'train_r2' in training_history:
            metrics = ['MSE', 'R²']
            values = [training_history['train_mse'], training_history['train_r2']]
            
            bars = axes[1, 1].bar(metrics, values, color=['red', 'blue'], alpha=0.7)
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Training Performance')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.4f}', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_report(self, 
                                  detector_responses: np.ndarray,
                                  true_spectra: np.ndarray,
                                  reconstructed_spectra: np.ndarray,
                                  metrics_dict: Dict[str, float],
                                  training_history: Dict[str, Any],
                                  cv_results: Optional[Dict[str, Any]] = None,
                                  save_dir: Optional[str] = None) -> None:
        """
        Create a comprehensive visualization report.
        
        Args:
            detector_responses: Detector response matrix
            true_spectra: Ground truth spectra
            reconstructed_spectra: Reconstructed spectra
            metrics_dict: Performance metrics
            training_history: Training history
            cv_results: Optional cross-validation results
            save_dir: Optional directory to save all plots
        """
        print("Generating comprehensive visualization report...")
        
        # Plot 1: Detector responses
        save_path1 = f"{save_dir}/detector_responses.png" if save_dir else None
        self.plot_detector_responses(detector_responses, save_path=save_path1)
        
        # Plot 2: Spectral comparisons
        save_path2 = f"{save_dir}/spectral_comparison.png" if save_dir else None
        self.plot_spectral_comparison(true_spectra, reconstructed_spectra, 
                                    num_samples=8, save_path=save_path2)
        
        # Plot 3: Reconstruction statistics
        save_path3 = f"{save_dir}/reconstruction_stats.png" if save_dir else None
        self.plot_reconstruction_statistics(metrics_dict, cv_results, save_path=save_path3)
        
        # Plot 4: Error analysis
        save_path4 = f"{save_dir}/error_analysis.png" if save_dir else None
        self.plot_error_analysis(true_spectra, reconstructed_spectra, save_path=save_path4)
        
        # Plot 5: PCA analysis
        save_path5 = f"{save_dir}/pca_analysis.png" if save_dir else None
        self.plot_pca_analysis(true_spectra, reconstructed_spectra, save_path=save_path5)
        
        # Plot 6: Alpha selection analysis
        save_path6 = f"{save_dir}/alpha_selection.png" if save_dir else None
        self.plot_alpha_selection_analysis(training_history, save_path=save_path6)
        
        print("Comprehensive report generation completed!")


if __name__ == "__main__":
    # Example usage
    print("Spectral Visualizer Test")
    print("=" * 40)
    
    # Generate test data
    wavelengths = np.linspace(400, 1000, 250)
    num_samples = 100
    
    # Create synthetic detector responses
    num_detectors = 15
    center_wavelengths = np.linspace(450, 950, num_detectors)
    detector_responses = np.zeros((len(wavelengths), num_detectors))
    
    for i, center in enumerate(center_wavelengths):
        sigma = 50 / (2 * np.sqrt(2 * np.log(2)))
        detector_responses[:, i] = 0.5 * np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)
    
    # Generate synthetic spectra
    true_spectra = np.random.rand(num_samples, len(wavelengths))
    reconstructed_spectra = true_spectra + np.random.normal(0, 0.05, true_spectra.shape)
    
    # Create visualizer
    visualizer = SpectralVisualizer(wavelengths)
    
    # Test detector response visualization
    visualizer.plot_detector_responses(detector_responses, center_wavelengths)
    
    # Test spectral comparison
    visualizer.plot_spectral_comparison(true_spectra, reconstructed_spectra, num_samples=3)