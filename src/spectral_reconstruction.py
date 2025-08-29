"""
Spectral Reconstruction Module using Ridge Regression

This module implements ridge regression with k-fold cross-validation for
reconstructing high-resolution hyperspectral data from limited detector measurements.
"""

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional, Dict, Any
import warnings
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


class SpectralReconstructor:
    """Ridge regression-based spectral reconstructor with cross-validation."""
    
    def __init__(self, detector_responses: np.ndarray, wavelengths: np.ndarray):
        """
        Initialize the spectral reconstructor.
        
        Args:
            detector_responses: Detector response matrix (num_wavelengths, num_detectors)
            wavelengths: Wavelength array (num_wavelengths,)
        """
        self.detector_responses = detector_responses
        self.wavelengths = wavelengths
        self.num_wavelengths = len(wavelengths)
        self.num_detectors = detector_responses.shape[1]
        
        # Trained model parameters
        self.ridge_model = None
        self.optimal_alpha = None
        self.scaler_X = None
        self.scaler_y = None
        self.training_history = {}
        
    def prepare_training_data(self, 
                            hyperspectral_data: np.ndarray,
                            detector_measurements: np.ndarray,
                            normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for the ridge regression model.
        
        Args:
            hyperspectral_data: Ground truth spectra (num_samples, num_wavelengths)
            detector_measurements: Detector measurements (num_samples, num_detectors)
            normalize: Whether to normalize the data
            
        Returns:
            Tuple of (X_normalized, y_normalized)
        """
        # Reshape if necessary
        if hyperspectral_data.ndim > 2:
            original_shape = hyperspectral_data.shape
            hyperspectral_data = hyperspectral_data.reshape(-1, original_shape[-1])
            detector_measurements = detector_measurements.reshape(-1, detector_measurements.shape[-1])
        
        X = detector_measurements.copy()
        y = hyperspectral_data.copy()
        
        if normalize:
            # Normalize detector measurements (X)
            self.scaler_X = StandardScaler()
            X = self.scaler_X.fit_transform(X)
            
            # Normalize hyperspectral data (y)
            self.scaler_y = StandardScaler()
            y = self.scaler_y.fit_transform(y)
        
        return X, y
    
    def find_optimal_alpha_cv(self, 
                            X: np.ndarray, 
                            y: np.ndarray,
                            alpha_range: Tuple[float, float] = (1e-6, 1e2),
                            n_alphas: int = 50,
                            cv_folds: int = 5,
                            scoring: str = 'neg_mean_squared_error') -> float:
        """
        Find optimal regularization parameter using cross-validation.
        
        Args:
            X: Training features (detector measurements)
            y: Training targets (hyperspectral data)
            alpha_range: Range of alpha values to test
            n_alphas: Number of alpha values to test
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for cross-validation
            
        Returns:
            Optimal alpha value
        """
        # Generate alpha values on log scale
        alphas = np.logspace(np.log10(alpha_range[0]), 
                           np.log10(alpha_range[1]), 
                           n_alphas)
        
        # Use RidgeCV for efficient cross-validation
        ridge_cv = RidgeCV(alphas=alphas, cv=cv_folds, scoring=scoring)
        ridge_cv.fit(X, y)
        
        self.optimal_alpha = ridge_cv.alpha_
        
        # Store cross-validation results - use manual CV since cv_values_ may not exist
        cv_scores = []
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        for alpha in alphas:
            scores = []
            for train_idx, val_idx in kf.split(X):
                ridge_temp = Ridge(alpha=alpha)
                ridge_temp.fit(X[train_idx], y[train_idx])
                y_pred = ridge_temp.predict(X[val_idx])
                score = -mean_squared_error(y[val_idx], y_pred)  # Negative MSE
                scores.append(score)
            cv_scores.append(scores)
        
        self.training_history['cv_scores'] = np.array(cv_scores)
        self.training_history['alphas'] = alphas
        self.training_history['optimal_alpha'] = self.optimal_alpha
        
        print(f"Optimal alpha found: {self.optimal_alpha:.6e}")
        
        return self.optimal_alpha
    
    def find_optimal_alpha_gcv(self, 
                             X: np.ndarray, 
                             y: np.ndarray,
                             alpha_range: Tuple[float, float] = (1e-6, 1e2)) -> float:
        """
        Find optimal regularization parameter using Generalized Cross-Validation.
        
        Args:
            X: Training features (detector measurements)
            y: Training targets (hyperspectral data)
            alpha_range: Range of alpha values to search
            
        Returns:
            Optimal alpha value
        """
        def gcv_score(alpha):
            """Compute GCV score for given alpha."""
            ridge = Ridge(alpha=alpha)
            ridge.fit(X, y)
            
            # Prediction
            y_pred = ridge.predict(X)
            
            # Compute hat matrix diagonal
            # H = X(X'X + αI)^(-1)X'
            XtX = X.T @ X
            XtX_reg = XtX + alpha * np.eye(XtX.shape[0])
            try:
                XtX_inv = np.linalg.inv(XtX_reg)
                H_diag = np.sum((X @ XtX_inv) * X, axis=1)
            except np.linalg.LinAlgError:
                return np.inf
            
            # GCV score
            mse = np.mean((y - y_pred) ** 2)
            trace_H = np.sum(H_diag)
            n = len(y)
            
            if trace_H >= n:
                return np.inf
            
            gcv = mse / (1 - trace_H / n) ** 2
            return gcv
        
        # Optimize alpha using GCV
        result = minimize_scalar(gcv_score, bounds=alpha_range, method='bounded')
        
        if result.success:
            self.optimal_alpha = result.x
            print(f"Optimal alpha found via GCV: {self.optimal_alpha:.6e}")
        else:
            print("GCV optimization failed, using default alpha")
            self.optimal_alpha = 1e-3
        
        return self.optimal_alpha
    
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray,
              alpha_selection: str = 'cv',
              cv_folds: int = 5,
              alpha_range: Tuple[float, float] = (1e-6, 1e2)) -> None:
        """
        Train the ridge regression model.
        
        Args:
            X: Training features (detector measurements)
            y: Training targets (hyperspectral data)
            alpha_selection: Method for alpha selection ('cv', 'gcv', 'manual')
            cv_folds: Number of cross-validation folds
            alpha_range: Range of alpha values to search
        """
        # Find optimal alpha
        if alpha_selection == 'cv':
            self.find_optimal_alpha_cv(X, y, alpha_range=alpha_range, cv_folds=cv_folds)
        elif alpha_selection == 'gcv':
            self.find_optimal_alpha_gcv(X, y, alpha_range=alpha_range)
        elif alpha_selection == 'manual':
            self.optimal_alpha = 1e-3  # Default value
        else:
            raise ValueError(f"Unknown alpha selection method: {alpha_selection}")
        
        # Train final model
        self.ridge_model = Ridge(alpha=self.optimal_alpha)
        self.ridge_model.fit(X, y)
        
        # Evaluate training performance
        y_pred = self.ridge_model.predict(X)
        train_mse = mean_squared_error(y, y_pred)
        train_r2 = r2_score(y, y_pred)
        
        self.training_history.update({
            'train_mse': train_mse,
            'train_r2': train_r2,
            'model_coefficients': self.ridge_model.coef_,
            'model_intercept': self.ridge_model.intercept_
        })
        
        print(f"Training completed:")
        print(f"  Train MSE: {train_mse:.6f}")
        print(f"  Train R²: {train_r2:.6f}")
    
    def predict(self, detector_measurements: np.ndarray) -> np.ndarray:
        """
        Predict hyperspectral data from detector measurements.
        
        Args:
            detector_measurements: Detector measurements (num_samples, num_detectors)
            
        Returns:
            Predicted hyperspectral data (num_samples, num_wavelengths)
        """
        if self.ridge_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Store original shape for reshaping
        original_shape = detector_measurements.shape
        if detector_measurements.ndim > 2:
            detector_measurements = detector_measurements.reshape(-1, original_shape[-1])
        
        # Normalize if scaler was used during training
        X = detector_measurements.copy()
        if self.scaler_X is not None:
            X = self.scaler_X.transform(X)
        
        # Predict
        y_pred = self.ridge_model.predict(X)
        
        # Denormalize if scaler was used during training
        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(y_pred)
        
        # Reshape back to original dimensions
        if detector_measurements.ndim > 2:
            output_shape = original_shape[:-1] + (self.num_wavelengths,)
            y_pred = y_pred.reshape(output_shape)
        
        return y_pred
    
    def evaluate_reconstruction(self, 
                              true_spectra: np.ndarray,
                              predicted_spectra: np.ndarray) -> Dict[str, float]:
        """
        Evaluate reconstruction quality.
        
        Args:
            true_spectra: Ground truth hyperspectral data
            predicted_spectra: Reconstructed hyperspectral data
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Reshape if necessary
        if true_spectra.ndim > 2:
            true_spectra = true_spectra.reshape(-1, true_spectra.shape[-1])
            predicted_spectra = predicted_spectra.reshape(-1, predicted_spectra.shape[-1])
        
        # Compute metrics
        mse = mean_squared_error(true_spectra, predicted_spectra)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_spectra, predicted_spectra)
        
        # Spectral Angle Mapper (SAM)
        sam_angles = []
        for i in range(len(true_spectra)):
            true_norm = np.linalg.norm(true_spectra[i])
            pred_norm = np.linalg.norm(predicted_spectra[i])
            if true_norm > 0 and pred_norm > 0:
                cos_angle = np.dot(true_spectra[i], predicted_spectra[i]) / (true_norm * pred_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                sam_angle = np.arccos(cos_angle)
                sam_angles.append(sam_angle)
        
        mean_sam = np.mean(sam_angles) if sam_angles else 0
        
        # Relative error
        relative_error = np.mean(np.abs(true_spectra - predicted_spectra) / 
                               (np.abs(true_spectra) + 1e-8))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'sam_radians': mean_sam,
            'sam_degrees': np.degrees(mean_sam),
            'relative_error': relative_error
        }
    
    def cross_validate_reconstruction(self, 
                                    X: np.ndarray, 
                                    y: np.ndarray,
                                    cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation to assess reconstruction performance.
        
        Args:
            X: Detector measurements
            y: Ground truth hyperspectral data
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_results = {
            'mse_scores': [],
            'r2_scores': [],
            'sam_scores': [],
            'fold_predictions': [],
            'fold_indices': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Processing fold {fold + 1}/{cv_folds}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create temporary model for this fold
            temp_model = Ridge(alpha=self.optimal_alpha if self.optimal_alpha else 1e-3)
            
            # Normalize data for this fold
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_train_norm = scaler_X.fit_transform(X_train)
            y_train_norm = scaler_y.fit_transform(y_train)
            X_val_norm = scaler_X.transform(X_val)
            
            # Train and predict
            temp_model.fit(X_train_norm, y_train_norm)
            y_pred_norm = temp_model.predict(X_val_norm)
            y_pred = scaler_y.inverse_transform(y_pred_norm)
            
            # Evaluate
            metrics = self.evaluate_reconstruction(y_val, y_pred)
            cv_results['mse_scores'].append(metrics['mse'])
            cv_results['r2_scores'].append(metrics['r2'])
            cv_results['sam_scores'].append(metrics['sam_degrees'])
            cv_results['fold_predictions'].append(y_pred)
            cv_results['fold_indices'].append(val_idx)
        
        # Compute statistics
        cv_results['mean_mse'] = np.mean(cv_results['mse_scores'])
        cv_results['std_mse'] = np.std(cv_results['mse_scores'])
        cv_results['mean_r2'] = np.mean(cv_results['r2_scores'])
        cv_results['std_r2'] = np.std(cv_results['r2_scores'])
        cv_results['mean_sam'] = np.mean(cv_results['sam_scores'])
        cv_results['std_sam'] = np.std(cv_results['sam_scores'])
        
        print(f"Cross-validation results:")
        print(f"  MSE: {cv_results['mean_mse']:.6f} ± {cv_results['std_mse']:.6f}")
        print(f"  R²: {cv_results['mean_r2']:.6f} ± {cv_results['std_r2']:.6f}")
        print(f"  SAM: {cv_results['mean_sam']:.3f}° ± {cv_results['std_sam']:.3f}°")
        
        return cv_results
    
    def plot_alpha_selection(self, save_path: Optional[str] = None) -> None:
        """
        Plot alpha selection results.
        
        Args:
            save_path: Optional path to save the plot
        """
        if 'alphas' not in self.training_history:
            print("No alpha selection history available.")
            return
        
        plt.figure(figsize=(10, 6))
        
        alphas = self.training_history['alphas']
        cv_scores = np.mean(self.training_history['cv_scores'], axis=1)
        cv_std = np.std(self.training_history['cv_scores'], axis=1)
        
        plt.semilogx(alphas, -cv_scores, 'b-', label='CV Score')
        plt.fill_between(alphas, -cv_scores - cv_std, -cv_scores + cv_std, alpha=0.3)
        plt.axvline(x=self.optimal_alpha, color='r', linestyle='--', 
                   label=f'Optimal α = {self.optimal_alpha:.2e}')
        
        plt.xlabel('Regularization Parameter (α)')
        plt.ylabel('Cross-Validation Score (MSE)')
        plt.title('Ridge Regression Alpha Selection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, file_path: str) -> None:
        """Save the trained model to file."""
        if self.ridge_model is None:
            raise ValueError("No model to save. Train the model first.")
        
        save_dict = {
            'model_coef': self.ridge_model.coef_,
            'model_intercept': self.ridge_model.intercept_,
            'optimal_alpha': self.optimal_alpha,
            'detector_responses': self.detector_responses,
            'wavelengths': self.wavelengths,
            'training_history': self.training_history
        }
        
        if self.scaler_X is not None:
            save_dict['scaler_X_mean'] = self.scaler_X.mean_
            save_dict['scaler_X_scale'] = self.scaler_X.scale_
        
        if self.scaler_y is not None:
            save_dict['scaler_y_mean'] = self.scaler_y.mean_
            save_dict['scaler_y_scale'] = self.scaler_y.scale_
        
        np.savez_compressed(file_path, **save_dict)
        print(f"Model saved to {file_path}")


if __name__ == "__main__":
    # Example usage
    print("Spectral Reconstructor Test")
    print("=" * 40)
    
    # Generate synthetic test data
    num_wavelengths = 250
    num_detectors = 15
    num_samples = 1000
    
    wavelengths = np.linspace(400, 1000, num_wavelengths)
    
    # Create synthetic detector responses
    center_wavelengths = np.linspace(450, 950, num_detectors)
    detector_responses = np.zeros((num_wavelengths, num_detectors))
    
    for i, center in enumerate(center_wavelengths):
        sigma = 50 / (2 * np.sqrt(2 * np.log(2)))  # FWHM = 50nm
        detector_responses[:, i] = np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)
    
    # Generate synthetic spectra
    true_spectra = np.random.rand(num_samples, num_wavelengths)
    detector_measurements = true_spectra @ detector_responses
    
    # Add noise
    noise = np.random.normal(0, 0.01 * np.std(detector_measurements), detector_measurements.shape)
    detector_measurements += noise
    
    # Create and train reconstructor
    reconstructor = SpectralReconstructor(detector_responses, wavelengths)
    
    # Prepare training data
    X, y = reconstructor.prepare_training_data(true_spectra, detector_measurements)
    
    # Train model
    reconstructor.train(X, y, alpha_selection='cv')
    
    # Test prediction
    predicted_spectra = reconstructor.predict(detector_measurements)
    
    # Evaluate
    metrics = reconstructor.evaluate_reconstruction(true_spectra, predicted_spectra)
    print(f"Reconstruction metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")