"""
Data normalization and scaling utilities for SABR volatility surface modeling.

This module provides comprehensive normalization and scaling utilities for
different types of data (patches, features, targets) with support for
various normalization methods and robust handling of edge cases.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
import warnings
import pickle
from pathlib import Path


@dataclass
class NormalizationConfig:
    """
    Configuration for data normalization.
    
    Attributes:
        method: Normalization method ('standard', 'minmax', 'robust', 'quantile')
        clip_outliers: Whether to clip outliers before normalization
        outlier_percentiles: Percentiles for outlier clipping (lower, upper)
        epsilon: Small value to avoid division by zero
        per_feature: Whether to normalize each feature independently
        handle_nan: How to handle NaN values ('skip', 'zero', 'mean')
    """
    method: str = 'standard'
    clip_outliers: bool = False
    outlier_percentiles: Tuple[float, float] = (1.0, 99.0)
    epsilon: float = 1e-8
    per_feature: bool = True
    handle_nan: str = 'skip'
    
    def __post_init__(self):
        """Validate configuration."""
        valid_methods = ['standard', 'minmax', 'robust', 'quantile']
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
            
        valid_nan_handling = ['skip', 'zero', 'mean']
        if self.handle_nan not in valid_nan_handling:
            raise ValueError(f"handle_nan must be one of {valid_nan_handling}")
            
        if not (0 <= self.outlier_percentiles[0] < self.outlier_percentiles[1] <= 100):
            raise ValueError("outlier_percentiles must be in range [0, 100] with lower < upper")


@dataclass
class NormalizationStats:
    """
    Statistics for data normalization and denormalization.
    
    Attributes:
        method: Normalization method used
        means: Mean values for standard normalization
        stds: Standard deviations for standard normalization
        mins: Minimum values for minmax normalization
        maxs: Maximum values for minmax normalization
        medians: Median values for robust normalization
        mads: Median absolute deviations for robust normalization
        quantiles: Quantile values for quantile normalization
        clip_bounds: Bounds used for outlier clipping
        shape: Original data shape
        n_samples: Number of samples used for fitting
    """
    method: str
    means: Optional[np.ndarray] = None
    stds: Optional[np.ndarray] = None
    mins: Optional[np.ndarray] = None
    maxs: Optional[np.ndarray] = None
    medians: Optional[np.ndarray] = None
    mads: Optional[np.ndarray] = None
    quantiles: Optional[Dict[str, np.ndarray]] = None
    clip_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    shape: Optional[Tuple[int, ...]] = None
    n_samples: Optional[int] = None


class DataNormalizer:
    """
    Comprehensive data normalizer supporting multiple normalization methods.
    
    Supports normalization of patches, features, and target values with
    robust handling of edge cases and proper statistics tracking.
    """
    
    def __init__(self, config: NormalizationConfig = None):
        """
        Initialize data normalizer.
        
        Args:
            config: Normalization configuration
        """
        self.config = config or NormalizationConfig()
        self.stats = None
        self.is_fitted = False
        
    def fit(self, data: np.ndarray) -> 'DataNormalizer':
        """
        Fit normalization parameters to data.
        
        Args:
            data: Input data array of any shape
            
        Returns:
            Self for method chaining
        """
        if data.size == 0:
            raise ValueError("Cannot fit on empty data")
            
        # Handle NaN values
        clean_data = self._handle_nan_values(data)
        
        # Clip outliers if requested
        if self.config.clip_outliers:
            clean_data, clip_bounds = self._clip_outliers(clean_data)
        else:
            clip_bounds = None
            
        # Compute normalization statistics
        self.stats = NormalizationStats(
            method=self.config.method,
            shape=data.shape,
            n_samples=data.shape[0] if data.ndim > 0 else 1,
            clip_bounds=clip_bounds
        )
        
        if self.config.method == 'standard':
            self._fit_standard(clean_data)
        elif self.config.method == 'minmax':
            self._fit_minmax(clean_data)
        elif self.config.method == 'robust':
            self._fit_robust(clean_data)
        elif self.config.method == 'quantile':
            self._fit_quantile(clean_data)
            
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted normalization parameters.
        
        Args:
            data: Input data to normalize
            
        Returns:
            Normalized data
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before transform()")
            
        # Handle NaN values
        clean_data = self._handle_nan_values(data)
        
        # Apply outlier clipping if it was used during fitting
        if self.stats.clip_bounds is not None:
            clean_data = np.clip(clean_data, 
                               self.stats.clip_bounds[0], 
                               self.stats.clip_bounds[1])
        
        # Apply normalization
        if self.config.method == 'standard':
            return self._transform_standard(clean_data)
        elif self.config.method == 'minmax':
            return self._transform_minmax(clean_data)
        elif self.config.method == 'robust':
            return self._transform_robust(clean_data)
        elif self.config.method == 'quantile':
            return self._transform_quantile(clean_data)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit normalization parameters and transform data in one step.
        
        Args:
            data: Input data
            
        Returns:
            Normalized data
        """
        return self.fit(data).transform(data)
    
    def inverse_transform(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            normalized_data: Normalized data
            
        Returns:
            Data in original scale
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before inverse_transform()")
            
        if self.config.method == 'standard':
            return self._inverse_transform_standard(normalized_data)
        elif self.config.method == 'minmax':
            return self._inverse_transform_minmax(normalized_data)
        elif self.config.method == 'robust':
            return self._inverse_transform_robust(normalized_data)
        elif self.config.method == 'quantile':
            return self._inverse_transform_quantile(normalized_data)
    
    def _handle_nan_values(self, data: np.ndarray) -> np.ndarray:
        """Handle NaN values according to configuration."""
        if not np.any(np.isnan(data)):
            return data
            
        if self.config.handle_nan == 'skip':
            return data  # Keep NaNs, handle in normalization methods
        elif self.config.handle_nan == 'zero':
            return np.nan_to_num(data, nan=0.0)
        elif self.config.handle_nan == 'mean':
            if self.is_fitted and self.stats.means is not None:
                # Use fitted means for replacement
                mean_vals = self.stats.means
            else:
                # Compute means from current data
                mean_vals = np.nanmean(data, axis=0, keepdims=True)
                if data.ndim > 1:
                    mean_vals = np.broadcast_to(mean_vals, data.shape[1:])
                else:
                    mean_vals = np.nanmean(data)
            
            result = data.copy()
            result[np.isnan(result)] = mean_vals
            return result
        
        return data
    
    def _clip_outliers(self, data: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Clip outliers based on percentiles."""
        lower_p, upper_p = self.config.outlier_percentiles
        
        if self.config.per_feature and data.ndim > 1:
            # Compute percentiles per feature
            lower_bounds = np.nanpercentile(data, lower_p, axis=0)
            upper_bounds = np.nanpercentile(data, upper_p, axis=0)
        else:
            # Global percentiles
            lower_bounds = np.nanpercentile(data, lower_p)
            upper_bounds = np.nanpercentile(data, upper_p)
        
        clipped_data = np.clip(data, lower_bounds, upper_bounds)
        return clipped_data, (lower_bounds, upper_bounds)
    
    def _fit_standard(self, data: np.ndarray):
        """Fit standard (z-score) normalization."""
        if self.config.per_feature and data.ndim > 1:
            axis = 0
        else:
            axis = None
            
        self.stats.means = np.nanmean(data, axis=axis)
        self.stats.stds = np.nanstd(data, axis=axis)
        
        # Avoid division by zero
        self.stats.stds = np.where(
            self.stats.stds < self.config.epsilon,
            1.0,
            self.stats.stds
        )
    
    def _fit_minmax(self, data: np.ndarray):
        """Fit min-max normalization."""
        if self.config.per_feature and data.ndim > 1:
            axis = 0
        else:
            axis = None
            
        self.stats.mins = np.nanmin(data, axis=axis)
        self.stats.maxs = np.nanmax(data, axis=axis)
        
        # Avoid division by zero
        ranges = self.stats.maxs - self.stats.mins
        ranges = np.where(ranges < self.config.epsilon, 1.0, ranges)
        self.stats.maxs = self.stats.mins + ranges
    
    def _fit_robust(self, data: np.ndarray):
        """Fit robust (median-based) normalization."""
        if self.config.per_feature and data.ndim > 1:
            axis = 0
        else:
            axis = None
            
        self.stats.medians = np.nanmedian(data, axis=axis)
        
        # Compute median absolute deviation
        deviations = np.abs(data - self.stats.medians)
        self.stats.mads = np.nanmedian(deviations, axis=axis)
        
        # Avoid division by zero
        self.stats.mads = np.where(
            self.stats.mads < self.config.epsilon,
            1.0,
            self.stats.mads
        )
    
    def _fit_quantile(self, data: np.ndarray):
        """Fit quantile normalization."""
        if self.config.per_feature and data.ndim > 1:
            axis = 0
        else:
            axis = None
            
        # Compute quantiles for transformation
        quantile_levels = np.linspace(0, 100, 1001)  # 0.1% resolution
        
        if axis is None:
            quantiles = np.nanpercentile(data.flatten(), quantile_levels)
            self.stats.quantiles = {'levels': quantile_levels, 'values': quantiles}
        else:
            quantiles = np.nanpercentile(data, quantile_levels, axis=axis)
            self.stats.quantiles = {'levels': quantile_levels, 'values': quantiles}
    
    def _transform_standard(self, data: np.ndarray) -> np.ndarray:
        """Apply standard normalization."""
        return (data - self.stats.means) / self.stats.stds
    
    def _transform_minmax(self, data: np.ndarray) -> np.ndarray:
        """Apply min-max normalization."""
        return (data - self.stats.mins) / (self.stats.maxs - self.stats.mins)
    
    def _transform_robust(self, data: np.ndarray) -> np.ndarray:
        """Apply robust normalization."""
        return (data - self.stats.medians) / self.stats.mads
    
    def _transform_quantile(self, data: np.ndarray) -> np.ndarray:
        """Apply quantile normalization."""
        from scipy.interpolate import interp1d
        
        levels = self.stats.quantiles['levels']
        values = self.stats.quantiles['values']
        
        if data.ndim == 1 or not self.config.per_feature:
            # Global quantile transformation
            interpolator = interp1d(values, levels, 
                                  bounds_error=False, 
                                  fill_value=(levels[0], levels[-1]))
            return interpolator(data.flatten()).reshape(data.shape) / 100.0
        else:
            # Per-feature quantile transformation
            result = np.zeros_like(data)
            for i in range(data.shape[1]):
                interpolator = interp1d(values[:, i], levels,
                                      bounds_error=False,
                                      fill_value=(levels[0], levels[-1]))
                result[:, i] = interpolator(data[:, i]) / 100.0
            return result
    
    def _inverse_transform_standard(self, normalized_data: np.ndarray) -> np.ndarray:
        """Inverse standard normalization."""
        return normalized_data * self.stats.stds + self.stats.means
    
    def _inverse_transform_minmax(self, normalized_data: np.ndarray) -> np.ndarray:
        """Inverse min-max normalization."""
        return normalized_data * (self.stats.maxs - self.stats.mins) + self.stats.mins
    
    def _inverse_transform_robust(self, normalized_data: np.ndarray) -> np.ndarray:
        """Inverse robust normalization."""
        return normalized_data * self.stats.mads + self.stats.medians
    
    def _inverse_transform_quantile(self, normalized_data: np.ndarray) -> np.ndarray:
        """Inverse quantile normalization."""
        from scipy.interpolate import interp1d
        
        levels = self.stats.quantiles['levels']
        values = self.stats.quantiles['values']
        
        # Convert back from [0,1] to percentile scale
        percentiles = normalized_data * 100.0
        
        if normalized_data.ndim == 1 or not self.config.per_feature:
            # Global inverse transformation
            interpolator = interp1d(levels, values,
                                  bounds_error=False,
                                  fill_value=(values[0], values[-1]))
            return interpolator(percentiles.flatten()).reshape(normalized_data.shape)
        else:
            # Per-feature inverse transformation
            result = np.zeros_like(normalized_data)
            for i in range(normalized_data.shape[1]):
                interpolator = interp1d(levels, values[:, i],
                                      bounds_error=False,
                                      fill_value=(values[0, i], values[-1, i]))
                result[:, i] = interpolator(percentiles[:, i])
            return result
    
    def save(self, filepath: str):
        """
        Save normalization statistics to file.
        
        Args:
            filepath: Path to save statistics
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted normalizer")
            
        save_data = {
            'config': self.config,
            'stats': self.stats,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, filepath: str):
        """
        Load normalization statistics from file.
        
        Args:
            filepath: Path to load statistics from
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
            
        self.config = save_data['config']
        self.stats = save_data['stats']
        self.is_fitted = save_data['is_fitted']
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """
        Get summary of normalization statistics.
        
        Returns:
            Dictionary with statistics summary
        """
        if not self.is_fitted:
            return {'error': 'Normalizer not fitted'}
            
        summary = {
            'method': self.stats.method,
            'n_samples': self.stats.n_samples,
            'shape': self.stats.shape,
            'clip_outliers': self.config.clip_outliers
        }
        
        if self.stats.means is not None:
            summary['means'] = {
                'min': float(np.min(self.stats.means)),
                'max': float(np.max(self.stats.means)),
                'mean': float(np.mean(self.stats.means))
            }
            
        if self.stats.stds is not None:
            summary['stds'] = {
                'min': float(np.min(self.stats.stds)),
                'max': float(np.max(self.stats.stds)),
                'mean': float(np.mean(self.stats.stds))
            }
            
        if self.stats.clip_bounds is not None:
            summary['clip_bounds'] = {
                'lower': self.stats.clip_bounds[0].tolist() if hasattr(self.stats.clip_bounds[0], 'tolist') else float(self.stats.clip_bounds[0]),
                'upper': self.stats.clip_bounds[1].tolist() if hasattr(self.stats.clip_bounds[1], 'tolist') else float(self.stats.clip_bounds[1])
            }
        
        return summary


class PatchNormalizer(DataNormalizer):
    """
    Specialized normalizer for surface patches.
    
    Handles patch-specific normalization with support for spatial
    normalization and boundary handling.
    """
    
    def __init__(self, config: NormalizationConfig = None, 
                 spatial_normalization: bool = False):
        """
        Initialize patch normalizer.
        
        Args:
            config: Normalization configuration
            spatial_normalization: Whether to normalize spatially within patches
        """
        super().__init__(config)
        self.spatial_normalization = spatial_normalization
    
    def fit(self, patches: np.ndarray) -> 'PatchNormalizer':
        """
        Fit normalization to patch data.
        
        Args:
            patches: Array of patches (n_samples, height, width)
            
        Returns:
            Self for method chaining
        """
        if patches.ndim != 3:
            raise ValueError("Patches must be 3D array (n_samples, height, width)")
            
        if self.spatial_normalization:
            # Normalize each patch individually (spatial normalization)
            normalized_patches = np.zeros_like(patches)
            for i in range(patches.shape[0]):
                patch = patches[i]
                if not np.all(np.isnan(patch)):
                    patch_normalizer = DataNormalizer(self.config)
                    normalized_patches[i] = patch_normalizer.fit_transform(patch)
                else:
                    normalized_patches[i] = patch
            
            # Fit global statistics on spatially normalized patches
            super().fit(normalized_patches.reshape(patches.shape[0], -1))
        else:
            # Global normalization across all patches
            super().fit(patches.reshape(patches.shape[0], -1))
        
        return self
    
    def transform(self, patches: np.ndarray) -> np.ndarray:
        """
        Transform patches using fitted normalization.
        
        Args:
            patches: Array of patches to normalize
            
        Returns:
            Normalized patches
        """
        if patches.ndim != 3:
            raise ValueError("Patches must be 3D array (n_samples, height, width)")
            
        original_shape = patches.shape
        
        if self.spatial_normalization:
            # Apply spatial normalization first
            spatially_normalized = np.zeros_like(patches)
            for i in range(patches.shape[0]):
                patch = patches[i]
                if not np.all(np.isnan(patch)):
                    patch_normalizer = DataNormalizer(self.config)
                    spatially_normalized[i] = patch_normalizer.fit_transform(patch)
                else:
                    spatially_normalized[i] = patch
            
            # Apply global normalization
            flattened = spatially_normalized.reshape(patches.shape[0], -1)
            normalized_flat = super().transform(flattened)
            return normalized_flat.reshape(original_shape)
        else:
            # Global normalization only
            flattened = patches.reshape(patches.shape[0], -1)
            normalized_flat = super().transform(flattened)
            return normalized_flat.reshape(original_shape)
    
    def inverse_transform(self, normalized_patches: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized patches back to original scale.
        
        Args:
            normalized_patches: Normalized patches
            
        Returns:
            Patches in original scale
        """
        if normalized_patches.ndim != 3:
            raise ValueError("Patches must be 3D array (n_samples, height, width)")
            
        original_shape = normalized_patches.shape
        
        if self.spatial_normalization:
            # This is complex for spatial normalization since each patch was normalized individually
            # For now, we'll apply global inverse transform and warn
            warnings.warn("Inverse transform with spatial normalization may not be exact")
            
        # Apply global inverse normalization
        flattened = normalized_patches.reshape(normalized_patches.shape[0], -1)
        inverse_flat = super().inverse_transform(flattened)
        return inverse_flat.reshape(original_shape)


class TargetNormalizer(DataNormalizer):
    """
    Specialized normalizer for target residuals.
    
    Handles target-specific normalization with support for
    different residual distributions and outlier handling.
    """
    
    def __init__(self, config: NormalizationConfig = None,
                 symmetric_bounds: bool = True):
        """
        Initialize target normalizer.
        
        Args:
            config: Normalization configuration
            symmetric_bounds: Whether to use symmetric bounds for clipping
        """
        super().__init__(config)
        self.symmetric_bounds = symmetric_bounds
    
    def fit(self, targets: np.ndarray) -> 'TargetNormalizer':
        """
        Fit normalization to target data.
        
        Args:
            targets: Array of target values
            
        Returns:
            Self for method chaining
        """
        if self.symmetric_bounds and self.config.clip_outliers:
            # Use symmetric bounds for residuals
            abs_percentile = max(self.config.outlier_percentiles[0], 
                               100 - self.config.outlier_percentiles[1])
            lower_bound = np.nanpercentile(targets, abs_percentile)
            upper_bound = np.nanpercentile(targets, 100 - abs_percentile)
            
            # Make bounds symmetric around zero
            max_abs = max(abs(lower_bound), abs(upper_bound))
            symmetric_targets = np.clip(targets, -max_abs, max_abs)
            
            super().fit(symmetric_targets)
            
            # Update clip bounds to be symmetric
            if self.stats.clip_bounds is not None:
                self.stats.clip_bounds = (-max_abs, max_abs)
        else:
            super().fit(targets)
        
        return self


def create_normalization_pipeline(patch_config: NormalizationConfig = None,
                                feature_config: NormalizationConfig = None,
                                target_config: NormalizationConfig = None) -> Dict[str, DataNormalizer]:
    """
    Create a complete normalization pipeline for SABR data.
    
    Args:
        patch_config: Configuration for patch normalization
        feature_config: Configuration for feature normalization
        target_config: Configuration for target normalization
        
    Returns:
        Dictionary with normalizers for each data type
    """
    normalizers = {
        'patches': PatchNormalizer(patch_config),
        'features': DataNormalizer(feature_config),
        'targets': TargetNormalizer(target_config)
    }
    
    return normalizers


def validate_normalization(original_data: np.ndarray, 
                         normalized_data: np.ndarray,
                         normalizer: DataNormalizer,
                         tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Validate normalization by checking inverse transformation.
    
    Args:
        original_data: Original data before normalization
        normalized_data: Data after normalization
        normalizer: Fitted normalizer
        tolerance: Tolerance for inverse transformation check
        
    Returns:
        Dictionary with validation results
    """
    try:
        # Test inverse transformation
        reconstructed = normalizer.inverse_transform(normalized_data)
        
        # Compute reconstruction error
        valid_mask = ~(np.isnan(original_data) | np.isnan(reconstructed))
        if np.any(valid_mask):
            error = np.abs(original_data[valid_mask] - reconstructed[valid_mask])
            max_error = np.max(error)
            mean_error = np.mean(error)
        else:
            max_error = 0.0
            mean_error = 0.0
        
        # Check normalization properties
        valid_normalized = normalized_data[~np.isnan(normalized_data)]
        
        validation_results = {
            'inverse_transform_valid': max_error < tolerance,
            'max_reconstruction_error': float(max_error),
            'mean_reconstruction_error': float(mean_error),
            'normalized_mean': float(np.mean(valid_normalized)) if len(valid_normalized) > 0 else 0.0,
            'normalized_std': float(np.std(valid_normalized)) if len(valid_normalized) > 0 else 0.0,
            'n_valid_samples': int(np.sum(valid_mask)),
            'n_nan_original': int(np.sum(np.isnan(original_data))),
            'n_nan_normalized': int(np.sum(np.isnan(normalized_data)))
        }
        
        # Method-specific checks
        if normalizer.config.method == 'standard':
            expected_mean = 0.0
            expected_std = 1.0
            validation_results['mean_close_to_zero'] = abs(validation_results['normalized_mean']) < 0.1
            validation_results['std_close_to_one'] = abs(validation_results['normalized_std'] - 1.0) < 0.1
            
        elif normalizer.config.method == 'minmax':
            validation_results['in_unit_range'] = (
                np.all(valid_normalized >= -0.1) and np.all(valid_normalized <= 1.1)
            )
        
        return validation_results
        
    except Exception as e:
        return {
            'error': str(e),
            'inverse_transform_valid': False
        }