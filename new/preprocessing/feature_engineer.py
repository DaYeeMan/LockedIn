"""
Feature engineering module for MDA-CNN volatility surface modeling.

This module implements the FeatureEngineer class that creates and normalizes
point features for the MLP branch of the MDA-CNN architecture.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

from data_generation.sabr_params import SABRParams


@dataclass
class FeatureConfig:
    """
    Configuration for feature engineering.
    
    Attributes:
        include_sabr_params: Whether to include SABR parameters as features
        include_moneyness: Whether to include moneyness (K/F) as feature
        include_log_moneyness: Whether to include log-moneyness ln(K/F) as feature
        include_time_to_maturity: Whether to include time to maturity as feature
        include_hagan_volatility: Whether to include Hagan LF volatility as feature
        include_derived_features: Whether to include derived features (interactions, etc.)
        normalize_features: Whether to apply normalization to features
        normalization_method: Method for normalization ('standard', 'minmax', 'robust')
        feature_selection: List of specific features to include (None = all enabled)
    """
    include_sabr_params: bool = True
    include_moneyness: bool = True
    include_log_moneyness: bool = True
    include_time_to_maturity: bool = True
    include_hagan_volatility: bool = True
    include_derived_features: bool = True
    normalize_features: bool = True
    normalization_method: str = 'standard'
    feature_selection: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate feature configuration."""
        valid_methods = ['standard', 'minmax', 'robust']
        if self.normalization_method not in valid_methods:
            raise ValueError(f"normalization_method must be one of {valid_methods}")


@dataclass
class PointFeatures:
    """
    Point features for a single data point.
    
    Attributes:
        raw_features: Dictionary of raw feature values
        normalized_features: Normalized feature array (if normalization applied)
        feature_names: Names of features in order
        sabr_params: Associated SABR parameters
        strike: Strike price
        maturity: Time to maturity
        hagan_volatility: Hagan LF volatility (if available)
    """
    raw_features: Dict[str, float]
    normalized_features: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    sabr_params: Optional[SABRParams] = None
    strike: Optional[float] = None
    maturity: Optional[float] = None
    hagan_volatility: Optional[float] = None


@dataclass
class NormalizationStats:
    """
    Statistics for feature normalization.
    
    Attributes:
        method: Normalization method used
        feature_names: Names of features
        means: Mean values for each feature (standard normalization)
        stds: Standard deviations for each feature (standard normalization)
        mins: Minimum values for each feature (minmax normalization)
        maxs: Maximum values for each feature (minmax normalization)
        medians: Median values for each feature (robust normalization)
        mads: Median absolute deviations for each feature (robust normalization)
    """
    method: str
    feature_names: List[str]
    means: Optional[np.ndarray] = None
    stds: Optional[np.ndarray] = None
    mins: Optional[np.ndarray] = None
    maxs: Optional[np.ndarray] = None
    medians: Optional[np.ndarray] = None
    mads: Optional[np.ndarray] = None


class FeatureEngineer:
    """
    Create and normalize point features for MLP input.
    
    This class handles the creation of point features from SABR parameters,
    strike/maturity information, and Hagan volatilities. It supports various
    normalization methods and feature selection strategies.
    """
    
    def __init__(self, config: FeatureConfig = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config or FeatureConfig()
        self.normalization_stats = None
        self.is_fitted = False
        
    def create_point_features(self, sabr_params: SABRParams, 
                            strike: float, 
                            maturity: float,
                            hagan_volatility: Optional[float] = None) -> PointFeatures:
        """
        Create point features for a single data point.
        
        Args:
            sabr_params: SABR model parameters
            strike: Strike price
            maturity: Time to maturity
            hagan_volatility: Hagan LF volatility (optional)
            
        Returns:
            PointFeatures object containing raw and normalized features
        """
        raw_features = {}
        
        # SABR parameters
        if self.config.include_sabr_params:
            raw_features['F0'] = sabr_params.F0
            raw_features['alpha'] = sabr_params.alpha
            raw_features['beta'] = sabr_params.beta
            raw_features['nu'] = sabr_params.nu
            raw_features['rho'] = sabr_params.rho
        
        # Strike and maturity features
        raw_features['strike'] = strike
        raw_features['maturity'] = maturity
        
        # Moneyness features
        if self.config.include_moneyness:
            raw_features['moneyness'] = strike / sabr_params.F0
        
        if self.config.include_log_moneyness:
            if strike > 0 and sabr_params.F0 > 0:
                raw_features['log_moneyness'] = np.log(strike / sabr_params.F0)
            else:
                raw_features['log_moneyness'] = 0.0
                warnings.warn("Invalid strike or F0 for log_moneyness calculation")
        
        # Time to maturity
        if self.config.include_time_to_maturity:
            raw_features['time_to_maturity'] = maturity
        
        # Hagan volatility
        if self.config.include_hagan_volatility and hagan_volatility is not None:
            raw_features['hagan_volatility'] = hagan_volatility
        
        # Derived features
        if self.config.include_derived_features:
            raw_features.update(self._create_derived_features(sabr_params, strike, maturity))
        
        # Apply feature selection if specified
        if self.config.feature_selection is not None:
            selected_features = {k: v for k, v in raw_features.items() 
                               if k in self.config.feature_selection}
            raw_features = selected_features
        
        # Create PointFeatures object
        point_features = PointFeatures(
            raw_features=raw_features,
            sabr_params=sabr_params,
            strike=strike,
            maturity=maturity,
            hagan_volatility=hagan_volatility
        )
        
        # Apply normalization if fitted
        if self.is_fitted and self.config.normalize_features:
            point_features.normalized_features = self._normalize_single_point(raw_features)
            point_features.feature_names = self.normalization_stats.feature_names
        
        return point_features
    
    def _create_derived_features(self, sabr_params: SABRParams, 
                               strike: float, 
                               maturity: float) -> Dict[str, float]:
        """
        Create derived features from basic inputs.
        
        Args:
            sabr_params: SABR parameters
            strike: Strike price
            maturity: Time to maturity
            
        Returns:
            Dictionary of derived features
        """
        derived = {}
        
        # Volatility-related features
        derived['alpha_sqrt_T'] = sabr_params.alpha * np.sqrt(maturity)
        derived['nu_sqrt_T'] = sabr_params.nu * np.sqrt(maturity)
        
        # Interaction terms
        derived['alpha_nu'] = sabr_params.alpha * sabr_params.nu
        derived['rho_nu'] = sabr_params.rho * sabr_params.nu
        derived['beta_alpha'] = sabr_params.beta * sabr_params.alpha
        
        # Moneyness-related features
        moneyness = strike / sabr_params.F0
        derived['moneyness_squared'] = moneyness ** 2
        derived['moneyness_beta'] = moneyness ** sabr_params.beta
        
        # Time-related features
        derived['maturity_squared'] = maturity ** 2
        derived['sqrt_maturity'] = np.sqrt(maturity)
        
        # SABR-specific derived features
        if sabr_params.beta != 1.0:
            FK = sabr_params.F0 * strike
            if FK > 0:
                derived['FK_power_beta_minus_1'] = FK ** (sabr_params.beta - 1)
        
        # Volatility of volatility scaled by time
        derived['nu_T'] = sabr_params.nu * maturity
        
        # Correlation effects
        derived['rho_squared'] = sabr_params.rho ** 2
        derived['one_minus_rho_squared'] = 1 - sabr_params.rho ** 2
        
        return derived
    
    def fit_normalization(self, features_list: List[PointFeatures]) -> None:
        """
        Fit normalization parameters from a list of point features.
        
        Args:
            features_list: List of PointFeatures objects to fit normalization on
        """
        if not features_list:
            raise ValueError("Cannot fit normalization on empty feature list")
        
        # Extract feature matrices
        feature_names = list(features_list[0].raw_features.keys())
        n_features = len(feature_names)
        n_samples = len(features_list)
        
        feature_matrix = np.zeros((n_samples, n_features))
        
        for i, point_features in enumerate(features_list):
            for j, feature_name in enumerate(feature_names):
                if feature_name in point_features.raw_features:
                    feature_matrix[i, j] = point_features.raw_features[feature_name]
                else:
                    feature_matrix[i, j] = 0.0
                    warnings.warn(f"Feature {feature_name} missing in sample {i}")
        
        # Compute normalization statistics
        self.normalization_stats = NormalizationStats(
            method=self.config.normalization_method,
            feature_names=feature_names
        )
        
        if self.config.normalization_method == 'standard':
            self.normalization_stats.means = np.mean(feature_matrix, axis=0)
            self.normalization_stats.stds = np.std(feature_matrix, axis=0)
            
            # Avoid division by zero
            self.normalization_stats.stds = np.where(
                self.normalization_stats.stds < 1e-12, 
                1.0, 
                self.normalization_stats.stds
            )
            
        elif self.config.normalization_method == 'minmax':
            self.normalization_stats.mins = np.min(feature_matrix, axis=0)
            self.normalization_stats.maxs = np.max(feature_matrix, axis=0)
            
            # Avoid division by zero
            ranges = self.normalization_stats.maxs - self.normalization_stats.mins
            ranges = np.where(ranges < 1e-12, 1.0, ranges)
            self.normalization_stats.maxs = self.normalization_stats.mins + ranges
            
        elif self.config.normalization_method == 'robust':
            self.normalization_stats.medians = np.median(feature_matrix, axis=0)
            
            # Compute median absolute deviation
            mads = np.median(np.abs(feature_matrix - self.normalization_stats.medians), axis=0)
            self.normalization_stats.mads = np.where(mads < 1e-12, 1.0, mads)
        
        self.is_fitted = True
    
    def _normalize_single_point(self, raw_features: Dict[str, float]) -> np.ndarray:
        """
        Normalize features for a single point using fitted statistics.
        
        Args:
            raw_features: Dictionary of raw feature values
            
        Returns:
            Normalized feature array
        """
        if not self.is_fitted:
            raise ValueError("Must call fit_normalization before normalizing features")
        
        # Convert to array in correct order
        feature_array = np.zeros(len(self.normalization_stats.feature_names))
        
        for i, feature_name in enumerate(self.normalization_stats.feature_names):
            if feature_name in raw_features:
                feature_array[i] = raw_features[feature_name]
            else:
                feature_array[i] = 0.0
                warnings.warn(f"Feature {feature_name} missing, using 0.0")
        
        # Apply normalization
        if self.config.normalization_method == 'standard':
            normalized = (feature_array - self.normalization_stats.means) / self.normalization_stats.stds
            
        elif self.config.normalization_method == 'minmax':
            normalized = (feature_array - self.normalization_stats.mins) / \
                        (self.normalization_stats.maxs - self.normalization_stats.mins)
            
        elif self.config.normalization_method == 'robust':
            normalized = (feature_array - self.normalization_stats.medians) / self.normalization_stats.mads
        
        return normalized
    
    def normalize_features_batch(self, features_list: List[PointFeatures]) -> np.ndarray:
        """
        Normalize features for a batch of points.
        
        Args:
            features_list: List of PointFeatures objects
            
        Returns:
            2D array of normalized features (n_samples x n_features)
        """
        if not self.is_fitted:
            raise ValueError("Must call fit_normalization before normalizing features")
        
        n_samples = len(features_list)
        n_features = len(self.normalization_stats.feature_names)
        
        normalized_batch = np.zeros((n_samples, n_features))
        
        for i, point_features in enumerate(features_list):
            normalized_batch[i] = self._normalize_single_point(point_features.raw_features)
            
            # Update the point features object
            point_features.normalized_features = normalized_batch[i]
            point_features.feature_names = self.normalization_stats.feature_names
        
        return normalized_batch
    
    def create_features_batch(self, sabr_params_list: List[SABRParams],
                            strikes: List[float],
                            maturities: List[float],
                            hagan_volatilities: Optional[List[float]] = None) -> List[PointFeatures]:
        """
        Create features for a batch of data points.
        
        Args:
            sabr_params_list: List of SABR parameters
            strikes: List of strike prices
            maturities: List of maturities
            hagan_volatilities: List of Hagan volatilities (optional)
            
        Returns:
            List of PointFeatures objects
        """
        if not (len(sabr_params_list) == len(strikes) == len(maturities)):
            raise ValueError("All input lists must have the same length")
        
        if hagan_volatilities is not None and len(hagan_volatilities) != len(strikes):
            raise ValueError("hagan_volatilities must have same length as other inputs")
        
        features_list = []
        
        for i, (sabr_params, strike, maturity) in enumerate(zip(sabr_params_list, strikes, maturities)):
            hagan_vol = hagan_volatilities[i] if hagan_volatilities is not None else None
            
            point_features = self.create_point_features(
                sabr_params, strike, maturity, hagan_vol
            )
            features_list.append(point_features)
        
        return features_list
    
    def get_feature_importance_analysis(self, features_list: List[PointFeatures]) -> Dict[str, Any]:
        """
        Analyze feature importance and correlations.
        
        Args:
            features_list: List of PointFeatures objects
            
        Returns:
            Dictionary with feature analysis results
        """
        if not features_list:
            return {'error': 'No features provided'}
        
        # Extract feature matrix
        feature_names = list(features_list[0].raw_features.keys())
        n_features = len(feature_names)
        n_samples = len(features_list)
        
        feature_matrix = np.zeros((n_samples, n_features))
        
        for i, point_features in enumerate(features_list):
            for j, feature_name in enumerate(feature_names):
                feature_matrix[i, j] = point_features.raw_features.get(feature_name, 0.0)
        
        # Basic statistics
        feature_stats = {}
        for j, feature_name in enumerate(feature_names):
            values = feature_matrix[:, j]
            feature_stats[feature_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'range': float(np.max(values) - np.min(values))
            }
        
        # Correlation matrix
        correlation_matrix = np.corrcoef(feature_matrix.T)
        
        # Find highly correlated features
        high_correlations = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = correlation_matrix[i, j]
                if abs(corr) > 0.8:  # High correlation threshold
                    high_correlations.append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'correlation': float(corr)
                    })
        
        # Feature variance analysis
        variances = np.var(feature_matrix, axis=0)
        low_variance_features = [
            feature_names[i] for i, var in enumerate(variances) if var < 1e-6
        ]
        
        analysis = {
            'feature_count': n_features,
            'sample_count': n_samples,
            'feature_statistics': feature_stats,
            'correlation_matrix': correlation_matrix.tolist(),
            'high_correlations': high_correlations,
            'low_variance_features': low_variance_features,
            'feature_names': feature_names
        }
        
        return analysis
    
    def validate_features(self, features_list: List[PointFeatures]) -> Dict[str, Any]:
        """
        Validate feature quality and consistency.
        
        Args:
            features_list: List of PointFeatures objects
            
        Returns:
            Dictionary with validation results
        """
        if not features_list:
            return {'valid': False, 'error': 'No features provided'}
        
        validation_results = {
            'valid': True,
            'feature_count': len(features_list),
            'consistency_check': True,
            'missing_values': 0,
            'infinite_values': 0,
            'warnings': []
        }
        
        # Check feature consistency
        first_features = set(features_list[0].raw_features.keys())
        
        for i, point_features in enumerate(features_list):
            current_features = set(point_features.raw_features.keys())
            if current_features != first_features:
                validation_results['consistency_check'] = False
                validation_results['warnings'].append(
                    f"Feature set inconsistency at sample {i}: "
                    f"expected {first_features}, got {current_features}"
                )
        
        # Check for missing and infinite values
        for point_features in features_list:
            for feature_name, value in point_features.raw_features.items():
                if np.isnan(value):
                    validation_results['missing_values'] += 1
                elif np.isinf(value):
                    validation_results['infinite_values'] += 1
        
        # Check normalization consistency
        if self.is_fitted:
            normalized_count = sum(1 for pf in features_list if pf.normalized_features is not None)
            if normalized_count != len(features_list) and normalized_count != 0:
                validation_results['warnings'].append(
                    f"Inconsistent normalization: {normalized_count}/{len(features_list)} features normalized"
                )
        
        # Overall validation
        if validation_results['missing_values'] > 0:
            validation_results['warnings'].append(f"Found {validation_results['missing_values']} missing values")
        
        if validation_results['infinite_values'] > 0:
            validation_results['warnings'].append(f"Found {validation_results['infinite_values']} infinite values")
            validation_results['valid'] = False
        
        if not validation_results['consistency_check']:
            validation_results['valid'] = False
        
        return validation_results
    
    def save_normalization_stats(self, filepath: str) -> None:
        """
        Save normalization statistics to file.
        
        Args:
            filepath: Path to save normalization statistics
        """
        if not self.is_fitted:
            raise ValueError("No normalization statistics to save")
        
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.normalization_stats, f)
    
    def load_normalization_stats(self, filepath: str) -> None:
        """
        Load normalization statistics from file.
        
        Args:
            filepath: Path to load normalization statistics from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            self.normalization_stats = pickle.load(f)
        
        self.is_fitted = True