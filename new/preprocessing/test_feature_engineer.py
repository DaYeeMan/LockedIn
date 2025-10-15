"""
Tests for feature engineering functionality.
"""

import pytest
import numpy as np
from typing import List
import tempfile
import os

from preprocessing.feature_engineer import (
    FeatureEngineer, FeatureConfig, PointFeatures, NormalizationStats
)
from data_generation.sabr_params import SABRParams


class TestFeatureConfig:
    """Test FeatureConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = FeatureConfig()
        assert config.include_sabr_params is True
        assert config.include_moneyness is True
        assert config.normalize_features is True
        assert config.normalization_method == 'standard'
    
    def test_invalid_normalization_method(self):
        """Test validation of normalization method."""
        with pytest.raises(ValueError, match="normalization_method must be one of"):
            FeatureConfig(normalization_method='invalid')


class TestPointFeatures:
    """Test PointFeatures class."""
    
    def test_point_features_creation(self):
        """Test PointFeatures creation."""
        raw_features = {'alpha': 0.2, 'beta': 0.5, 'strike': 100.0}
        point_features = PointFeatures(raw_features=raw_features)
        
        assert point_features.raw_features == raw_features
        assert point_features.normalized_features is None


class TestFeatureEngineer:
    """Test FeatureEngineer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = FeatureConfig()
        self.engineer = FeatureEngineer(self.config)
        
        # Create test SABR parameters
        self.sabr_params = SABRParams(F0=100.0, alpha=0.2, beta=0.5, nu=0.3, rho=-0.1)
        
        # Test data points
        self.strike = 105.0
        self.maturity = 1.5
        self.hagan_volatility = 0.25
    
    def test_create_point_features_basic(self):
        """Test basic point feature creation."""
        point_features = self.engineer.create_point_features(
            self.sabr_params, self.strike, self.maturity, self.hagan_volatility
        )
        
        assert isinstance(point_features, PointFeatures)
        assert 'F0' in point_features.raw_features
        assert 'alpha' in point_features.raw_features
        assert 'beta' in point_features.raw_features
        assert 'nu' in point_features.raw_features
        assert 'rho' in point_features.raw_features
        assert 'strike' in point_features.raw_features
        assert 'maturity' in point_features.raw_features
        assert 'moneyness' in point_features.raw_features
        assert 'log_moneyness' in point_features.raw_features
        assert 'hagan_volatility' in point_features.raw_features
        
        # Check values
        assert point_features.raw_features['F0'] == 100.0
        assert point_features.raw_features['alpha'] == 0.2
        assert point_features.raw_features['moneyness'] == 1.05  # 105/100
        assert abs(point_features.raw_features['log_moneyness'] - np.log(1.05)) < 1e-10
    
    def test_create_point_features_without_hagan(self):
        """Test point feature creation without Hagan volatility."""
        point_features = self.engineer.create_point_features(
            self.sabr_params, self.strike, self.maturity
        )
        
        assert 'hagan_volatility' not in point_features.raw_features
    
    def test_create_point_features_selective(self):
        """Test point feature creation with feature selection."""
        config = FeatureConfig(
            include_sabr_params=True,
            include_moneyness=False,
            include_derived_features=False
        )
        engineer = FeatureEngineer(config)
        
        point_features = engineer.create_point_features(
            self.sabr_params, self.strike, self.maturity
        )
        
        assert 'alpha' in point_features.raw_features
        assert 'moneyness' not in point_features.raw_features
        assert 'alpha_nu' not in point_features.raw_features  # Derived feature
    
    def test_derived_features(self):
        """Test derived feature creation."""
        point_features = self.engineer.create_point_features(
            self.sabr_params, self.strike, self.maturity
        )
        
        # Check some derived features
        assert 'alpha_sqrt_T' in point_features.raw_features
        assert 'nu_sqrt_T' in point_features.raw_features
        assert 'alpha_nu' in point_features.raw_features
        assert 'rho_nu' in point_features.raw_features
        assert 'moneyness_squared' in point_features.raw_features
        
        # Check values
        expected_alpha_sqrt_T = self.sabr_params.alpha * np.sqrt(self.maturity)
        assert abs(point_features.raw_features['alpha_sqrt_T'] - expected_alpha_sqrt_T) < 1e-10
        
        expected_alpha_nu = self.sabr_params.alpha * self.sabr_params.nu
        assert abs(point_features.raw_features['alpha_nu'] - expected_alpha_nu) < 1e-10
    
    def test_feature_selection(self):
        """Test explicit feature selection."""
        config = FeatureConfig(feature_selection=['alpha', 'beta', 'strike'])
        engineer = FeatureEngineer(config)
        
        point_features = engineer.create_point_features(
            self.sabr_params, self.strike, self.maturity
        )
        
        assert len(point_features.raw_features) == 3
        assert 'alpha' in point_features.raw_features
        assert 'beta' in point_features.raw_features
        assert 'strike' in point_features.raw_features
        assert 'nu' not in point_features.raw_features
    
    def test_fit_normalization_standard(self):
        """Test fitting standard normalization."""
        # Create multiple point features
        features_list = []
        for i in range(10):
            sabr_params = SABRParams(
                F0=100.0, 
                alpha=0.1 + i * 0.05, 
                beta=0.5, 
                nu=0.2 + i * 0.02, 
                rho=-0.5 + i * 0.1
            )
            point_features = self.engineer.create_point_features(
                sabr_params, 100.0 + i * 5, 1.0 + i * 0.1
            )
            features_list.append(point_features)
        
        # Fit normalization
        self.engineer.fit_normalization(features_list)
        
        assert self.engineer.is_fitted
        assert self.engineer.normalization_stats is not None
        assert self.engineer.normalization_stats.method == 'standard'
        assert self.engineer.normalization_stats.means is not None
        assert self.engineer.normalization_stats.stds is not None
    
    def test_fit_normalization_minmax(self):
        """Test fitting minmax normalization."""
        config = FeatureConfig(normalization_method='minmax')
        engineer = FeatureEngineer(config)
        
        # Create test features
        features_list = []
        for i in range(5):
            point_features = engineer.create_point_features(
                self.sabr_params, 90.0 + i * 10, 1.0 + i * 0.5
            )
            features_list.append(point_features)
        
        engineer.fit_normalization(features_list)
        
        assert engineer.normalization_stats.method == 'minmax'
        assert engineer.normalization_stats.mins is not None
        assert engineer.normalization_stats.maxs is not None
    
    def test_fit_normalization_robust(self):
        """Test fitting robust normalization."""
        config = FeatureConfig(normalization_method='robust')
        engineer = FeatureEngineer(config)
        
        # Create test features
        features_list = []
        for i in range(5):
            point_features = engineer.create_point_features(
                self.sabr_params, 90.0 + i * 10, 1.0 + i * 0.5
            )
            features_list.append(point_features)
        
        engineer.fit_normalization(features_list)
        
        assert engineer.normalization_stats.method == 'robust'
        assert engineer.normalization_stats.medians is not None
        assert engineer.normalization_stats.mads is not None
    
    def test_normalize_features_batch(self):
        """Test batch feature normalization."""
        # Create and fit normalization
        features_list = []
        for i in range(10):
            sabr_params = SABRParams(
                F0=100.0, 
                alpha=0.1 + i * 0.05, 
                beta=0.5, 
                nu=0.2, 
                rho=-0.1
            )
            point_features = self.engineer.create_point_features(
                sabr_params, 100.0, 1.0
            )
            features_list.append(point_features)
        
        self.engineer.fit_normalization(features_list)
        
        # Normalize batch
        normalized_batch = self.engineer.normalize_features_batch(features_list)
        
        assert normalized_batch.shape[0] == len(features_list)
        assert normalized_batch.shape[1] == len(self.engineer.normalization_stats.feature_names)
        
        # Check that features are updated
        for point_features in features_list:
            assert point_features.normalized_features is not None
            assert point_features.feature_names is not None
    
    def test_create_features_batch(self):
        """Test batch feature creation."""
        sabr_params_list = [self.sabr_params] * 5
        strikes = [95.0, 100.0, 105.0, 110.0, 115.0]
        maturities = [1.0, 1.5, 2.0, 2.5, 3.0]
        hagan_vols = [0.20, 0.22, 0.24, 0.26, 0.28]
        
        features_list = self.engineer.create_features_batch(
            sabr_params_list, strikes, maturities, hagan_vols
        )
        
        assert len(features_list) == 5
        
        for i, point_features in enumerate(features_list):
            assert point_features.raw_features['strike'] == strikes[i]
            assert point_features.raw_features['maturity'] == maturities[i]
            assert point_features.raw_features['hagan_volatility'] == hagan_vols[i]
    
    def test_get_feature_importance_analysis(self):
        """Test feature importance analysis."""
        # Create diverse features
        features_list = []
        for i in range(20):
            sabr_params = SABRParams(
                F0=100.0, 
                alpha=0.1 + i * 0.02, 
                beta=0.3 + i * 0.03, 
                nu=0.1 + i * 0.04, 
                rho=-0.7 + i * 0.07
            )
            point_features = self.engineer.create_point_features(
                sabr_params, 80.0 + i * 2, 0.5 + i * 0.1
            )
            features_list.append(point_features)
        
        analysis = self.engineer.get_feature_importance_analysis(features_list)
        
        assert 'feature_count' in analysis
        assert 'sample_count' in analysis
        assert 'feature_statistics' in analysis
        assert 'correlation_matrix' in analysis
        assert 'high_correlations' in analysis
        assert analysis['sample_count'] == 20
    
    def test_validate_features(self):
        """Test feature validation."""
        # Create consistent features
        features_list = []
        for i in range(5):
            point_features = self.engineer.create_point_features(
                self.sabr_params, 100.0 + i, 1.0 + i * 0.1
            )
            features_list.append(point_features)
        
        validation = self.engineer.validate_features(features_list)
        
        assert validation['valid']
        assert validation['consistency_check']
        assert validation['missing_values'] == 0
        assert validation['infinite_values'] == 0
    
    def test_validate_features_inconsistent(self):
        """Test validation with inconsistent features."""
        # Create inconsistent features
        features_list = []
        
        # First feature with all features
        point_features1 = self.engineer.create_point_features(
            self.sabr_params, 100.0, 1.0
        )
        features_list.append(point_features1)
        
        # Second feature with missing some features
        point_features2 = PointFeatures(raw_features={'alpha': 0.2, 'beta': 0.5})
        features_list.append(point_features2)
        
        validation = self.engineer.validate_features(features_list)
        
        assert not validation['valid']
        assert not validation['consistency_check']
    
    def test_save_load_normalization_stats(self):
        """Test saving and loading normalization statistics."""
        # Create and fit normalization
        features_list = []
        for i in range(5):
            point_features = self.engineer.create_point_features(
                self.sabr_params, 100.0 + i, 1.0
            )
            features_list.append(point_features)
        
        self.engineer.fit_normalization(features_list)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            self.engineer.save_normalization_stats(tmp_path)
            
            # Create new engineer and load stats
            new_engineer = FeatureEngineer(self.config)
            new_engineer.load_normalization_stats(tmp_path)
            
            assert new_engineer.is_fitted
            assert new_engineer.normalization_stats is not None
            
            # Check that loaded stats match original
            np.testing.assert_array_equal(
                new_engineer.normalization_stats.means,
                self.engineer.normalization_stats.means
            )
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_normalize_without_fitting(self):
        """Test error when normalizing without fitting."""
        point_features = self.engineer.create_point_features(
            self.sabr_params, self.strike, self.maturity
        )
        
        with pytest.raises(ValueError, match="Must call fit_normalization"):
            self.engineer._normalize_single_point(point_features.raw_features)
    
    def test_save_without_fitting(self):
        """Test error when saving without fitting."""
        with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp_file:
            with pytest.raises(ValueError, match="No normalization statistics to save"):
                self.engineer.save_normalization_stats(tmp_file.name)
    
    def test_invalid_log_moneyness(self):
        """Test handling of invalid values for log moneyness."""
        # Test with zero strike
        with pytest.warns(UserWarning, match="Invalid strike or F0"):
            point_features = self.engineer.create_point_features(
                self.sabr_params, 0.0, self.maturity
            )
            assert point_features.raw_features['log_moneyness'] == 0.0
    
    def test_empty_features_list(self):
        """Test error handling for empty features list."""
        with pytest.raises(ValueError, match="Cannot fit normalization on empty feature list"):
            self.engineer.fit_normalization([])
    
    def test_mismatched_batch_lengths(self):
        """Test error handling for mismatched batch input lengths."""
        sabr_params_list = [self.sabr_params] * 3
        strikes = [100.0, 105.0]  # Different length
        maturities = [1.0, 1.5, 2.0]
        
        with pytest.raises(ValueError, match="All input lists must have the same length"):
            self.engineer.create_features_batch(sabr_params_list, strikes, maturities)
    
    def test_zero_variance_features(self):
        """Test handling of zero variance features in normalization."""
        # Create features with constant values
        features_list = []
        for i in range(5):
            # All features will have same values
            point_features = self.engineer.create_point_features(
                self.sabr_params, 100.0, 1.0  # Same values
            )
            features_list.append(point_features)
        
        # Should handle zero variance gracefully
        self.engineer.fit_normalization(features_list)
        
        assert self.engineer.is_fitted
        # Standard deviations should be set to 1.0 for zero variance features
        assert np.all(self.engineer.normalization_stats.stds >= 1.0)


if __name__ == "__main__":
    pytest.main([__file__])