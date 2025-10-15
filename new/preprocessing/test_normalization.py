"""
Tests for data normalization and scaling utilities.

This module contains comprehensive tests for the normalization components
to ensure correctness, robustness, and performance.
"""

import pytest
import numpy as np
import tempfile
import os
from typing import Tuple, Dict, Any

from preprocessing.normalization import (
    NormalizationConfig, NormalizationStats, DataNormalizer,
    PatchNormalizer, TargetNormalizer, create_normalization_pipeline,
    validate_normalization
)


class TestNormalizationConfig:
    """Test normalization configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = NormalizationConfig()
        
        assert config.method == 'standard'
        assert config.clip_outliers == False
        assert config.outlier_percentiles == (1.0, 99.0)
        assert config.epsilon == 1e-8
        assert config.per_feature == True
        assert config.handle_nan == 'skip'
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = NormalizationConfig(method='minmax', handle_nan='zero')
        assert config.method == 'minmax'
        
        # Invalid method
        with pytest.raises(ValueError, match="method must be one of"):
            NormalizationConfig(method='invalid')
        
        # Invalid nan handling
        with pytest.raises(ValueError, match="handle_nan must be one of"):
            NormalizationConfig(handle_nan='invalid')
        
        # Invalid percentiles
        with pytest.raises(ValueError, match="outlier_percentiles must be in range"):
            NormalizationConfig(outlier_percentiles=(50.0, 10.0))


class TestDataNormalizer:
    """Test basic data normalizer functionality."""
    
    def create_test_data(self, shape: Tuple[int, ...], 
                        add_outliers: bool = False,
                        add_nan: bool = False) -> np.ndarray:
        """Create test data with specified characteristics."""
        np.random.seed(42)
        data = np.random.randn(*shape)
        
        if add_outliers:
            # Add some outliers
            n_outliers = max(1, shape[0] // 20)
            outlier_indices = np.random.choice(shape[0], n_outliers, replace=False)
            data[outlier_indices] *= 10  # Make them 10x larger
        
        if add_nan:
            # Add some NaN values
            n_nan = max(1, shape[0] // 50)
            nan_indices = np.random.choice(shape[0], n_nan, replace=False)
            if data.ndim == 1:
                data[nan_indices] = np.nan
            else:
                data[nan_indices, 0] = np.nan
        
        return data
    
    def test_standard_normalization_1d(self):
        """Test standard normalization on 1D data."""
        data = self.create_test_data((1000,))
        
        normalizer = DataNormalizer(NormalizationConfig(method='standard'))
        normalized = normalizer.fit_transform(data)
        
        # Check normalization properties
        assert abs(np.mean(normalized)) < 1e-10
        assert abs(np.std(normalized) - 1.0) < 1e-10
        
        # Check inverse transformation
        reconstructed = normalizer.inverse_transform(normalized)
        np.testing.assert_array_almost_equal(data, reconstructed)
    
    def test_standard_normalization_2d(self):
        """Test standard normalization on 2D data."""
        data = self.create_test_data((1000, 5))
        
        config = NormalizationConfig(method='standard', per_feature=True)
        normalizer = DataNormalizer(config)
        normalized = normalizer.fit_transform(data)
        
        # Check per-feature normalization
        for i in range(data.shape[1]):
            feature_data = normalized[:, i]
            assert abs(np.mean(feature_data)) < 1e-10
            assert abs(np.std(feature_data) - 1.0) < 1e-10
        
        # Check inverse transformation
        reconstructed = normalizer.inverse_transform(normalized)
        np.testing.assert_array_almost_equal(data, reconstructed)
    
    def test_minmax_normalization(self):
        """Test min-max normalization."""
        data = self.create_test_data((500, 3))
        
        config = NormalizationConfig(method='minmax')
        normalizer = DataNormalizer(config)
        normalized = normalizer.fit_transform(data)
        
        # Check min-max properties
        for i in range(data.shape[1]):
            feature_data = normalized[:, i]
            assert np.min(feature_data) >= -1e-10  # Should be >= 0
            assert np.max(feature_data) <= 1 + 1e-10  # Should be <= 1
        
        # Check inverse transformation
        reconstructed = normalizer.inverse_transform(normalized)
        np.testing.assert_array_almost_equal(data, reconstructed)
    
    def test_robust_normalization(self):
        """Test robust normalization."""
        data = self.create_test_data((500, 3), add_outliers=True)
        
        config = NormalizationConfig(method='robust')
        normalizer = DataNormalizer(config)
        normalized = normalizer.fit_transform(data)
        
        # Check that normalization completed without errors
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))
        
        # Check inverse transformation
        reconstructed = normalizer.inverse_transform(normalized)
        np.testing.assert_array_almost_equal(data, reconstructed)
    
    def test_quantile_normalization(self):
        """Test quantile normalization."""
        data = self.create_test_data((1000, 2))
        
        config = NormalizationConfig(method='quantile')
        normalizer = DataNormalizer(config)
        normalized = normalizer.fit_transform(data)
        
        # Check quantile properties (should be roughly uniform in [0,1])
        for i in range(data.shape[1]):
            feature_data = normalized[:, i]
            assert np.min(feature_data) >= -0.1  # Allow small tolerance
            assert np.max(feature_data) <= 1.1
        
        # Check inverse transformation
        reconstructed = normalizer.inverse_transform(normalized)
        np.testing.assert_array_almost_equal(data, reconstructed, decimal=3)
    
    def test_outlier_clipping(self):
        """Test outlier clipping functionality."""
        data = self.create_test_data((1000,), add_outliers=True)
        
        config = NormalizationConfig(
            method='standard',
            clip_outliers=True,
            outlier_percentiles=(5.0, 95.0)
        )
        normalizer = DataNormalizer(config)
        normalized = normalizer.fit_transform(data)
        
        # Check that clipping was applied
        assert normalizer.stats.clip_bounds is not None
        
        # Normalized data should be well-behaved
        assert abs(np.mean(normalized)) < 0.1  # Close to zero
        assert 0.5 < np.std(normalized) < 2.0  # Reasonable std
    
    def test_nan_handling_skip(self):
        """Test NaN handling with skip method."""
        data = self.create_test_data((100,), add_nan=True)
        
        config = NormalizationConfig(method='standard', handle_nan='skip')
        normalizer = DataNormalizer(config)
        normalized = normalizer.fit_transform(data)
        
        # NaN values should remain NaN
        nan_mask = np.isnan(data)
        assert np.all(np.isnan(normalized[nan_mask]))
        
        # Non-NaN values should be normalized
        valid_mask = ~nan_mask
        valid_normalized = normalized[valid_mask]
        assert not np.any(np.isnan(valid_normalized))
    
    def test_nan_handling_zero(self):
        """Test NaN handling with zero replacement."""
        data = self.create_test_data((100,), add_nan=True)
        
        config = NormalizationConfig(method='standard', handle_nan='zero')
        normalizer = DataNormalizer(config)
        normalized = normalizer.fit_transform(data)
        
        # Should have no NaN values
        assert not np.any(np.isnan(normalized))
    
    def test_nan_handling_mean(self):
        """Test NaN handling with mean replacement."""
        data = self.create_test_data((100,), add_nan=True)
        
        config = NormalizationConfig(method='standard', handle_nan='mean')
        normalizer = DataNormalizer(config)
        normalized = normalizer.fit_transform(data)
        
        # Should have no NaN values
        assert not np.any(np.isnan(normalized))
    
    def test_save_load_normalizer(self):
        """Test saving and loading normalizer."""
        data = self.create_test_data((500, 3))
        
        normalizer = DataNormalizer(NormalizationConfig(method='standard'))
        normalizer.fit(data)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            save_path = f.name
        
        try:
            # Save normalizer
            normalizer.save(save_path)
            
            # Load new normalizer
            new_normalizer = DataNormalizer()
            new_normalizer.load(save_path)
            
            # Test that loaded normalizer works the same
            original_normalized = normalizer.transform(data)
            loaded_normalized = new_normalizer.transform(data)
            
            np.testing.assert_array_almost_equal(original_normalized, loaded_normalized)
        
        finally:
            os.unlink(save_path)
    
    def test_statistics_summary(self):
        """Test statistics summary generation."""
        data = self.create_test_data((500, 3))
        
        normalizer = DataNormalizer(NormalizationConfig(method='standard'))
        normalizer.fit(data)
        
        summary = normalizer.get_statistics_summary()
        
        assert summary['method'] == 'standard'
        assert summary['n_samples'] == 500
        assert summary['shape'] == (500, 3)
        assert 'means' in summary
        assert 'stds' in summary
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        normalizer = DataNormalizer()
        
        # Test transform before fit
        with pytest.raises(ValueError, match="Must call fit"):
            normalizer.transform(np.array([1, 2, 3]))
        
        # Test inverse transform before fit
        with pytest.raises(ValueError, match="Must call fit"):
            normalizer.inverse_transform(np.array([1, 2, 3]))
        
        # Test fit on empty data
        with pytest.raises(ValueError, match="Cannot fit on empty data"):
            normalizer.fit(np.array([]))
        
        # Test constant data (should not crash)
        constant_data = np.ones((100, 2))
        normalizer.fit(constant_data)
        normalized = normalizer.transform(constant_data)
        
        # Should handle constant data gracefully
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))


class TestPatchNormalizer:
    """Test patch-specific normalizer."""
    
    def create_test_patches(self, n_patches: int = 100, 
                          patch_size: Tuple[int, int] = (9, 9)) -> np.ndarray:
        """Create test patch data."""
        np.random.seed(42)
        patches = np.random.randn(n_patches, *patch_size)
        
        # Add some variation between patches
        for i in range(n_patches):
            patches[i] *= (0.5 + np.random.random())  # Scale variation
            patches[i] += np.random.randn() * 0.1  # Offset variation
        
        return patches.astype(np.float32)
    
    def test_global_patch_normalization(self):
        """Test global patch normalization."""
        patches = self.create_test_patches(50, (9, 9))
        
        config = NormalizationConfig(method='standard')
        normalizer = PatchNormalizer(config, spatial_normalization=False)
        
        normalized = normalizer.fit_transform(patches)
        
        # Check shape preservation
        assert normalized.shape == patches.shape
        
        # Check global normalization properties
        flattened = normalized.reshape(patches.shape[0], -1)
        global_mean = np.mean(flattened)
        global_std = np.std(flattened)
        
        assert abs(global_mean) < 1e-6
        assert abs(global_std - 1.0) < 1e-6
    
    def test_spatial_patch_normalization(self):
        """Test spatial patch normalization."""
        patches = self.create_test_patches(50, (9, 9))
        
        config = NormalizationConfig(method='standard')
        normalizer = PatchNormalizer(config, spatial_normalization=True)
        
        normalized = normalizer.fit_transform(patches)
        
        # Check shape preservation
        assert normalized.shape == patches.shape
        
        # Each patch should be individually normalized (approximately)
        # Note: This is approximate due to the two-stage normalization
        for i in range(min(5, patches.shape[0])):  # Check first few patches
            patch = normalized[i]
            patch_mean = np.mean(patch)
            patch_std = np.std(patch)
            
            # Should be roughly normalized (allowing for global normalization effect)
            assert abs(patch_mean) < 1.0
            assert 0.1 < patch_std < 10.0
    
    def test_patch_with_nan(self):
        """Test patch normalization with NaN values."""
        patches = self.create_test_patches(20, (5, 5))
        
        # Add NaN to some patches
        patches[0, 0, 0] = np.nan
        patches[1, :, :] = np.nan  # Entire patch NaN
        
        config = NormalizationConfig(method='standard', handle_nan='skip')
        normalizer = PatchNormalizer(config)
        
        normalized = normalizer.fit_transform(patches)
        
        # Should handle NaN gracefully
        assert normalized.shape == patches.shape
        assert np.isnan(normalized[0, 0, 0])  # NaN preserved
        assert np.all(np.isnan(normalized[1, :, :]))  # All-NaN patch preserved


class TestTargetNormalizer:
    """Test target-specific normalizer."""
    
    def create_test_targets(self, n_samples: int = 1000) -> np.ndarray:
        """Create test target data (residuals)."""
        np.random.seed(42)
        
        # Create residuals with realistic distribution
        # Most residuals small, some larger outliers
        targets = np.random.normal(0, 0.02, n_samples)  # Small residuals
        
        # Add some larger residuals (outliers)
        n_outliers = n_samples // 20
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        targets[outlier_indices] += np.random.normal(0, 0.1, n_outliers)
        
        return targets
    
    def test_target_normalization(self):
        """Test basic target normalization."""
        targets = self.create_test_targets(1000)
        
        config = NormalizationConfig(method='standard')
        normalizer = TargetNormalizer(config)
        
        normalized = normalizer.fit_transform(targets)
        
        # Check normalization properties
        assert abs(np.mean(normalized)) < 1e-10
        assert abs(np.std(normalized) - 1.0) < 1e-10
        
        # Check inverse transformation
        reconstructed = normalizer.inverse_transform(normalized)
        np.testing.assert_array_almost_equal(targets, reconstructed)
    
    def test_symmetric_bounds(self):
        """Test symmetric bounds for target clipping."""
        targets = self.create_test_targets(1000)
        
        config = NormalizationConfig(
            method='standard',
            clip_outliers=True,
            outlier_percentiles=(10.0, 90.0)
        )
        normalizer = TargetNormalizer(config, symmetric_bounds=True)
        
        normalizer.fit(targets)
        
        # Check that bounds are symmetric
        if normalizer.stats.clip_bounds is not None:
            lower, upper = normalizer.stats.clip_bounds
            assert abs(abs(lower) - abs(upper)) < 1e-10
    
    def test_asymmetric_targets(self):
        """Test normalization of asymmetric target distribution."""
        # Create asymmetric residuals (more negative than positive)
        np.random.seed(42)
        targets = np.concatenate([
            np.random.normal(-0.05, 0.02, 700),  # Negative bias
            np.random.normal(0.02, 0.01, 300)   # Smaller positive
        ])
        
        config = NormalizationConfig(method='standard')
        normalizer = TargetNormalizer(config)
        
        normalized = normalizer.fit_transform(targets)
        
        # Should still normalize properly
        assert abs(np.mean(normalized)) < 1e-10
        assert abs(np.std(normalized) - 1.0) < 1e-10


class TestNormalizationPipeline:
    """Test complete normalization pipeline."""
    
    def test_create_pipeline(self):
        """Test creating normalization pipeline."""
        patch_config = NormalizationConfig(method='standard')
        feature_config = NormalizationConfig(method='minmax')
        target_config = NormalizationConfig(method='robust')
        
        pipeline = create_normalization_pipeline(
            patch_config, feature_config, target_config
        )
        
        assert 'patches' in pipeline
        assert 'features' in pipeline
        assert 'targets' in pipeline
        
        assert isinstance(pipeline['patches'], PatchNormalizer)
        assert isinstance(pipeline['features'], DataNormalizer)
        assert isinstance(pipeline['targets'], TargetNormalizer)
    
    def test_pipeline_integration(self):
        """Test integrated pipeline usage."""
        # Create test data
        patches = np.random.randn(100, 9, 9).astype(np.float32)
        features = np.random.randn(100, 8).astype(np.float32)
        targets = np.random.randn(100).astype(np.float32)
        
        # Create pipeline
        pipeline = create_normalization_pipeline()
        
        # Fit and transform all data types
        normalized_patches = pipeline['patches'].fit_transform(patches)
        normalized_features = pipeline['features'].fit_transform(features)
        normalized_targets = pipeline['targets'].fit_transform(targets)
        
        # Check shapes preserved
        assert normalized_patches.shape == patches.shape
        assert normalized_features.shape == features.shape
        assert normalized_targets.shape == targets.shape
        
        # Check inverse transformations
        reconstructed_patches = pipeline['patches'].inverse_transform(normalized_patches)
        reconstructed_features = pipeline['features'].inverse_transform(normalized_features)
        reconstructed_targets = pipeline['targets'].inverse_transform(normalized_targets)
        
        np.testing.assert_array_almost_equal(patches, reconstructed_patches)
        np.testing.assert_array_almost_equal(features, reconstructed_features)
        np.testing.assert_array_almost_equal(targets, reconstructed_targets)


class TestNormalizationValidation:
    """Test normalization validation utilities."""
    
    def test_validate_normalization_success(self):
        """Test successful normalization validation."""
        data = np.random.randn(1000, 5)
        
        normalizer = DataNormalizer(NormalizationConfig(method='standard'))
        normalized = normalizer.fit_transform(data)
        
        validation = validate_normalization(data, normalized, normalizer)
        
        assert validation['inverse_transform_valid'] == True
        assert validation['max_reconstruction_error'] < 1e-10
        assert validation['mean_close_to_zero'] == True
        assert validation['std_close_to_one'] == True
    
    def test_validate_normalization_with_nan(self):
        """Test validation with NaN values."""
        data = np.random.randn(100, 3)
        data[10:15, 0] = np.nan  # Add some NaN values
        
        config = NormalizationConfig(method='standard', handle_nan='skip')
        normalizer = DataNormalizer(config)
        normalized = normalizer.fit_transform(data)
        
        validation = validate_normalization(data, normalized, normalizer)
        
        assert 'n_nan_original' in validation
        assert 'n_nan_normalized' in validation
        assert validation['n_nan_original'] == 5
        assert validation['n_nan_normalized'] == 5
    
    def test_validate_minmax_normalization(self):
        """Test validation of min-max normalization."""
        data = np.random.randn(500, 2)
        
        normalizer = DataNormalizer(NormalizationConfig(method='minmax'))
        normalized = normalizer.fit_transform(data)
        
        validation = validate_normalization(data, normalized, normalizer)
        
        assert validation['inverse_transform_valid'] == True
        assert validation['in_unit_range'] == True
    
    def test_validate_normalization_failure(self):
        """Test validation with corrupted normalization."""
        data = np.random.randn(100, 2)
        
        normalizer = DataNormalizer(NormalizationConfig(method='standard'))
        normalized = normalizer.fit_transform(data)
        
        # Corrupt the normalized data
        corrupted_normalized = normalized * 1000  # Scale it up
        
        validation = validate_normalization(data, corrupted_normalized, normalizer)
        
        assert validation['inverse_transform_valid'] == False
        assert validation['max_reconstruction_error'] > 1.0


class TestNormalizationPerformance:
    """Test normalization performance characteristics."""
    
    @pytest.mark.slow
    def test_large_data_normalization(self):
        """Test normalization performance on large datasets."""
        import time
        
        # Create large dataset
        large_data = np.random.randn(50000, 20).astype(np.float32)
        
        normalizer = DataNormalizer(NormalizationConfig(method='standard'))
        
        # Time fitting
        start_time = time.time()
        normalizer.fit(large_data)
        fit_time = time.time() - start_time
        
        # Time transformation
        start_time = time.time()
        normalized = normalizer.transform(large_data)
        transform_time = time.time() - start_time
        
        print(f"Fit time: {fit_time:.2f}s, Transform time: {transform_time:.2f}s")
        
        # Performance should be reasonable
        assert fit_time < 5.0  # Should fit quickly
        assert transform_time < 2.0  # Should transform quickly
        
        # Check correctness
        assert normalized.shape == large_data.shape
        assert abs(np.mean(normalized)) < 1e-6
    
    def test_memory_efficiency(self):
        """Test memory efficiency of normalization."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create moderately large dataset
        data = np.random.randn(10000, 50).astype(np.float32)
        
        normalizer = DataNormalizer(NormalizationConfig(method='standard'))
        normalized = normalizer.fit_transform(data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase: {memory_increase:.1f} MB")
        
        # Should not use excessive memory
        assert memory_increase < 200  # Should be reasonable
        
        # Clean up
        del data, normalized


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])