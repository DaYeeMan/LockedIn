"""
Tests for patch extraction functionality.
"""

import pytest
import numpy as np
from typing import List

from preprocessing.patch_extractor import (
    PatchExtractor, PatchConfig, HFPoint, ExtractedPatch
)
from data_generation.sabr_params import SABRParams


class TestPatchConfig:
    """Test PatchConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = PatchConfig()
        assert config.patch_size == (9, 9)
        assert config.boundary_mode == 'reflect'
        assert config.normalize_patches is True
        
    def test_invalid_patch_size(self):
        """Test validation of patch size."""
        with pytest.raises(ValueError, match="Patch size must be positive"):
            PatchConfig(patch_size=(0, 5))
            
        with pytest.raises(ValueError, match="Patch size must be positive"):
            PatchConfig(patch_size=(5, -1))
    
    def test_invalid_boundary_mode(self):
        """Test validation of boundary mode."""
        with pytest.raises(ValueError, match="boundary_mode must be one of"):
            PatchConfig(boundary_mode='invalid')
    
    def test_even_patch_size_warning(self):
        """Test warning for even patch sizes."""
        with pytest.warns(UserWarning, match="Even patch sizes may not center properly"):
            PatchConfig(patch_size=(8, 8))


class TestHFPoint:
    """Test HFPoint class."""
    
    def test_hf_point_creation(self):
        """Test HF point creation."""
        point = HFPoint(strike=100.0, maturity=1.0, volatility=0.2)
        assert point.strike == 100.0
        assert point.maturity == 1.0
        assert point.volatility == 0.2
        assert point.grid_coords is None


class TestPatchExtractor:
    """Test PatchExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PatchConfig(patch_size=(5, 5), boundary_mode='constant', pad_value=0.0, normalize_patches=False)
        self.extractor = PatchExtractor(self.config)
        
        # Create test surface
        self.surface = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0]
        ])
        
        # Create test grids
        self.strikes_grid = np.array([
            [90.0, 95.0, 100.0, 105.0, 110.0],
            [90.0, 95.0, 100.0, 105.0, 110.0],
            [90.0, 95.0, 100.0, 105.0, 110.0]
        ])
        self.maturities_grid = np.array([1.0, 2.0, 3.0])
        
        # Create test SABR params
        self.sabr_params = SABRParams(F0=100.0, alpha=0.2, beta=0.5, nu=0.3, rho=-0.1)
    
    def test_extract_patch_center(self):
        """Test extracting patch from center of surface."""
        patch_result = self.extractor.extract_patch(self.surface, (2, 2))
        
        assert patch_result.patch.shape == (5, 5)
        assert patch_result.center_coords == (2, 2)
        
        # Check that center value is correct
        assert patch_result.patch[2, 2] == self.surface[2, 2]
        
        # Check boundary flags (should be all False for center patch)
        assert not any(patch_result.boundary_flags.values())
    
    def test_extract_patch_corner(self):
        """Test extracting patch from corner (boundary handling)."""
        patch_result = self.extractor.extract_patch(self.surface, (0, 0))
        
        assert patch_result.patch.shape == (5, 5)
        assert patch_result.center_coords == (0, 0)
        
        # Check boundary flags
        assert patch_result.boundary_flags['top']
        assert patch_result.boundary_flags['left']
        assert not patch_result.boundary_flags['bottom']
        assert not patch_result.boundary_flags['right']
        
        # Check that padding was applied (should have zeros)
        assert patch_result.patch[0, 0] == 0.0  # Padded area
        assert patch_result.patch[2, 2] == self.surface[0, 0]  # Center of patch
    
    def test_align_hf_to_grid(self):
        """Test alignment of HF points to grid."""
        hf_points = [
            HFPoint(strike=100.0, maturity=1.0, volatility=0.25),
            HFPoint(strike=105.0, maturity=2.0, volatility=0.30)
        ]
        
        aligned_points = self.extractor.align_hf_to_grid(
            hf_points, self.strikes_grid, self.maturities_grid
        )
        
        assert len(aligned_points) == 2
        
        # Check first point alignment
        assert aligned_points[0].grid_coords == (0, 2)  # Closest to (1.0, 100.0)
        
        # Check second point alignment  
        assert aligned_points[1].grid_coords == (1, 3)  # Closest to (2.0, 105.0)
    
    def test_extract_patches_for_hf_points(self):
        """Test extracting patches for multiple HF points."""
        # Create larger surface for this test
        large_surface = np.random.rand(10, 10)
        large_strikes = np.tile(np.linspace(80, 120, 10), (5, 1))
        large_maturities = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        
        hf_points = [
            HFPoint(strike=100.0, maturity=1.0, volatility=0.25),
            HFPoint(strike=110.0, maturity=1.5, volatility=0.30)
        ]
        
        patches = self.extractor.extract_patches_for_hf_points(
            large_surface, hf_points, large_strikes, large_maturities
        )
        
        assert len(patches) == 2
        
        for patch in patches:
            assert isinstance(patch, ExtractedPatch)
            assert patch.patch.shape == self.config.patch_size
            assert patch.hf_point.grid_coords is not None
    
    def test_boundary_modes(self):
        """Test different boundary handling modes."""
        configs = [
            PatchConfig(patch_size=(5, 5), boundary_mode='constant', pad_value=999.0, normalize_patches=False),
            PatchConfig(patch_size=(5, 5), boundary_mode='reflect', normalize_patches=False),
            PatchConfig(patch_size=(5, 5), boundary_mode='wrap', normalize_patches=False),
            PatchConfig(patch_size=(5, 5), boundary_mode='pad', normalize_patches=False)
        ]
        
        for config in configs:
            extractor = PatchExtractor(config)
            patch_result = extractor.extract_patch(self.surface, (0, 0))
            
            assert patch_result.patch.shape == (5, 5)
            
            if config.boundary_mode == 'constant':
                # Check that padding value is used
                assert patch_result.patch[0, 0] == 999.0
    
    def test_patch_normalization(self):
        """Test patch normalization."""
        config = PatchConfig(patch_size=(3, 3), normalize_patches=True)
        extractor = PatchExtractor(config)
        
        patch_result = extractor.extract_patch(self.surface, (2, 2))
        
        assert patch_result.normalization_stats is not None
        assert 'mean' in patch_result.normalization_stats
        assert 'std' in patch_result.normalization_stats
        
        # Check that patch is normalized (mean should be close to 0)
        patch_mean = np.mean(patch_result.patch[~np.isnan(patch_result.patch)])
        assert abs(patch_mean) < 1e-10
    
    def test_batch_extract_patches(self):
        """Test batch patch extraction."""
        surfaces = [self.surface, self.surface * 2]
        hf_points_list = [
            [HFPoint(strike=100.0, maturity=1.0, volatility=0.25)],
            [HFPoint(strike=95.0, maturity=1.0, volatility=0.30)]
        ]
        strikes_grids = [self.strikes_grid, self.strikes_grid]
        maturities_grids = [self.maturities_grid, self.maturities_grid]
        
        all_patches = self.extractor.batch_extract_patches(
            surfaces, hf_points_list, strikes_grids, maturities_grids
        )
        
        assert len(all_patches) == 2
        assert len(all_patches[0]) == 1
        assert len(all_patches[1]) == 1
    
    def test_validate_patch_extraction(self):
        """Test patch extraction validation."""
        patches = []
        
        # Create some test patches
        for i in range(3):
            patch_result = self.extractor.extract_patch(self.surface, (1, 1))
            patches.append(patch_result)
        
        validation = self.extractor.validate_patch_extraction(patches)
        
        assert validation['valid']
        assert validation['patch_count'] == 3
        assert validation['size_consistency']
    
    def test_get_patch_statistics(self):
        """Test patch statistics computation."""
        patches = []
        
        # Create test patches
        for i in range(5):
            patch_result = self.extractor.extract_patch(self.surface, (2, 2))
            patches.append(patch_result)
        
        stats = self.extractor.get_patch_statistics(patches)
        
        assert 'total_patches' in stats
        assert 'value_statistics' in stats
        assert 'boundary_usage' in stats
        assert stats['total_patches'] == 5
    
    def test_nan_surface_handling(self):
        """Test handling of surfaces with NaN values."""
        nan_surface = self.surface.copy()
        nan_surface[0, 0] = np.nan
        nan_surface[4, 4] = np.nan
        
        patch_result = self.extractor.extract_patch(nan_surface, (2, 2))
        
        # Should handle NaN values gracefully
        assert patch_result.patch.shape == (5, 5)
        
        # If normalization is enabled, should handle NaN values
        if self.config.normalize_patches:
            assert patch_result.normalization_stats is not None
    
    def test_empty_hf_points(self):
        """Test handling of empty HF points list."""
        patches = self.extractor.extract_patches_for_hf_points(
            self.surface, [], self.strikes_grid, self.maturities_grid
        )
        
        assert len(patches) == 0
    
    def test_mismatched_input_lengths(self):
        """Test error handling for mismatched input lengths."""
        with pytest.raises(ValueError, match="All input lists must have the same length"):
            self.extractor.batch_extract_patches(
                [self.surface], 
                [[], []], 
                [self.strikes_grid], 
                [self.maturities_grid]
            )


if __name__ == "__main__":
    pytest.main([__file__])