"""
Integration tests for patch extraction and feature engineering.
"""

import pytest
import numpy as np

from preprocessing.patch_extractor import PatchExtractor, PatchConfig, HFPoint
from preprocessing.feature_engineer import FeatureEngineer, FeatureConfig
from data_generation.sabr_params import SABRParams


def test_patch_extraction_and_feature_engineering_integration():
    """Test that patch extraction and feature engineering work together."""
    
    # Create test data
    sabr_params = SABRParams(F0=100.0, alpha=0.2, beta=0.5, nu=0.3, rho=-0.1)
    
    # Create test surface (LF Hagan surface)
    surface = np.random.rand(10, 15) * 0.3 + 0.1  # Volatilities between 0.1 and 0.4
    
    # Create test grids
    strikes_grid = np.tile(np.linspace(80, 120, 15), (10, 1))
    maturities_grid = np.linspace(0.5, 5.0, 10)
    
    # Create HF points
    hf_points = [
        HFPoint(strike=95.0, maturity=1.0, volatility=0.25),
        HFPoint(strike=105.0, maturity=2.0, volatility=0.30),
        HFPoint(strike=110.0, maturity=3.0, volatility=0.28)
    ]
    
    # Initialize extractors
    patch_config = PatchConfig(patch_size=(7, 7), normalize_patches=True)
    patch_extractor = PatchExtractor(patch_config)
    
    feature_config = FeatureConfig(normalize_features=True)
    feature_engineer = FeatureEngineer(feature_config)
    
    # Extract patches
    patches = patch_extractor.extract_patches_for_hf_points(
        surface, hf_points, strikes_grid, maturities_grid
    )
    
    assert len(patches) == 3
    
    # Create point features for the same HF points
    point_features_list = []
    for hf_point in hf_points:
        # Get Hagan volatility from surface at aligned grid point
        if hf_point.grid_coords is not None:
            i, j = hf_point.grid_coords
            hagan_vol = surface[i, j] if i < surface.shape[0] and j < surface.shape[1] else 0.2
        else:
            hagan_vol = 0.2
        
        point_features = feature_engineer.create_point_features(
            sabr_params, hf_point.strike, hf_point.maturity, hagan_vol
        )
        point_features_list.append(point_features)
    
    # Fit normalization
    feature_engineer.fit_normalization(point_features_list)
    
    # Normalize features
    normalized_features = feature_engineer.normalize_features_batch(point_features_list)
    
    # Verify shapes and consistency
    assert normalized_features.shape[0] == len(hf_points)
    assert normalized_features.shape[1] > 0  # Should have features
    
    # Verify patches have correct shape
    for patch in patches:
        assert patch.patch.shape == patch_config.patch_size
        assert patch.hf_point.grid_coords is not None
    
    # Verify feature normalization worked
    for point_features in point_features_list:
        assert point_features.normalized_features is not None
        assert len(point_features.normalized_features) == normalized_features.shape[1]
    
    print(f"Successfully processed {len(patches)} patches and {len(point_features_list)} feature sets")
    print(f"Patch shape: {patches[0].patch.shape}")
    print(f"Feature vector length: {normalized_features.shape[1]}")
    print(f"Feature names: {point_features_list[0].feature_names[:5]}...")  # First 5 features


def test_mda_cnn_data_preparation():
    """Test preparing data in the format expected by MDA-CNN."""
    
    # Create multiple SABR parameter sets
    sabr_params_list = [
        SABRParams(F0=100.0, alpha=0.15, beta=0.4, nu=0.25, rho=-0.2),
        SABRParams(F0=100.0, alpha=0.25, beta=0.6, nu=0.35, rho=0.1),
        SABRParams(F0=100.0, alpha=0.20, beta=0.5, nu=0.30, rho=-0.1)
    ]
    
    # Create corresponding surfaces
    surfaces = []
    strikes_grids = []
    maturities_grids = []
    
    for _ in sabr_params_list:
        surface = np.random.rand(8, 12) * 0.4 + 0.1
        strikes_grid = np.tile(np.linspace(70, 130, 12), (8, 1))
        maturities_grid = np.linspace(0.25, 4.0, 8)
        
        surfaces.append(surface)
        strikes_grids.append(strikes_grid)
        maturities_grids.append(maturities_grid)
    
    # Create HF points for each surface
    hf_points_list = []
    for i, sabr_params in enumerate(sabr_params_list):
        hf_points = [
            HFPoint(strike=90.0, maturity=1.0, volatility=0.22 + i * 0.02),
            HFPoint(strike=110.0, maturity=2.0, volatility=0.28 + i * 0.02)
        ]
        hf_points_list.append(hf_points)
    
    # Initialize processors
    patch_extractor = PatchExtractor(PatchConfig(patch_size=(5, 5)))
    feature_engineer = FeatureEngineer(FeatureConfig())
    
    # Process all data
    all_patches = patch_extractor.batch_extract_patches(
        surfaces, hf_points_list, strikes_grids, maturities_grids
    )
    
    # Collect all point features
    all_point_features = []
    for i, (sabr_params, hf_points) in enumerate(zip(sabr_params_list, hf_points_list)):
        for hf_point in hf_points:
            point_features = feature_engineer.create_point_features(
                sabr_params, hf_point.strike, hf_point.maturity, 0.25  # Mock Hagan vol
            )
            all_point_features.append(point_features)
    
    # Fit normalization on all features
    feature_engineer.fit_normalization(all_point_features)
    normalized_features = feature_engineer.normalize_features_batch(all_point_features)
    
    # Prepare MDA-CNN input format
    patch_data = []
    feature_data = []
    target_data = []
    
    patch_idx = 0
    for surface_patches in all_patches:
        for patch in surface_patches:
            patch_data.append(patch.patch)
            feature_data.append(normalized_features[patch_idx])
            target_data.append(patch.hf_point.volatility - 0.25)  # Mock residual
            patch_idx += 1
    
    # Convert to arrays
    patch_array = np.array(patch_data)
    feature_array = np.array(feature_data)
    target_array = np.array(target_data)
    
    # Verify shapes for MDA-CNN
    assert patch_array.ndim == 3  # (n_samples, patch_height, patch_width)
    assert feature_array.ndim == 2  # (n_samples, n_features)
    assert target_array.ndim == 1  # (n_samples,)
    assert patch_array.shape[0] == feature_array.shape[0] == target_array.shape[0]
    
    print(f"MDA-CNN data preparation successful:")
    print(f"  Patch data shape: {patch_array.shape}")
    print(f"  Feature data shape: {feature_array.shape}")
    print(f"  Target data shape: {target_array.shape}")
    print(f"  Total samples: {len(patch_data)}")


if __name__ == "__main__":
    test_patch_extraction_and_feature_engineering_integration()
    test_mda_cnn_data_preparation()
    print("All integration tests passed!")