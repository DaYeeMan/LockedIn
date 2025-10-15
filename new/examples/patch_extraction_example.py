"""
Example demonstrating patch extraction and feature engineering for MDA-CNN.

This example shows how to:
1. Extract patches from LF surfaces around HF points
2. Create point features for MLP input
3. Prepare data in MDA-CNN format
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from preprocessing.patch_extractor import PatchExtractor, PatchConfig, HFPoint
from preprocessing.feature_engineer import FeatureEngineer, FeatureConfig
from data_generation.sabr_params import SABRParams
from data_generation.hagan_surface_generator import HaganSurfaceGenerator
from data_generation.sabr_params import GridConfig


def main():
    """Run patch extraction and feature engineering example."""
    
    print("=== Patch Extraction and Feature Engineering Example ===\n")
    
    # 1. Create SABR parameters and generate LF surface
    print("1. Generating SABR volatility surface...")
    sabr_params = SABRParams(F0=100.0, alpha=0.2, beta=0.5, nu=0.3, rho=-0.1)
    grid_config = GridConfig(n_strikes=15, n_maturities=8)
    
    # Generate Hagan surface
    hagan_generator = HaganSurfaceGenerator()
    hagan_result = hagan_generator.generate_volatility_surface(sabr_params, grid_config)
    
    print(f"   Surface shape: {hagan_result.volatility_surface.shape}")
    print(f"   Maturity range: {hagan_result.maturities[0]:.2f} - {hagan_result.maturities[-1]:.2f} years")
    print(f"   Strike range: {hagan_result.strikes[0, 0]:.1f} - {hagan_result.strikes[0, -1]:.1f}")
    
    # 2. Define high-fidelity points
    print("\n2. Defining high-fidelity points...")
    hf_points = [
        HFPoint(strike=95.0, maturity=1.5, volatility=0.25),
        HFPoint(strike=100.0, maturity=2.0, volatility=0.22),
        HFPoint(strike=105.0, maturity=3.0, volatility=0.28),
        HFPoint(strike=110.0, maturity=1.0, volatility=0.30)
    ]
    
    print(f"   Number of HF points: {len(hf_points)}")
    for i, hf_point in enumerate(hf_points):
        print(f"   HF Point {i+1}: K={hf_point.strike:.1f}, T={hf_point.maturity:.1f}, σ_MC={hf_point.volatility:.3f}")
    
    # 3. Extract patches around HF points
    print("\n3. Extracting patches around HF points...")
    patch_config = PatchConfig(
        patch_size=(7, 7),
        boundary_mode='reflect',
        normalize_patches=True
    )
    patch_extractor = PatchExtractor(patch_config)
    
    patches = patch_extractor.extract_patches_for_hf_points(
        hagan_result.volatility_surface,
        hf_points,
        hagan_result.strikes,
        hagan_result.maturities
    )
    
    print(f"   Extracted {len(patches)} patches")
    print(f"   Patch size: {patch_config.patch_size}")
    
    # Validate patches
    validation = patch_extractor.validate_patch_extraction(patches)
    print(f"   Validation passed: {validation['valid']}")
    
    # 4. Create point features
    print("\n4. Creating point features...")
    feature_config = FeatureConfig(
        include_sabr_params=True,
        include_derived_features=True,
        normalize_features=True
    )
    feature_engineer = FeatureEngineer(feature_config)
    
    # Create features for each HF point
    point_features_list = []
    for i, hf_point in enumerate(hf_points):
        # Get corresponding Hagan volatility from aligned grid point
        patch = patches[i]
        if patch.hf_point.grid_coords is not None:
            grid_i, grid_j = patch.hf_point.grid_coords
            hagan_vol = hagan_result.volatility_surface[grid_i, grid_j]
        else:
            hagan_vol = 0.2  # Fallback
        
        point_features = feature_engineer.create_point_features(
            sabr_params, hf_point.strike, hf_point.maturity, hagan_vol
        )
        point_features_list.append(point_features)
    
    print(f"   Created features for {len(point_features_list)} points")
    print(f"   Features per point: {len(point_features_list[0].raw_features)}")
    
    # 5. Fit normalization and normalize features
    print("\n5. Fitting normalization...")
    feature_engineer.fit_normalization(point_features_list)
    normalized_features = feature_engineer.normalize_features_batch(point_features_list)
    
    print(f"   Normalization method: {feature_engineer.normalization_stats.method}")
    print(f"   Normalized feature shape: {normalized_features.shape}")
    
    # 6. Prepare MDA-CNN training data
    print("\n6. Preparing MDA-CNN training data...")
    
    # Extract patch data (CNN input)
    patch_data = np.array([patch.patch for patch in patches])
    
    # Extract feature data (MLP input)
    feature_data = normalized_features
    
    # Create target residuals (MC - Hagan)
    target_residuals = []
    for i, hf_point in enumerate(hf_points):
        patch = patches[i]
        if patch.hf_point.grid_coords is not None:
            grid_i, grid_j = patch.hf_point.grid_coords
            hagan_vol = hagan_result.volatility_surface[grid_i, grid_j]
            residual = hf_point.volatility - hagan_vol
        else:
            residual = 0.0
        target_residuals.append(residual)
    
    target_data = np.array(target_residuals)
    
    print(f"   Patch data shape (CNN input): {patch_data.shape}")
    print(f"   Feature data shape (MLP input): {feature_data.shape}")
    print(f"   Target data shape (residuals): {target_data.shape}")
    
    # 7. Display statistics
    print("\n7. Data statistics:")
    print(f"   Patch value range: [{np.min(patch_data):.3f}, {np.max(patch_data):.3f}]")
    print(f"   Feature value range: [{np.min(feature_data):.3f}, {np.max(feature_data):.3f}]")
    print(f"   Residual range: [{np.min(target_data):.4f}, {np.max(target_data):.4f}]")
    print(f"   Mean absolute residual: {np.mean(np.abs(target_data)):.4f}")
    
    # 8. Feature analysis
    print("\n8. Feature analysis:")
    analysis = feature_engineer.get_feature_importance_analysis(point_features_list)
    
    print(f"   Total features: {analysis['feature_count']}")
    print(f"   High correlations found: {len(analysis['high_correlations'])}")
    
    if analysis['high_correlations']:
        print("   Highly correlated feature pairs:")
        for corr in analysis['high_correlations'][:3]:  # Show first 3
            print(f"     {corr['feature1']} <-> {corr['feature2']}: {corr['correlation']:.3f}")
    
    # 9. Visualization (optional)
    print("\n9. Creating visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot original surface
    im1 = axes[0, 0].imshow(hagan_result.volatility_surface, aspect='auto', cmap='viridis')
    axes[0, 0].set_title('Hagan LF Surface')
    axes[0, 0].set_xlabel('Strike Index')
    axes[0, 0].set_ylabel('Maturity Index')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Mark HF points on surface
    for i, patch in enumerate(patches):
        if patch.hf_point.grid_coords is not None:
            grid_i, grid_j = patch.hf_point.grid_coords
            axes[0, 0].plot(grid_j, grid_i, 'ro', markersize=8)
            axes[0, 0].text(grid_j, grid_i, f'{i+1}', color='white', ha='center', va='center')
    
    # Plot sample patches
    for i in range(min(4, len(patches))):
        row = i // 2
        col = (i % 2) + 1
        
        im = axes[row, col].imshow(patches[i].patch, aspect='auto', cmap='viridis')
        axes[row, col].set_title(f'Patch {i+1} (K={hf_points[i].strike:.1f}, T={hf_points[i].maturity:.1f})')
        plt.colorbar(im, ax=axes[row, col])
    
    # Plot residuals
    axes[1, 2].bar(range(len(target_data)), target_data)
    axes[1, 2].set_title('MC - Hagan Residuals')
    axes[1, 2].set_xlabel('HF Point Index')
    axes[1, 2].set_ylabel('Residual')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/patch_extraction_example.png', dpi=150, bbox_inches='tight')
    print("   Visualization saved to 'results/patch_extraction_example.png'")
    
    print("\n=== Example completed successfully! ===")
    
    return {
        'patch_data': patch_data,
        'feature_data': feature_data,
        'target_data': target_data,
        'patches': patches,
        'point_features': point_features_list,
        'sabr_params': sabr_params
    }


if __name__ == "__main__":
    results = main()