"""
Example usage of the data preprocessing and loading pipeline.

This script demonstrates how to use the complete data preprocessing and loading
pipeline for SABR volatility surface modeling with MDA-CNN.
"""

import numpy as np
import tempfile
import os
import sys
sys.path.append('.')

from data_generation.sabr_params import SABRParams
from preprocessing.data_loader import (
    DataLoaderConfig, DataSample, HDF5DataStorage, DataSplitter, 
    SABRDataLoader, DataPreprocessor
)
from preprocessing.patch_extractor import PatchConfig
from preprocessing.feature_engineer import FeatureConfig
from preprocessing.normalization import (
    DataNormalizer, PatchNormalizer, TargetNormalizer,
    NormalizationConfig, create_normalization_pipeline
)


def create_example_dataset(n_samples: int = 500) -> list:
    """
    Create an example dataset for demonstration.
    
    Args:
        n_samples: Number of samples to create
        
    Returns:
        List of DataSample objects
    """
    print(f"Creating example dataset with {n_samples} samples...")
    
    samples = []
    np.random.seed(42)  # For reproducibility
    
    for i in range(n_samples):
        # Create realistic SABR parameters
        sabr_params = SABRParams(
            F0=1.0,
            alpha=0.05 + 0.55 * np.random.random(),  # [0.05, 0.6]
            beta=0.3 + 0.6 * np.random.random(),     # [0.3, 0.9]
            nu=0.05 + 0.85 * np.random.random(),     # [0.05, 0.9]
            rho=-0.75 + 1.5 * np.random.random()     # [-0.75, 0.75]
        )
        
        # Create realistic strike and maturity
        strike = 0.5 + 1.0 * np.random.random()  # [0.5, 1.5]
        maturity = 1.0 + 9.0 * np.random.random()  # [1, 10] years
        
        # Create realistic volatilities
        base_vol = 0.15 + 0.25 * np.random.random()  # [0.15, 0.4]
        hf_volatility = base_vol + 0.02 * np.random.randn()
        lf_volatility = base_vol + 0.01 * np.random.randn()
        
        # Create realistic patch (representing local LF surface)
        patch = np.random.normal(base_vol, 0.05, (9, 9)).astype(np.float32)
        
        # Create realistic point features (will be normalized later)
        point_features = np.array([
            sabr_params.F0, sabr_params.alpha, sabr_params.beta,
            sabr_params.nu, sabr_params.rho, strike, maturity, lf_volatility
        ], dtype=np.float32)
        
        # Calculate residual
        target_residual = hf_volatility - lf_volatility
        
        sample = DataSample(
            patch=patch,
            point_features=point_features,
            target_residual=target_residual,
            sample_id=f"example_sample_{i:06d}",
            sabr_params=sabr_params,
            strike=strike,
            maturity=maturity,
            hf_volatility=hf_volatility,
            lf_volatility=lf_volatility
        )
        
        samples.append(sample)
    
    print(f"✓ Created {len(samples)} samples")
    return samples


def demonstrate_hdf5_storage():
    """Demonstrate HDF5 storage functionality."""
    print("\n" + "="*60)
    print("DEMONSTRATING HDF5 STORAGE")
    print("="*60)
    
    # Create example data
    samples = create_example_dataset(100)
    
    # Create temporary HDF5 file
    temp_file = tempfile.mktemp(suffix='.h5')
    
    try:
        print(f"Saving data to HDF5 file: {temp_file}")
        
        # Write data
        with HDF5DataStorage(temp_file, 'w') as storage:
            storage.create_dataset_structure(
                n_samples=len(samples),
                patch_shape=(9, 9),
                n_features=8,
                compression='gzip'
            )
            storage.write_samples(samples)
        
        # Get file info
        file_size_mb = os.path.getsize(temp_file) / (1024 * 1024)
        print(f"✓ Data saved. File size: {file_size_mb:.2f} MB")
        
        # Read data back
        with HDF5DataStorage(temp_file, 'r') as storage:
            info = storage.get_dataset_info()
            print(f"✓ Dataset info: {info['n_samples']} samples, "
                  f"patch shape {info['patch_shape']}, "
                  f"{info['n_features']} features")
            
            # Read a few samples
            read_samples = storage.read_samples([0, 1, 2])
            print(f"✓ Successfully read {len(read_samples)} samples")
            
            # Verify data integrity
            original = samples[0]
            read = read_samples[0]
            
            patch_match = np.allclose(original.patch, read.patch)
            features_match = np.allclose(original.point_features, read.point_features)
            
            print(f"✓ Data integrity check: patch_match={patch_match}, "
                  f"features_match={features_match}")
    
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def demonstrate_data_splitting():
    """Demonstrate data splitting functionality."""
    print("\n" + "="*60)
    print("DEMONSTRATING DATA SPLITTING")
    print("="*60)
    
    # Create SABR parameters for stratified splitting
    sabr_params_list = []
    
    # Create different volatility regimes
    for regime, n_samples in [("low_vol", 200), ("med_vol", 200), ("high_vol", 100)]:
        for _ in range(n_samples):
            if regime == "low_vol":
                alpha = 0.05 + 0.15 * np.random.random()
                nu = 0.05 + 0.25 * np.random.random()
            elif regime == "med_vol":
                alpha = 0.2 + 0.2 * np.random.random()
                nu = 0.3 + 0.3 * np.random.random()
            else:  # high_vol
                alpha = 0.4 + 0.2 * np.random.random()
                nu = 0.6 + 0.3 * np.random.random()
            
            sabr_params = SABRParams(
                F0=1.0, alpha=alpha, beta=0.5, nu=nu, rho=0.0
            )
            sabr_params_list.append(sabr_params)
    
    print(f"Created {len(sabr_params_list)} SABR parameter sets")
    
    # Test different splitting strategies
    splitter = DataSplitter(random_seed=42)
    
    # Random splitting
    random_splits = splitter.create_splits(
        n_samples=len(sabr_params_list),
        validation_split=0.15,
        test_split=0.15
    )
    
    print("Random splits:")
    for split_name, indices in random_splits.items():
        print(f"  {split_name}: {len(indices)} samples")
    
    # Stratified splitting
    stratified_splits = splitter.stratified_split(
        sabr_params_list,
        validation_split=0.15,
        test_split=0.15,
        n_strata=3
    )
    
    print("Stratified splits:")
    for split_name, indices in stratified_splits.items():
        print(f"  {split_name}: {len(indices)} samples")
        
        # Analyze alpha distribution in each split
        alphas = [sabr_params_list[i].alpha for i in indices]
        print(f"    Alpha range: [{np.min(alphas):.3f}, {np.max(alphas):.3f}]")
    
    # Save and load splits
    temp_splits_file = tempfile.mktemp(suffix='.npz')
    
    try:
        splitter.save_splits(stratified_splits, temp_splits_file)
        loaded_splits = splitter.load_splits(temp_splits_file)
        
        print("✓ Successfully saved and loaded splits")
        
        # Verify splits match
        for split_name in stratified_splits:
            np.testing.assert_array_equal(
                stratified_splits[split_name], 
                loaded_splits[split_name]
            )
        
        print("✓ Split integrity verified")
    
    finally:
        if os.path.exists(temp_splits_file):
            os.unlink(temp_splits_file)


def demonstrate_normalization():
    """Demonstrate normalization functionality."""
    print("\n" + "="*60)
    print("DEMONSTRATING NORMALIZATION")
    print("="*60)
    
    # Create test data
    n_samples = 200
    patches = np.random.randn(n_samples, 9, 9).astype(np.float32)
    features = np.random.randn(n_samples, 8).astype(np.float32)
    targets = np.random.randn(n_samples).astype(np.float32)
    
    # Add some realistic scaling
    patches *= 0.1  # Volatility patches around 0.1
    patches += 0.2  # Base volatility level
    
    features[:, 0] = 1.0  # F0 = 1.0
    features[:, 1] *= 0.2  # Alpha scaling
    features[:, 1] += 0.3
    
    targets *= 0.02  # Small residuals
    
    print(f"Created test data: {patches.shape} patches, "
          f"{features.shape} features, {targets.shape} targets")
    
    # Create normalization pipeline
    pipeline = create_normalization_pipeline(
        patch_config=NormalizationConfig(method='standard'),
        feature_config=NormalizationConfig(method='minmax'),
        target_config=NormalizationConfig(method='robust')
    )
    
    print("Created normalization pipeline with:")
    print("  - Standard normalization for patches")
    print("  - Min-max normalization for features")
    print("  - Robust normalization for targets")
    
    # Fit and transform data
    normalized_patches = pipeline['patches'].fit_transform(patches)
    normalized_features = pipeline['features'].fit_transform(features)
    normalized_targets = pipeline['targets'].fit_transform(targets)
    
    print("✓ Data normalized")
    
    # Show normalization statistics
    print("\nNormalization statistics:")
    
    # Patches
    patch_stats = pipeline['patches'].get_statistics_summary()
    print(f"  Patches: method={patch_stats['method']}, "
          f"shape={patch_stats['shape']}")
    
    # Features
    feature_stats = pipeline['features'].get_statistics_summary()
    print(f"  Features: method={feature_stats['method']}, "
          f"shape={feature_stats['shape']}")
    
    # Targets
    target_stats = pipeline['targets'].get_statistics_summary()
    print(f"  Targets: method={target_stats['method']}, "
          f"shape={target_stats['shape']}")
    
    # Test inverse transformation
    reconstructed_patches = pipeline['patches'].inverse_transform(normalized_patches)
    reconstructed_features = pipeline['features'].inverse_transform(normalized_features)
    reconstructed_targets = pipeline['targets'].inverse_transform(normalized_targets)
    
    # Check reconstruction accuracy
    patch_error = np.mean(np.abs(patches - reconstructed_patches))
    feature_error = np.mean(np.abs(features - reconstructed_features))
    target_error = np.mean(np.abs(targets - reconstructed_targets))
    
    print(f"\nReconstruction errors:")
    print(f"  Patches: {patch_error:.2e}")
    print(f"  Features: {feature_error:.2e}")
    print(f"  Targets: {target_error:.2e}")
    
    print("✓ Inverse transformation verified")


def demonstrate_data_loader():
    """Demonstrate complete data loader functionality."""
    print("\n" + "="*60)
    print("DEMONSTRATING DATA LOADER")
    print("="*60)
    
    # Create example dataset
    samples = create_example_dataset(300)
    
    # Create temporary files
    data_file = tempfile.mktemp(suffix='.h5')
    splits_file = tempfile.mktemp(suffix='.npz')
    
    try:
        # Save data to HDF5
        print("Saving data to HDF5...")
        with HDF5DataStorage(data_file, 'w') as storage:
            storage.create_dataset_structure(
                n_samples=len(samples),
                patch_shape=(9, 9),
                n_features=8
            )
            storage.write_samples(samples)
        
        # Create and save splits
        print("Creating data splits...")
        splitter = DataSplitter(random_seed=42)
        splits = splitter.create_splits(
            n_samples=len(samples),
            validation_split=0.15,
            test_split=0.15
        )
        splitter.save_splits(splits, splits_file)
        
        # Create data loader
        config = DataLoaderConfig(
            batch_size=32,
            shuffle=True,
            validation_split=0.15,
            test_split=0.15,
            random_seed=42
        )
        
        loader = SABRDataLoader(config)
        loader.load_from_hdf5(data_file, splits_file)
        
        print("✓ Data loader initialized")
        
        # Get split information
        split_info = loader.get_split_info()
        print("\nSplit information:")
        for split_name, info in split_info.items():
            print(f"  {split_name}: {info['n_samples']} samples, "
                  f"{info['n_batches']} batches")
        
        # Demonstrate batch iteration
        print("\nDemonstrating batch iteration:")
        
        for split_name in ['train', 'validation', 'test']:
            loader.set_split(split_name)
            print(f"\n{split_name.upper()} split:")
            
            batch_count = 0
            total_samples = 0
            
            for batch in loader:
                batch_count += 1
                total_samples += batch.patches.shape[0]
                
                if batch_count == 1:  # Show details for first batch
                    print(f"  First batch: {batch.patches.shape[0]} samples")
                    print(f"    Patch shape: {batch.patches.shape}")
                    print(f"    Features shape: {batch.point_features.shape}")
                    print(f"    Targets shape: {batch.target_residuals.shape}")
                    print(f"    Sample IDs: {batch.sample_ids[:3]}...")
                
                if batch_count >= 3:  # Only show first few batches
                    break
            
            print(f"  Processed {batch_count} batches, {total_samples} samples")
        
        # Demonstrate single sample access
        print("\nDemonstrating single sample access:")
        sample = loader.get_sample(0)
        print(f"  Sample ID: {sample.sample_id}")
        print(f"  SABR params: α={sample.sabr_params.alpha:.3f}, "
              f"β={sample.sabr_params.beta:.3f}")
        print(f"  Strike: {sample.strike:.3f}, Maturity: {sample.maturity:.1f}")
        print(f"  Residual: {sample.target_residual:.4f}")
        
        # Close loader
        loader.close()
        print("✓ Data loader demonstration completed")
    
    finally:
        for temp_file in [data_file, splits_file]:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


def demonstrate_performance():
    """Demonstrate performance characteristics."""
    print("\n" + "="*60)
    print("DEMONSTRATING PERFORMANCE")
    print("="*60)
    
    import time
    
    # Test with different dataset sizes
    for n_samples in [100, 500, 1000]:
        print(f"\nTesting with {n_samples} samples:")
        
        # Create data
        start_time = time.time()
        samples = create_example_dataset(n_samples)
        creation_time = time.time() - start_time
        
        # Save to HDF5
        temp_file = tempfile.mktemp(suffix='.h5')
        
        try:
            start_time = time.time()
            with HDF5DataStorage(temp_file, 'w') as storage:
                storage.create_dataset_structure(n_samples, (9, 9), 8)
                storage.write_samples(samples)
            write_time = time.time() - start_time
            
            # Read from HDF5
            start_time = time.time()
            with HDF5DataStorage(temp_file, 'r') as storage:
                read_samples = storage.read_samples(list(range(min(100, n_samples))))
            read_time = time.time() - start_time
            
            # File size
            file_size_mb = os.path.getsize(temp_file) / (1024 * 1024)
            
            print(f"  Creation time: {creation_time:.3f}s")
            print(f"  Write time: {write_time:.3f}s")
            print(f"  Read time (100 samples): {read_time:.3f}s")
            print(f"  File size: {file_size_mb:.2f} MB")
            print(f"  Compression ratio: {(n_samples * 9 * 9 * 4 + n_samples * 8 * 4) / (1024 * 1024) / file_size_mb:.1f}x")
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


def main():
    """Run all demonstrations."""
    print("SABR VOLATILITY SURFACE DATA PREPROCESSING AND LOADING PIPELINE")
    print("=" * 80)
    print("This script demonstrates the complete data preprocessing and loading")
    print("pipeline for MDA-CNN volatility surface modeling.")
    print("=" * 80)
    
    try:
        demonstrate_hdf5_storage()
        demonstrate_data_splitting()
        demonstrate_normalization()
        demonstrate_data_loader()
        demonstrate_performance()
        
        print("\n" + "="*80)
        print("✓ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nThe data preprocessing and loading pipeline is ready for use.")
        print("Key features demonstrated:")
        print("  • Efficient HDF5 storage with compression")
        print("  • Flexible data splitting (random and stratified)")
        print("  • Comprehensive normalization (patches, features, targets)")
        print("  • High-performance batch loading with shuffling")
        print("  • Memory-efficient processing of large datasets")
        
    except Exception as e:
        print(f"\n✗ Demonstration failed: {e}")
        raise


if __name__ == '__main__':
    main()