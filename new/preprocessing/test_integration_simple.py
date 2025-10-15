"""
Simple integration test for data preprocessing and loading pipeline.

This test verifies that the complete pipeline works end-to-end without
requiring the full test framework.
"""

import numpy as np
import tempfile
import os
from pathlib import Path

from data_generation.sabr_params import SABRParams
from preprocessing.data_loader import (
    DataLoaderConfig, DataSample, HDF5DataStorage, DataSplitter, SABRDataLoader
)
from preprocessing.normalization import DataNormalizer, NormalizationConfig


def create_test_sample(i: int) -> DataSample:
    """Create a single test sample."""
    sabr_params = SABRParams(
        F0=1.0,
        alpha=0.1 + 0.4 * np.random.random(),
        beta=0.3 + 0.6 * np.random.random(),
        nu=0.1 + 0.8 * np.random.random(),
        rho=-0.7 + 1.4 * np.random.random()
    )
    
    return DataSample(
        patch=np.random.randn(9, 9).astype(np.float32),
        point_features=np.random.randn(8).astype(np.float32),
        target_residual=np.random.randn(),
        sample_id=f"test_sample_{i:04d}",
        sabr_params=sabr_params,
        strike=0.8 + 0.4 * np.random.random(),
        maturity=1.0 + 9.0 * np.random.random(),
        hf_volatility=0.1 + 0.4 * np.random.random(),
        lf_volatility=0.1 + 0.4 * np.random.random()
    )


def test_hdf5_storage():
    """Test HDF5 storage functionality."""
    print("Testing HDF5 storage...")
    
    # Create temporary file
    temp_file = tempfile.mktemp(suffix='.h5')
    
    try:
        # Create test samples
        samples = [create_test_sample(i) for i in range(50)]
        
        # Write samples
        with HDF5DataStorage(temp_file, 'w') as storage:
            storage.create_dataset_structure(50, (9, 9), 8)
            storage.write_samples(samples)
        
        # Read samples back
        with HDF5DataStorage(temp_file, 'r') as storage:
            read_samples = storage.read_samples(list(range(50)))
            
            assert len(read_samples) == 50
            
            # Check first sample
            original = samples[0]
            read = read_samples[0]
            
            np.testing.assert_array_almost_equal(original.patch, read.patch)
            np.testing.assert_array_almost_equal(original.point_features, read.point_features)
            assert abs(original.target_residual - read.target_residual) < 1e-6
            assert original.sample_id == read.sample_id
        
        print("✓ HDF5 storage test passed")
        
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_data_splitting():
    """Test data splitting functionality."""
    print("Testing data splitting...")
    
    splitter = DataSplitter(random_seed=42)
    splits = splitter.create_splits(n_samples=100, validation_split=0.15, test_split=0.15)
    
    assert 'train' in splits
    assert 'validation' in splits
    assert 'test' in splits
    
    # Check split sizes
    assert len(splits['test']) == 15
    assert len(splits['validation']) == 15
    assert len(splits['train']) == 70
    
    # Check no overlap
    all_indices = np.concatenate([splits['train'], splits['validation'], splits['test']])
    assert len(np.unique(all_indices)) == 100
    
    print("✓ Data splitting test passed")


def test_normalization():
    """Test normalization functionality."""
    print("Testing normalization...")
    
    # Test different data types
    patches = np.random.randn(50, 9, 9).astype(np.float32)
    features = np.random.randn(50, 8).astype(np.float32)
    targets = np.random.randn(50).astype(np.float32)
    
    # Test patch normalization
    from preprocessing.normalization import PatchNormalizer
    patch_normalizer = PatchNormalizer()
    normalized_patches = patch_normalizer.fit_transform(patches)
    reconstructed_patches = patch_normalizer.inverse_transform(normalized_patches)
    np.testing.assert_array_almost_equal(patches, reconstructed_patches)
    
    # Test feature normalization
    feature_normalizer = DataNormalizer(NormalizationConfig(method='standard'))
    normalized_features = feature_normalizer.fit_transform(features)
    reconstructed_features = feature_normalizer.inverse_transform(normalized_features)
    np.testing.assert_array_almost_equal(features, reconstructed_features)
    
    # Test target normalization
    from preprocessing.normalization import TargetNormalizer
    target_normalizer = TargetNormalizer()
    normalized_targets = target_normalizer.fit_transform(targets)
    reconstructed_targets = target_normalizer.inverse_transform(normalized_targets)
    np.testing.assert_array_almost_equal(targets, reconstructed_targets)
    
    print("✓ Normalization test passed")


def test_data_loader():
    """Test complete data loader functionality."""
    print("Testing data loader...")
    
    # Create temporary files
    data_file = tempfile.mktemp(suffix='.h5')
    splits_file = tempfile.mktemp(suffix='.npz')
    
    try:
        # Create test data
        samples = [create_test_sample(i) for i in range(100)]
        
        # Write to HDF5
        with HDF5DataStorage(data_file, 'w') as storage:
            storage.create_dataset_structure(100, (9, 9), 8)
            storage.write_samples(samples)
        
        # Create splits
        splitter = DataSplitter(random_seed=42)
        splits = splitter.create_splits(100, validation_split=0.15, test_split=0.15)
        splitter.save_splits(splits, splits_file)
        
        # Test data loader
        config = DataLoaderConfig(batch_size=16, shuffle=False)
        loader = SABRDataLoader(config)
        loader.load_from_hdf5(data_file, splits_file)
        
        # Test different splits
        for split_name in ['train', 'validation', 'test']:
            loader.set_split(split_name)
            
            # Get first batch
            batch = next(iter(loader))
            
            assert batch.patches.shape[0] <= 16  # Batch size
            assert batch.patches.shape[1:] == (9, 9)  # Patch shape
            assert batch.point_features.shape[1] == 8  # Feature count
            assert len(batch.sample_ids) == batch.patches.shape[0]
        
        # Close the loader to release file handles
        loader.close()
        
        print("✓ Data loader test passed")
        
    finally:
        for temp_file in [data_file, splits_file]:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


def test_performance():
    """Test performance with larger dataset."""
    print("Testing performance...")
    
    # Create larger dataset
    data_file = tempfile.mktemp(suffix='.h5')
    
    try:
        n_samples = 1000
        samples = [create_test_sample(i) for i in range(n_samples)]
        
        # Write to HDF5
        import time
        start_time = time.time()
        
        with HDF5DataStorage(data_file, 'w') as storage:
            storage.create_dataset_structure(n_samples, (9, 9), 8)
            storage.write_samples(samples)
        
        write_time = time.time() - start_time
        
        # Read back
        start_time = time.time()
        
        with HDF5DataStorage(data_file, 'r') as storage:
            read_samples = storage.read_samples(list(range(100)))  # Read subset
        
        read_time = time.time() - start_time
        
        print(f"  Write time for {n_samples} samples: {write_time:.2f}s")
        print(f"  Read time for 100 samples: {read_time:.3f}s")
        
        # Performance should be reasonable
        assert write_time < 10.0  # Should write in under 10 seconds
        assert read_time < 1.0   # Should read quickly
        
        print("✓ Performance test passed")
        
    finally:
        if os.path.exists(data_file):
            os.unlink(data_file)


def main():
    """Run all integration tests."""
    print("Running data preprocessing and loading integration tests...")
    print("=" * 60)
    
    np.random.seed(42)  # For reproducibility
    
    try:
        test_hdf5_storage()
        test_data_splitting()
        test_normalization()
        test_data_loader()
        test_performance()
        
        print("=" * 60)
        print("✓ All integration tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise


if __name__ == '__main__':
    main()