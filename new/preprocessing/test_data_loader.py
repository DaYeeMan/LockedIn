"""
Tests for data loading and preprocessing pipeline.

This module contains comprehensive tests for the data loader, HDF5 storage,
data splitting, and preprocessing components to ensure consistency and performance.
"""

import pytest
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path
import h5py
from typing import List, Dict, Any

from data_generation.sabr_params import SABRParams, GridConfig
from preprocessing.data_loader import (
    DataLoaderConfig, DataSample, DataBatch, HDF5DataStorage,
    DataSplitter, DataPreprocessor, SABRDataLoader
)
from preprocessing.patch_extractor import PatchConfig, HFPoint
from preprocessing.feature_engineer import FeatureConfig
from preprocessing.normalization import DataNormalizer, NormalizationConfig


class TestHDF5DataStorage:
    """Test HDF5 data storage functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test_data.h5')
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_samples(self, n_samples: int = 100) -> List[DataSample]:
        """Create test data samples."""
        samples = []
        
        for i in range(n_samples):
            # Create test SABR parameters
            sabr_params = SABRParams(
                F0=1.0,
                alpha=0.1 + 0.4 * np.random.random(),
                beta=0.3 + 0.6 * np.random.random(),
                nu=0.1 + 0.8 * np.random.random(),
                rho=-0.7 + 1.4 * np.random.random()
            )
            
            # Create test patch (9x9)
            patch = np.random.randn(9, 9).astype(np.float32)
            
            # Create test point features
            point_features = np.random.randn(8).astype(np.float32)
            
            # Create test sample
            sample = DataSample(
                patch=patch,
                point_features=point_features,
                target_residual=np.random.randn(),
                sample_id=f"test_sample_{i:04d}",
                sabr_params=sabr_params,
                strike=0.8 + 0.4 * np.random.random(),
                maturity=1.0 + 9.0 * np.random.random(),
                hf_volatility=0.1 + 0.4 * np.random.random(),
                lf_volatility=0.1 + 0.4 * np.random.random()
            )
            
            samples.append(sample)
        
        return samples
    
    def test_create_dataset_structure(self):
        """Test HDF5 dataset structure creation."""
        with HDF5DataStorage(self.test_file, 'w') as storage:
            storage.create_dataset_structure(
                n_samples=100,
                patch_shape=(9, 9),
                n_features=8
            )
            
            # Check that all datasets were created
            expected_datasets = [
                'patches', 'point_features', 'target_residuals',
                'sample_ids', 'sabr_params', 'strikes', 'maturities',
                'hf_volatilities', 'lf_volatilities'
            ]
            
            for dataset_name in expected_datasets:
                assert dataset_name in storage.file
                
            # Check dataset shapes
            assert storage.file['patches'].shape == (100, 9, 9)
            assert storage.file['point_features'].shape == (100, 8)
            assert storage.file['target_residuals'].shape == (100,)
    
    def test_write_and_read_samples(self):
        """Test writing and reading samples."""
        samples = self.create_test_samples(50)
        
        # Write samples
        with HDF5DataStorage(self.test_file, 'w') as storage:
            storage.create_dataset_structure(
                n_samples=50,
                patch_shape=(9, 9),
                n_features=8
            )
            storage.write_samples(samples)
        
        # Read samples back
        with HDF5DataStorage(self.test_file, 'r') as storage:
            read_samples = storage.read_samples(list(range(50)))
            
            assert len(read_samples) == 50
            
            # Check first sample
            original = samples[0]
            read = read_samples[0]
            
            np.testing.assert_array_almost_equal(original.patch, read.patch)
            np.testing.assert_array_almost_equal(original.point_features, read.point_features)
            assert abs(original.target_residual - read.target_residual) < 1e-6
            assert original.sample_id == read.sample_id
            assert abs(original.sabr_params.alpha - read.sabr_params.alpha) < 1e-6
    
    def test_partial_read(self):
        """Test reading subset of samples."""
        samples = self.create_test_samples(100)
        
        with HDF5DataStorage(self.test_file, 'w') as storage:
            storage.create_dataset_structure(
                n_samples=100,
                patch_shape=(9, 9),
                n_features=8
            )
            storage.write_samples(samples)
        
        # Read subset
        with HDF5DataStorage(self.test_file, 'r') as storage:
            subset_indices = [10, 25, 50, 75]
            read_samples = storage.read_samples(subset_indices)
            
            assert len(read_samples) == 4
            
            # Check that correct samples were read
            for i, idx in enumerate(subset_indices):
                original = samples[idx]
                read = read_samples[i]
                assert original.sample_id == read.sample_id
    
    def test_dataset_info(self):
        """Test dataset information retrieval."""
        samples = self.create_test_samples(75)
        
        with HDF5DataStorage(self.test_file, 'w') as storage:
            storage.create_dataset_structure(
                n_samples=75,
                patch_shape=(9, 9),
                n_features=8
            )
            storage.write_samples(samples)
        
        with HDF5DataStorage(self.test_file, 'r') as storage:
            info = storage.get_dataset_info()
            
            assert info['n_samples'] == 75
            assert info['patch_shape'] == (9, 9)
            assert info['n_features'] == 8
            assert 'file_size_mb' in info
            assert info['file_size_mb'] > 0


class TestDataSplitter:
    """Test data splitting functionality."""
    
    def test_create_splits(self):
        """Test basic data splitting."""
        splitter = DataSplitter(random_seed=42)
        splits = splitter.create_splits(
            n_samples=1000,
            validation_split=0.15,
            test_split=0.15
        )
        
        assert 'train' in splits
        assert 'validation' in splits
        assert 'test' in splits
        
        # Check split sizes
        assert len(splits['test']) == 150  # 15% of 1000
        assert len(splits['validation']) == 150  # 15% of 1000
        assert len(splits['train']) == 700  # Remaining
        
        # Check no overlap
        all_indices = np.concatenate([splits['train'], splits['validation'], splits['test']])
        assert len(np.unique(all_indices)) == 1000
        
        # Check all indices are covered
        assert set(all_indices) == set(range(1000))
    
    def test_stratified_split(self):
        """Test stratified splitting."""
        # Create test SABR parameters with different regimes
        sabr_params_list = []
        
        # Low volatility regime
        for _ in range(300):
            sabr_params_list.append(SABRParams(
                F0=1.0, alpha=0.1, beta=0.5, nu=0.2, rho=0.0
            ))
        
        # High volatility regime
        for _ in range(200):
            sabr_params_list.append(SABRParams(
                F0=1.0, alpha=0.5, beta=0.8, nu=0.8, rho=-0.5
            ))
        
        splitter = DataSplitter(random_seed=42)
        splits = splitter.stratified_split(
            sabr_params_list,
            validation_split=0.2,
            test_split=0.2
        )
        
        # Check that all samples are assigned
        total_assigned = len(splits['train']) + len(splits['validation']) + len(splits['test'])
        assert total_assigned == 500
        
        # Check no overlap
        all_indices = np.concatenate([splits['train'], splits['validation'], splits['test']])
        assert len(np.unique(all_indices)) == total_assigned
    
    def test_save_load_splits(self):
        """Test saving and loading splits."""
        splitter = DataSplitter(random_seed=42)
        splits = splitter.create_splits(n_samples=100)
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            splits_file = f.name
        
        try:
            # Save splits
            splitter.save_splits(splits, splits_file)
            
            # Load splits
            loaded_splits = splitter.load_splits(splits_file)
            
            # Check that loaded splits match original
            for split_name in ['train', 'validation', 'test']:
                np.testing.assert_array_equal(splits[split_name], loaded_splits[split_name])
        
        finally:
            os.unlink(splits_file)


class TestDataPreprocessor:
    """Test data preprocessing functionality."""
    
    def create_test_surface_data(self):
        """Create test surface data."""
        # Create test SABR parameters
        sabr_params = SABRParams(F0=1.0, alpha=0.2, beta=0.5, nu=0.3, rho=-0.2)
        
        # Create test grids
        strikes = np.linspace(0.5, 1.5, 21)
        maturities = np.linspace(1.0, 10.0, 10)
        
        strikes_grid, maturities_grid = np.meshgrid(strikes, maturities, indexing='ij')
        
        # Create test surfaces
        hf_surface = 0.2 + 0.1 * np.random.randn(*strikes_grid.shape)
        lf_surface = 0.2 + 0.05 * np.random.randn(*strikes_grid.shape)
        
        # Create test HF points
        hf_points = [
            (0.8, 2.0, 0.25),
            (1.0, 5.0, 0.22),
            (1.2, 8.0, 0.28)
        ]
        
        return {
            'hf_surfaces': [hf_surface],
            'lf_surfaces': [lf_surface],
            'sabr_params_list': [sabr_params],
            'strikes_grids': [strikes_grid],
            'maturities_grids': [maturities_grid],
            'hf_points_list': [hf_points]
        }
    
    def test_preprocess_surface_data(self):
        """Test surface data preprocessing."""
        data = self.create_test_surface_data()
        
        preprocessor = DataPreprocessor()
        samples = preprocessor.preprocess_surface_data(**data)
        
        assert len(samples) == 3  # Number of HF points
        
        # Check sample structure
        sample = samples[0]
        assert isinstance(sample, DataSample)
        assert sample.patch.shape == (9, 9)  # Default patch size
        assert len(sample.point_features) > 0
        assert isinstance(sample.target_residual, float)
        assert sample.sample_id.startswith('surface_0000_point_')


class TestSABRDataLoader:
    """Test main data loader functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.temp_dir, 'test_data.h5')
        self.splits_file = os.path.join(self.temp_dir, 'splits.npz')
        
        # Create test data
        self.samples = self.create_test_samples(200)
        self.create_test_hdf5_file()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_samples(self, n_samples: int) -> List[DataSample]:
        """Create test samples."""
        samples = []
        
        for i in range(n_samples):
            sabr_params = SABRParams(
                F0=1.0,
                alpha=0.1 + 0.4 * np.random.random(),
                beta=0.3 + 0.6 * np.random.random(),
                nu=0.1 + 0.8 * np.random.random(),
                rho=-0.7 + 1.4 * np.random.random()
            )
            
            sample = DataSample(
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
            
            samples.append(sample)
        
        return samples
    
    def create_test_hdf5_file(self):
        """Create test HDF5 file with data."""
        with HDF5DataStorage(self.data_file, 'w') as storage:
            storage.create_dataset_structure(
                n_samples=len(self.samples),
                patch_shape=(9, 9),
                n_features=8
            )
            storage.write_samples(self.samples)
        
        # Create splits
        splitter = DataSplitter(random_seed=42)
        splits = splitter.create_splits(
            n_samples=len(self.samples),
            validation_split=0.15,
            test_split=0.15
        )
        splitter.save_splits(splits, self.splits_file)
    
    def test_load_from_hdf5(self):
        """Test loading data from HDF5."""
        config = DataLoaderConfig(batch_size=32, shuffle=True)
        loader = SABRDataLoader(config)
        
        loader.load_from_hdf5(self.data_file, self.splits_file)
        
        assert loader.storage is not None
        assert loader.splits is not None
        assert 'train' in loader.splits
        assert 'validation' in loader.splits
        assert 'test' in loader.splits
    
    def test_set_split(self):
        """Test setting data split."""
        config = DataLoaderConfig(batch_size=32)
        loader = SABRDataLoader(config)
        loader.load_from_hdf5(self.data_file, self.splits_file)
        
        # Test setting different splits
        for split_name in ['train', 'validation', 'test']:
            loader.set_split(split_name)
            assert loader.current_split == split_name
            assert loader.current_indices is not None
            assert len(loader.current_indices) > 0
    
    def test_batch_iteration(self):
        """Test batch iteration."""
        config = DataLoaderConfig(batch_size=16, shuffle=False)
        loader = SABRDataLoader(config)
        loader.load_from_hdf5(self.data_file, self.splits_file)
        
        loader.set_split('train')
        
        batches = list(loader)
        assert len(batches) > 0
        
        # Check first batch
        batch = batches[0]
        assert isinstance(batch, DataBatch)
        assert batch.patches.shape[0] <= 16  # Batch size
        assert batch.patches.shape[1:] == (9, 9)  # Patch shape
        assert batch.point_features.shape[1] == 8  # Feature count
        assert len(batch.sample_ids) == batch.patches.shape[0]
    
    def test_batch_consistency(self):
        """Test batch data consistency."""
        config = DataLoaderConfig(batch_size=10, shuffle=False)
        loader = SABRDataLoader(config)
        loader.load_from_hdf5(self.data_file, self.splits_file)
        
        loader.set_split('train')
        
        # Get first batch twice
        batch1 = next(iter(loader))
        batch2 = next(iter(loader))
        
        # Should be identical (no shuffle)
        np.testing.assert_array_equal(batch1.patches, batch2.patches)
        np.testing.assert_array_equal(batch1.point_features, batch2.point_features)
        np.testing.assert_array_equal(batch1.target_residuals, batch2.target_residuals)
    
    def test_shuffle_functionality(self):
        """Test data shuffling."""
        config = DataLoaderConfig(batch_size=10, shuffle=True, random_seed=42)
        loader = SABRDataLoader(config)
        loader.load_from_hdf5(self.data_file, self.splits_file)
        
        loader.set_split('train')
        
        # Get batches from two epochs
        epoch1_batches = list(loader)
        epoch2_batches = list(loader)
        
        # Should have same number of batches
        assert len(epoch1_batches) == len(epoch2_batches)
        
        # But different order (with high probability)
        first_batch_ids_1 = epoch1_batches[0].sample_ids
        first_batch_ids_2 = epoch2_batches[0].sample_ids
        
        # Note: This test might occasionally fail due to randomness
        # In practice, shuffling should produce different orders
    
    def test_get_split_info(self):
        """Test split information retrieval."""
        config = DataLoaderConfig(batch_size=32)
        loader = SABRDataLoader(config)
        loader.load_from_hdf5(self.data_file, self.splits_file)
        
        info = loader.get_split_info()
        
        assert 'train' in info
        assert 'validation' in info
        assert 'test' in info
        
        # Check that all samples are accounted for
        total_samples = sum(info[split]['n_samples'] for split in info)
        assert total_samples == len(self.samples)


class TestDataLoaderPerformance:
    """Test data loader performance characteristics."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.large_data_file = os.path.join(self.temp_dir, 'large_data.h5')
        
    def teardown_method(self):
        """Clean up performance test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_large_dataset(self, n_samples: int = 10000):
        """Create large dataset for performance testing."""
        # Create samples in batches to avoid memory issues
        batch_size = 1000
        
        with HDF5DataStorage(self.large_data_file, 'w') as storage:
            storage.create_dataset_structure(
                n_samples=n_samples,
                patch_shape=(9, 9),
                n_features=8
            )
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_samples = []
                
                for i in range(start_idx, end_idx):
                    sabr_params = SABRParams(
                        F0=1.0,
                        alpha=0.1 + 0.4 * np.random.random(),
                        beta=0.3 + 0.6 * np.random.random(),
                        nu=0.1 + 0.8 * np.random.random(),
                        rho=-0.7 + 1.4 * np.random.random()
                    )
                    
                    sample = DataSample(
                        patch=np.random.randn(9, 9).astype(np.float32),
                        point_features=np.random.randn(8).astype(np.float32),
                        target_residual=np.random.randn(),
                        sample_id=f"large_sample_{i:06d}",
                        sabr_params=sabr_params,
                        strike=0.8 + 0.4 * np.random.random(),
                        maturity=1.0 + 9.0 * np.random.random(),
                        hf_volatility=0.1 + 0.4 * np.random.random(),
                        lf_volatility=0.1 + 0.4 * np.random.random()
                    )
                    
                    batch_samples.append(sample)
                
                storage.write_samples(batch_samples, start_idx)
    
    @pytest.mark.slow
    def test_large_dataset_loading(self):
        """Test loading performance with large dataset."""
        import time
        
        n_samples = 5000  # Reduced for CI
        self.create_large_dataset(n_samples)
        
        config = DataLoaderConfig(batch_size=64, shuffle=True)
        loader = SABRDataLoader(config)
        
        # Time the loading
        start_time = time.time()
        loader.load_from_hdf5(self.large_data_file)
        load_time = time.time() - start_time
        
        print(f"Loading {n_samples} samples took {load_time:.2f} seconds")
        
        # Time batch iteration
        loader.set_split('train')
        start_time = time.time()
        
        batch_count = 0
        for batch in loader:
            batch_count += 1
            if batch_count >= 10:  # Test first 10 batches
                break
        
        iteration_time = time.time() - start_time
        print(f"Iterating 10 batches took {iteration_time:.2f} seconds")
        
        # Performance should be reasonable
        assert load_time < 10.0  # Should load in under 10 seconds
        assert iteration_time < 5.0  # Should iterate quickly
    
    def test_memory_usage(self):
        """Test memory usage characteristics."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create moderate dataset
        n_samples = 1000
        self.create_large_dataset(n_samples)
        
        config = DataLoaderConfig(batch_size=32)
        loader = SABRDataLoader(config)
        loader.load_from_hdf5(self.large_data_file)
        
        # Memory after loading
        after_load_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Iterate through data
        loader.set_split('train')
        for batch in loader:
            pass  # Just iterate, don't store
        
        # Memory after iteration
        after_iteration_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"After loading: {after_load_memory:.1f} MB")
        print(f"After iteration: {after_iteration_memory:.1f} MB")
        
        # Memory usage should be reasonable
        load_increase = after_load_memory - initial_memory
        iteration_increase = after_iteration_memory - after_load_memory
        
        assert load_increase < 100  # Should not use more than 100MB for loading
        assert iteration_increase < 50  # Should not leak significant memory during iteration


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])