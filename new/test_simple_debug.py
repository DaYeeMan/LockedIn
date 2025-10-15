"""
Simple debug test for data preprocessing components.
"""

import numpy as np
import tempfile
import os
import sys
sys.path.append('.')

from data_generation.sabr_params import SABRParams
from preprocessing.data_loader import DataSample, HDF5DataStorage


def test_basic_functionality():
    """Test basic functionality step by step."""
    print("Testing basic SABR params...")
    
    # Test SABR params
    sabr_params = SABRParams(F0=1.0, alpha=0.2, beta=0.5, nu=0.3, rho=-0.2)
    print(f"SABR params: {sabr_params}")
    
    # Test data sample creation
    print("Testing data sample creation...")
    sample = DataSample(
        patch=np.random.randn(9, 9).astype(np.float32),
        point_features=np.random.randn(8).astype(np.float32),
        target_residual=0.05,
        sample_id="test_001",
        sabr_params=sabr_params,
        strike=1.0,
        maturity=5.0,
        hf_volatility=0.25,
        lf_volatility=0.20
    )
    print(f"Sample created: {sample.sample_id}")
    
    # Test HDF5 storage
    print("Testing HDF5 storage...")
    temp_file = tempfile.mktemp(suffix='.h5')
    
    try:
        with HDF5DataStorage(temp_file, 'w') as storage:
            storage.create_dataset_structure(1, (9, 9), 8)
            storage.write_samples([sample])
            print("Sample written to HDF5")
        
        with HDF5DataStorage(temp_file, 'r') as storage:
            read_samples = storage.read_samples([0])
            read_sample = read_samples[0]
            
            print(f"Read sample: {read_sample.sample_id}")
            print(f"Patch shape: {read_sample.patch.shape}")
            print(f"Features shape: {read_sample.point_features.shape}")
            print(f"Target residual: {read_sample.target_residual}")
            
            # Check values match
            patch_match = np.allclose(sample.patch, read_sample.patch)
            features_match = np.allclose(sample.point_features, read_sample.point_features)
            residual_match = abs(sample.target_residual - read_sample.target_residual) < 1e-6
            id_match = sample.sample_id == read_sample.sample_id
            
            print(f"Patch match: {patch_match}")
            print(f"Features match: {features_match}")
            print(f"Residual match: {residual_match}")
            print(f"ID match: {id_match}")
            
            if all([patch_match, features_match, residual_match, id_match]):
                print("✓ HDF5 storage test passed")
            else:
                print("✗ HDF5 storage test failed")
                return False
    
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    return True


if __name__ == '__main__':
    success = test_basic_functionality()
    if success:
        print("\n✓ All basic tests passed!")
    else:
        print("\n✗ Some tests failed!")