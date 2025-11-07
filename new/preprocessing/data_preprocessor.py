"""
Data preprocessor for SABR volatility surface modeling.

This module converts raw volatility surface data into training-ready format
for MDA-CNN and baseline models. It handles patch extraction, feature engineering,
data splitting, and HDF5 storage.
"""

import numpy as np
import h5py
import os
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import time
import logging

from data_generation.sabr_params import SABRParams
from data_generation.sabr_mc_generator import MCResult
from data_generation.hagan_surface_generator import HaganResult
from preprocessing.patch_extractor import PatchExtractor, PatchConfig, HFPoint
from preprocessing.feature_engineer import FeatureEngineer, FeatureConfig


@dataclass
class PreprocessingConfig:
    """
    Configuration for data preprocessing.
    
    Attributes:
        patch_size: Size of surface patches for CNN
        output_dir: Directory to save preprocessed data
        validation_split: Fraction of data for validation
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility
        normalize_patches: Whether to normalize patches
        normalize_features: Whether to normalize point features
        create_hdf5: Whether to create HDF5 files for efficient loading
        batch_size: Batch size for data loading
        min_samples_per_surface: Minimum samples to extract per surface
        max_samples_per_surface: Maximum samples to extract per surface
    """
    patch_size: int = 9
    output_dir: str = "data/processed"
    validation_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    normalize_patches: bool = True
    normalize_features: bool = True
    create_hdf5: bool = True
    batch_size: int = 64
    min_samples_per_surface: int = 10
    max_samples_per_surface: int = 50


@dataclass
class PreprocessingResult:
    """
    Result from data preprocessing.
    
    Attributes:
        n_training_samples: Number of training samples
        n_validation_samples: Number of validation samples
        n_test_samples: Number of test samples
        file_paths: Paths to created files
        computation_time: Time taken for preprocessing
        feature_stats: Statistics for feature normalization
        patch_stats: Statistics for patch normalization
    """
    n_training_samples: int
    n_validation_samples: int
    n_test_samples: int
    file_paths: Dict[str, str]
    computation_time: float
    feature_stats: Dict[str, Any]
    patch_stats: Dict[str, Any]


class DataPreprocessor:
    """
    Preprocessor for converting raw surface data to training format.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize data preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        patch_config = PatchConfig(
            patch_size=(config.patch_size, config.patch_size),
            normalize_patches=config.normalize_patches
        )
        self.patch_extractor = PatchExtractor(patch_config)
        
        feature_config = FeatureConfig(
            normalize_features=config.normalize_features
        )
        self.feature_engineer = FeatureEngineer(feature_config)
        
        # Set random seed
        np.random.seed(config.random_seed)
    
    def process_raw_data(self, parameter_sets: List[SABRParams], 
                        mc_results: List[MCResult], 
                        hagan_results: List[HaganResult]) -> PreprocessingResult:
        """
        Process raw volatility surface data into training format.
        
        Args:
            parameter_sets: List of SABR parameters
            mc_results: List of Monte Carlo results
            hagan_results: List of Hagan analytical results
            
        Returns:
            PreprocessingResult with processing statistics
        """
        start_time = time.time()
        
        self.logger.info("Starting data preprocessing...")
        
        # Extract training samples
        all_samples = self._extract_all_samples(parameter_sets, mc_results, hagan_results)
        
        # Split data
        train_samples, val_samples, test_samples = self._split_data(all_samples)
        
        # Normalize data
        feature_stats, patch_stats = self._compute_normalization_stats(train_samples)
        self._apply_normalization(all_samples, feature_stats, patch_stats)
        
        # Save data
        file_paths = self._save_processed_data(train_samples, val_samples, test_samples,
                                             feature_stats, patch_stats)
        
        computation_time = time.time() - start_time
        
        self.logger.info(f"Preprocessing completed in {computation_time:.2f} seconds")
        
        return PreprocessingResult(
            n_training_samples=len(train_samples),
            n_validation_samples=len(val_samples),
            n_test_samples=len(test_samples),
            file_paths=file_paths,
            computation_time=computation_time,
            feature_stats=feature_stats,
            patch_stats=patch_stats
        )
    
    def _extract_all_samples(self, parameter_sets: List[SABRParams], 
                           mc_results: List[MCResult], 
                           hagan_results: List[HaganResult]) -> List[Dict[str, Any]]:
        """Extract all training samples from raw data."""
        all_samples = []
        
        for i, (params, mc_result, hagan_result) in enumerate(zip(parameter_sets, mc_results, hagan_results)):
            try:
                samples = self._extract_samples_from_surface(params, mc_result, hagan_result, i)
                all_samples.extend(samples)
            except Exception as e:
                self.logger.warning(f"Failed to extract samples from surface {i}: {e}")
                continue
        
        self.logger.info(f"Extracted {len(all_samples)} total samples")
        return all_samples
    
    def _extract_samples_from_surface(self, params: SABRParams, mc_result: MCResult, 
                                    hagan_result: HaganResult, surface_id: int) -> List[Dict[str, Any]]:
        """Extract training samples from a single surface pair."""
        samples = []
        
        # Get valid points where both MC and Hagan have values
        mc_surface = mc_result.volatility_surface
        hagan_surface = hagan_result.volatility_surface
        
        valid_mask = (~np.isnan(mc_surface)) & (~np.isnan(hagan_surface)) & (mc_surface > 0) & (hagan_surface > 0)
        
        if not np.any(valid_mask):
            return samples
        
        # Sample points strategically
        valid_indices = np.where(valid_mask)
        n_valid = len(valid_indices[0])
        
        # Determine number of samples to extract
        n_samples = min(max(self.config.min_samples_per_surface, n_valid // 4), 
                       min(self.config.max_samples_per_surface, n_valid))
        
        if n_samples == 0:
            return samples
        
        # Sample indices
        if n_samples >= n_valid:
            sample_indices = list(range(n_valid))
        else:
            sample_indices = np.random.choice(n_valid, n_samples, replace=False)
        
        for idx in sample_indices:
            i, j = valid_indices[0][idx], valid_indices[1][idx]
            
            try:
                # Get strike and maturity
                strike = mc_result.strikes[i][j] if mc_result.strikes.ndim > 1 else mc_result.strikes[j]
                maturity = mc_result.maturities[i] if len(mc_result.maturities) > i else mc_result.maturities[0]
                
                # Get volatilities
                mc_vol = mc_surface[i, j]
                hagan_vol = hagan_surface[i, j]
                residual = mc_vol - hagan_vol
                
                # Create HF point
                hf_point = HFPoint(
                    strike=strike,
                    maturity=maturity,
                    volatility=mc_vol,
                    grid_coords=(i, j)
                )
                
                # Extract patch
                patch = self._extract_patch_simple(hagan_surface, i, j)
                
                # Extract features
                features = self._extract_features_simple(params, strike, maturity)
                
                sample = {
                    'patch': patch,
                    'features': features,
                    'target': residual,
                    'surface_id': surface_id,
                    'grid_coords': (i, j),
                    'strike': strike,
                    'maturity': maturity,
                    'mc_vol': mc_vol,
                    'hagan_vol': hagan_vol,
                    'sabr_params': params
                }
                
                samples.append(sample)
                
            except Exception as e:
                self.logger.debug(f"Failed to extract sample at ({i}, {j}): {e}")
                continue
        
        return samples
    
    def _extract_patch_simple(self, surface: np.ndarray, center_i: int, center_j: int) -> np.ndarray:
        """Extract a patch around the given center point."""
        patch_size = self.config.patch_size
        half_size = patch_size // 2
        
        # Calculate patch boundaries
        i_start = max(0, center_i - half_size)
        i_end = min(surface.shape[0], center_i + half_size + 1)
        j_start = max(0, center_j - half_size)
        j_end = min(surface.shape[1], center_j + half_size + 1)
        
        # Extract patch
        patch = surface[i_start:i_end, j_start:j_end]
        
        # Pad if necessary
        if patch.shape != (patch_size, patch_size):
            padded_patch = np.full((patch_size, patch_size), np.nan)
            
            # Calculate where to place the patch in the padded array
            pad_i_start = max(0, half_size - (center_i - i_start))
            pad_j_start = max(0, half_size - (center_j - j_start))
            pad_i_end = pad_i_start + patch.shape[0]
            pad_j_end = pad_j_start + patch.shape[1]
            
            padded_patch[pad_i_start:pad_i_end, pad_j_start:pad_j_end] = patch
            patch = padded_patch
        
        return patch
    
    def _extract_features_simple(self, params: SABRParams, strike: float, maturity: float) -> np.ndarray:
        """Extract point features for the given parameters."""
        features = np.array([
            params.alpha,
            params.beta,
            params.nu,
            params.rho,
            params.F0,
            strike,
            maturity,
            strike / params.F0,  # Moneyness
            np.log(strike / params.F0),  # Log moneyness
            np.sqrt(maturity)  # Square root of time
        ])
        
        return features
    
    def _split_data(self, samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split data into train/validation/test sets."""
        n_samples = len(samples)
        
        # Shuffle samples
        indices = np.random.permutation(n_samples)
        
        # Calculate split sizes
        n_test = int(n_samples * self.config.test_split)
        n_val = int(n_samples * self.config.validation_split)
        n_train = n_samples - n_test - n_val
        
        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Create splits
        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]
        test_samples = [samples[i] for i in test_indices]
        
        self.logger.info(f"Data split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
        
        return train_samples, val_samples, test_samples
    
    def _compute_normalization_stats(self, train_samples: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compute normalization statistics from training data."""
        if not self.config.normalize_features and not self.config.normalize_patches:
            return {}, {}
        
        feature_stats = {}
        patch_stats = {}
        
        if self.config.normalize_features:
            # Collect all features
            all_features = np.array([sample['features'] for sample in train_samples])
            
            feature_stats = {
                'mean': np.mean(all_features, axis=0),
                'std': np.std(all_features, axis=0) + 1e-8  # Add small epsilon
            }
        
        if self.config.normalize_patches:
            # Collect all patches
            all_patches = []
            for sample in train_samples:
                patch = sample['patch']
                valid_values = patch[~np.isnan(patch)]
                if len(valid_values) > 0:
                    all_patches.extend(valid_values)
            
            if all_patches:
                patch_stats = {
                    'mean': np.mean(all_patches),
                    'std': np.std(all_patches) + 1e-8
                }
        
        return feature_stats, patch_stats
    
    def _apply_normalization(self, all_samples: List[Dict[str, Any]], 
                           feature_stats: Dict[str, Any], patch_stats: Dict[str, Any]):
        """Apply normalization to all samples."""
        for sample in all_samples:
            if self.config.normalize_features and feature_stats:
                sample['features'] = (sample['features'] - feature_stats['mean']) / feature_stats['std']
            
            if self.config.normalize_patches and patch_stats:
                patch = sample['patch']
                valid_mask = ~np.isnan(patch)
                patch[valid_mask] = (patch[valid_mask] - patch_stats['mean']) / patch_stats['std']
                sample['patch'] = patch
    
    def _save_processed_data(self, train_samples: List[Dict[str, Any]], 
                           val_samples: List[Dict[str, Any]], 
                           test_samples: List[Dict[str, Any]],
                           feature_stats: Dict[str, Any], 
                           patch_stats: Dict[str, Any]) -> Dict[str, str]:
        """Save processed data to files."""
        file_paths = {}
        
        # Save as pickle files (simple format)
        splits = {'train': train_samples, 'val': val_samples, 'test': test_samples}
        
        for split_name, samples in splits.items():
            pickle_path = os.path.join(self.config.output_dir, f"{split_name}_data.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(samples, f)
            file_paths[f"{split_name}_pickle"] = pickle_path
        
        # Save normalization stats
        stats_path = os.path.join(self.config.output_dir, "normalization_stats.json")
        stats = {
            'feature_stats': feature_stats,
            'patch_stats': patch_stats,
            'config': {
                'patch_size': self.config.patch_size,
                'normalize_features': self.config.normalize_features,
                'normalize_patches': self.config.normalize_patches
            }
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            else:
                return obj
        
        stats = convert_numpy(stats)
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        file_paths['stats'] = stats_path
        
        # Create HDF5 files if requested
        if self.config.create_hdf5:
            hdf5_paths = self._create_hdf5_files(splits)
            file_paths.update(hdf5_paths)
        
        return file_paths
    
    def _create_hdf5_files(self, splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
        """Create HDF5 files for efficient data loading."""
        hdf5_paths = {}
        
        for split_name, samples in splits.items():
            if not samples:
                continue
                
            hdf5_path = os.path.join(self.config.output_dir, f"{split_name}_data.h5")
            
            with h5py.File(hdf5_path, 'w') as f:
                n_samples = len(samples)
                patch_shape = samples[0]['patch'].shape
                feature_dim = len(samples[0]['features'])
                
                # Create datasets
                patches_ds = f.create_dataset('patches', (n_samples,) + patch_shape, dtype=np.float32)
                features_ds = f.create_dataset('features', (n_samples, feature_dim), dtype=np.float32)
                targets_ds = f.create_dataset('targets', (n_samples,), dtype=np.float32)
                
                # Fill datasets
                for i, sample in enumerate(samples):
                    patches_ds[i] = sample['patch']
                    features_ds[i] = sample['features']
                    targets_ds[i] = sample['target']
                
                # Save metadata
                f.attrs['n_samples'] = n_samples
                f.attrs['patch_shape'] = patch_shape
                f.attrs['feature_dim'] = feature_dim
            
            hdf5_paths[f"{split_name}_hdf5"] = hdf5_path
        
        return hdf5_paths