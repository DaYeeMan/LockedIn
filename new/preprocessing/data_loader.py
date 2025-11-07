"""
Data loading and preprocessing pipeline for MDA-CNN volatility surface modeling.

This module implements efficient data loading with batching, shuffling, and
HDF5-based storage for preprocessed training data. It handles the complete
pipeline from raw surface data to model-ready batches.
"""

import numpy as np
import h5py
import os
from typing import List, Dict, Any, Optional, Tuple, Union, Iterator
from dataclasses import dataclass
import warnings
from pathlib import Path
import pickle
import json
from concurrent.futures import ThreadPoolExecutor
import threading

from data_generation.sabr_params import SABRParams, GridConfig
from preprocessing.patch_extractor import PatchExtractor, ExtractedPatch, PatchConfig
from preprocessing.feature_engineer import FeatureEngineer, PointFeatures, FeatureConfig


@dataclass
class DataLoaderConfig:
    """
    Configuration for data loading and preprocessing.
    
    Attributes:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data during iteration
        validation_split: Fraction of data for validation (0.0-1.0)
        test_split: Fraction of data for testing (0.0-1.0)
        random_seed: Random seed for reproducibility
        prefetch_batches: Number of batches to prefetch
        num_workers: Number of worker threads for data loading
        cache_preprocessed: Whether to cache preprocessed data
        normalize_patches: Whether to normalize patches
        normalize_features: Whether to normalize point features
    """
    batch_size: int = 64
    shuffle: bool = True
    validation_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    prefetch_batches: int = 2
    num_workers: int = 4
    cache_preprocessed: bool = True
    normalize_patches: bool = True
    normalize_features: bool = True


@dataclass
class DataSample:
    """
    Single training sample containing patch, features, and target.
    
    Attributes:
        patch: 2D array containing LF surface patch
        point_features: 1D array of normalized point features
        target_residual: Target residual value (HF - LF)
        sample_id: Unique identifier for this sample
        sabr_params: Associated SABR parameters
        strike: Strike price
        maturity: Time to maturity
        hf_volatility: High-fidelity volatility
        lf_volatility: Low-fidelity (Hagan) volatility
    """
    patch: np.ndarray
    point_features: np.ndarray
    target_residual: float
    sample_id: str
    sabr_params: SABRParams
    strike: float
    maturity: float
    hf_volatility: float
    lf_volatility: float


@dataclass
class DataBatch:
    """
    Batch of training samples.
    
    Attributes:
        patches: Array of shape (batch_size, patch_height, patch_width)
        point_features: Array of shape (batch_size, n_features)
        target_residuals: Array of shape (batch_size,)
        sample_ids: List of sample identifiers
        batch_info: Additional batch metadata
    """
    patches: np.ndarray
    point_features: np.ndarray
    target_residuals: np.ndarray
    sample_ids: List[str]
    batch_info: Dict[str, Any]


class HDF5DataStorage:
    """
    HDF5-based storage for preprocessed training data.
    
    Handles efficient storage and retrieval of large datasets with
    proper compression and chunking for optimal I/O performance.
    """
    
    def __init__(self, filepath: str, mode: str = 'r'):
        """
        Initialize HDF5 data storage.
        
        Args:
            filepath: Path to HDF5 file
            mode: File access mode ('r', 'w', 'a')
        """
        self.filepath = filepath
        self.mode = mode
        self.file = None
        self._lock = threading.Lock()
        
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def open(self):
        """Open HDF5 file."""
        with self._lock:
            if self.file is None:
                self.file = h5py.File(self.filepath, self.mode)
                
    def close(self):
        """Close HDF5 file."""
        with self._lock:
            if self.file is not None:
                self.file.close()
                self.file = None
    
    def create_dataset_structure(self, n_samples: int, patch_shape: Tuple[int, int], 
                               n_features: int, compression: str = 'gzip'):
        """
        Create HDF5 dataset structure for storing preprocessed data.
        
        Args:
            n_samples: Total number of samples
            patch_shape: Shape of patches (height, width)
            n_features: Number of point features
            compression: Compression algorithm ('gzip', 'lzf', 'szip')
        """
        if self.mode == 'r':
            raise ValueError("Cannot create datasets in read-only mode")
            
        # Create main datasets with chunking for efficient access
        chunk_size = min(1000, n_samples)
        
        # Patches dataset
        self.file.create_dataset(
            'patches',
            shape=(n_samples, patch_shape[0], patch_shape[1]),
            dtype=np.float32,
            compression=compression,
            chunks=(chunk_size, patch_shape[0], patch_shape[1]),
            shuffle=True,
            fletcher32=True
        )
        
        # Point features dataset
        self.file.create_dataset(
            'point_features',
            shape=(n_samples, n_features),
            dtype=np.float32,
            compression=compression,
            chunks=(chunk_size, n_features),
            shuffle=True,
            fletcher32=True
        )
        
        # Target residuals dataset
        self.file.create_dataset(
            'target_residuals',
            shape=(n_samples,),
            dtype=np.float32,
            compression=compression,
            chunks=(chunk_size,),
            shuffle=True,
            fletcher32=True
        )
        
        # Metadata datasets (variable length strings)
        dt = h5py.special_dtype(vlen=str)
        self.file.create_dataset(
            'sample_ids',
            shape=(n_samples,),
            dtype=dt,
            compression=compression,
            chunks=(chunk_size,)
        )
        
        # SABR parameters (structured array)
        sabr_dtype = np.dtype([
            ('F0', np.float32),
            ('alpha', np.float32),
            ('beta', np.float32),
            ('nu', np.float32),
            ('rho', np.float32)
        ])
        
        self.file.create_dataset(
            'sabr_params',
            shape=(n_samples,),
            dtype=sabr_dtype,
            compression=compression,
            chunks=(chunk_size,)
        )
        
        # Strike and maturity data
        self.file.create_dataset(
            'strikes',
            shape=(n_samples,),
            dtype=np.float32,
            compression=compression,
            chunks=(chunk_size,)
        )
        
        self.file.create_dataset(
            'maturities',
            shape=(n_samples,),
            dtype=np.float32,
            compression=compression,
            chunks=(chunk_size,)
        )
        
        # Volatility data
        self.file.create_dataset(
            'hf_volatilities',
            shape=(n_samples,),
            dtype=np.float32,
            compression=compression,
            chunks=(chunk_size,)
        )
        
        self.file.create_dataset(
            'lf_volatilities',
            shape=(n_samples,),
            dtype=np.float32,
            compression=compression,
            chunks=(chunk_size,)
        )
        
        # Create attributes for metadata
        self.file.attrs['n_samples'] = n_samples
        self.file.attrs['patch_shape'] = patch_shape
        self.file.attrs['n_features'] = n_features
        self.file.attrs['compression'] = compression
        
    def write_samples(self, samples: List[DataSample], start_idx: int = 0):
        """
        Write samples to HDF5 datasets.
        
        Args:
            samples: List of DataSample objects
            start_idx: Starting index for writing
        """
        if self.mode == 'r':
            raise ValueError("Cannot write in read-only mode")
            
        end_idx = start_idx + len(samples)
        
        # Prepare arrays
        patches = np.array([s.patch for s in samples], dtype=np.float32)
        point_features = np.array([s.point_features for s in samples], dtype=np.float32)
        target_residuals = np.array([s.target_residual for s in samples], dtype=np.float32)
        sample_ids = [s.sample_id for s in samples]
        
        # SABR parameters
        sabr_data = np.array([
            (s.sabr_params.F0, s.sabr_params.alpha, s.sabr_params.beta, 
             s.sabr_params.nu, s.sabr_params.rho) for s in samples
        ], dtype=self.file['sabr_params'].dtype)
        
        strikes = np.array([s.strike for s in samples], dtype=np.float32)
        maturities = np.array([s.maturity for s in samples], dtype=np.float32)
        hf_volatilities = np.array([s.hf_volatility for s in samples], dtype=np.float32)
        lf_volatilities = np.array([s.lf_volatility for s in samples], dtype=np.float32)
        
        # Write to datasets
        self.file['patches'][start_idx:end_idx] = patches
        self.file['point_features'][start_idx:end_idx] = point_features
        self.file['target_residuals'][start_idx:end_idx] = target_residuals
        self.file['sample_ids'][start_idx:end_idx] = sample_ids
        self.file['sabr_params'][start_idx:end_idx] = sabr_data
        self.file['strikes'][start_idx:end_idx] = strikes
        self.file['maturities'][start_idx:end_idx] = maturities
        self.file['hf_volatilities'][start_idx:end_idx] = hf_volatilities
        self.file['lf_volatilities'][start_idx:end_idx] = lf_volatilities
        
        # Flush to ensure data is written
        self.file.flush()
    
    def read_samples(self, indices: Union[List[int], np.ndarray, slice]) -> List[DataSample]:
        """
        Read samples from HDF5 datasets.
        
        Args:
            indices: Indices of samples to read
            
        Returns:
            List of DataSample objects
        """
        if isinstance(indices, slice):
            indices = list(range(*indices.indices(len(self.file['patches']))))
        elif not isinstance(indices, (list, np.ndarray)):
            indices = [indices]
            
        samples = []
        
        for i in indices:
            # Read data for this sample
            patch = self.file['patches'][i]
            point_features = self.file['point_features'][i]
            target_residual = float(self.file['target_residuals'][i])
            sample_id = self.file['sample_ids'][i]
            
            # Handle string decoding for sample_id
            if isinstance(sample_id, bytes):
                sample_id = sample_id.decode('utf-8')
            
            # Reconstruct SABR parameters
            sabr_data = self.file['sabr_params'][i]
            sabr_params = SABRParams(
                F0=float(sabr_data['F0']),
                alpha=float(sabr_data['alpha']),
                beta=float(sabr_data['beta']),
                nu=float(sabr_data['nu']),
                rho=float(sabr_data['rho'])
            )
            
            strike = float(self.file['strikes'][i])
            maturity = float(self.file['maturities'][i])
            hf_volatility = float(self.file['hf_volatilities'][i])
            lf_volatility = float(self.file['lf_volatilities'][i])
            
            sample = DataSample(
                patch=patch,
                point_features=point_features,
                target_residual=target_residual,
                sample_id=sample_id,
                sabr_params=sabr_params,
                strike=strike,
                maturity=maturity,
                hf_volatility=hf_volatility,
                lf_volatility=lf_volatility
            )
            
            samples.append(sample)
        
        return samples
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about stored dataset.
        
        Returns:
            Dictionary with dataset information
        """
        if 'patches' not in self.file:
            return {'error': 'No dataset found in file'}
            
        info = {
            'n_samples': len(self.file['patches']),
            'patch_shape': self.file['patches'].shape[1:],
            'n_features': self.file['point_features'].shape[1],
            'compression': self.file.attrs.get('compression', 'unknown'),
            'file_size_mb': os.path.getsize(self.filepath) / (1024 * 1024),
            'datasets': list(self.file.keys())
        }
        
        return info


class DataSplitter:
    """
    Handle train/validation/test splits with proper indexing and stratification.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize data splitter.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(random_seed)
        
    def create_splits(self, n_samples: int, validation_split: float = 0.15, 
                     test_split: float = 0.15) -> Dict[str, np.ndarray]:
        """
        Create train/validation/test splits.
        
        Args:
            n_samples: Total number of samples
            validation_split: Fraction for validation
            test_split: Fraction for testing
            
        Returns:
            Dictionary with 'train', 'validation', 'test' indices
        """
        if validation_split + test_split >= 1.0:
            raise ValueError("validation_split + test_split must be < 1.0")
            
        # Create shuffled indices
        indices = np.arange(n_samples)
        self.rng.shuffle(indices)
        
        # Calculate split sizes
        n_test = int(n_samples * test_split)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_test - n_val
        
        # Create splits
        splits = {
            'test': indices[:n_test],
            'validation': indices[n_test:n_test + n_val],
            'train': indices[n_test + n_val:]
        }
        
        return splits
    
    def stratified_split(self, sabr_params_list: List[SABRParams], 
                        validation_split: float = 0.15,
                        test_split: float = 0.15,
                        n_strata: int = 5) -> Dict[str, np.ndarray]:
        """
        Create stratified splits based on SABR parameter ranges.
        
        Args:
            sabr_params_list: List of SABR parameters for stratification
            validation_split: Fraction for validation
            test_split: Fraction for testing
            n_strata: Number of strata per parameter
            
        Returns:
            Dictionary with stratified split indices
        """
        n_samples = len(sabr_params_list)
        
        # Create stratification based on alpha and nu (most important parameters)
        alphas = np.array([p.alpha for p in sabr_params_list])
        nus = np.array([p.nu for p in sabr_params_list])
        
        # Create strata based on quantiles
        alpha_bins = np.quantile(alphas, np.linspace(0, 1, n_strata + 1))
        nu_bins = np.quantile(nus, np.linspace(0, 1, n_strata + 1))
        
        # Assign samples to strata
        alpha_strata = np.digitize(alphas, alpha_bins) - 1
        nu_strata = np.digitize(nus, nu_bins) - 1
        
        # Combine strata (2D stratification)
        combined_strata = alpha_strata * n_strata + nu_strata
        
        # Create splits within each stratum
        train_indices = []
        val_indices = []
        test_indices = []
        
        for stratum in np.unique(combined_strata):
            stratum_indices = np.where(combined_strata == stratum)[0]
            
            if len(stratum_indices) < 3:
                # Too few samples in stratum, assign to train
                train_indices.extend(stratum_indices)
                continue
                
            # Shuffle stratum indices
            self.rng.shuffle(stratum_indices)
            
            # Calculate split sizes for this stratum
            n_stratum = len(stratum_indices)
            n_test_stratum = max(1, int(n_stratum * test_split))
            n_val_stratum = max(1, int(n_stratum * validation_split))
            
            # Ensure we don't exceed stratum size
            if n_test_stratum + n_val_stratum >= n_stratum:
                n_test_stratum = max(1, n_stratum // 3)
                n_val_stratum = max(1, n_stratum // 3)
            
            # Create splits for this stratum
            test_indices.extend(stratum_indices[:n_test_stratum])
            val_indices.extend(stratum_indices[n_test_stratum:n_test_stratum + n_val_stratum])
            train_indices.extend(stratum_indices[n_test_stratum + n_val_stratum:])
        
        splits = {
            'train': np.array(train_indices),
            'validation': np.array(val_indices),
            'test': np.array(test_indices)
        }
        
        return splits
    
    def save_splits(self, splits: Dict[str, np.ndarray], filepath: str):
        """
        Save split indices to file.
        
        Args:
            splits: Dictionary with split indices
            filepath: Path to save splits
        """
        np.savez(filepath, **splits)
        
    def load_splits(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        Load split indices from file.
        
        Args:
            filepath: Path to load splits from
            
        Returns:
            Dictionary with split indices
        """
        data = np.load(filepath)
        return {key: data[key] for key in data.files}


class DataPreprocessor:
    """
    Main data preprocessing pipeline that coordinates patch extraction,
    feature engineering, and data storage.
    """
    
    def __init__(self, patch_config: PatchConfig = None, 
                 feature_config: FeatureConfig = None):
        """
        Initialize data preprocessor.
        
        Args:
            patch_config: Configuration for patch extraction
            feature_config: Configuration for feature engineering
        """
        self.patch_extractor = PatchExtractor(patch_config)
        self.feature_engineer = FeatureEngineer(feature_config)
        self.is_fitted = False
        
    def preprocess_surface_data(self, hf_surfaces: List[np.ndarray],
                              lf_surfaces: List[np.ndarray],
                              sabr_params_list: List[SABRParams],
                              strikes_grids: List[np.ndarray],
                              maturities_grids: List[np.ndarray],
                              hf_points_list: List[List[Tuple[float, float, float]]]) -> List[DataSample]:
        """
        Preprocess raw surface data into training samples.
        
        Args:
            hf_surfaces: List of high-fidelity surfaces
            lf_surfaces: List of low-fidelity surfaces  
            sabr_params_list: List of SABR parameters
            strikes_grids: List of strike grids
            maturities_grids: List of maturity grids
            hf_points_list: List of HF points (strike, maturity, volatility) for each surface
            
        Returns:
            List of preprocessed DataSample objects
        """
        all_samples = []
        
        # First pass: collect all point features for normalization fitting
        all_point_features = []
        
        for i, (hf_surface, lf_surface, sabr_params, strikes_grid, maturities_grid, hf_points) in enumerate(
            zip(hf_surfaces, lf_surfaces, sabr_params_list, strikes_grids, maturities_grids, hf_points_list)
        ):
            # Convert HF points to proper format
            from preprocessing.patch_extractor import HFPoint
            hf_point_objects = [
                HFPoint(strike=strike, maturity=maturity, volatility=volatility)
                for strike, maturity, volatility in hf_points
            ]
            
            # Create point features for this surface
            for hf_point in hf_point_objects:
                # Get LF volatility at this point (interpolate if needed)
                lf_vol = self._interpolate_lf_volatility(
                    lf_surface, strikes_grid, maturities_grid, 
                    hf_point.strike, hf_point.maturity
                )
                
                point_features = self.feature_engineer.create_point_features(
                    sabr_params, hf_point.strike, hf_point.maturity, lf_vol
                )
                all_point_features.append(point_features)
        
        # Fit normalization on all point features
        if not self.is_fitted:
            self.feature_engineer.fit_normalization(all_point_features)
            self.is_fitted = True
        
        # Second pass: create actual samples with normalized features
        sample_id_counter = 0
        
        for i, (hf_surface, lf_surface, sabr_params, strikes_grid, maturities_grid, hf_points) in enumerate(
            zip(hf_surfaces, lf_surfaces, sabr_params_list, strikes_grids, maturities_grids, hf_points_list)
        ):
            # Convert HF points to proper format
            hf_point_objects = [
                HFPoint(strike=strike, maturity=maturity, volatility=volatility)
                for strike, maturity, volatility in hf_points
            ]
            
            # Extract patches for this surface
            extracted_patches = self.patch_extractor.extract_patches_for_hf_points(
                lf_surface, hf_point_objects, strikes_grid, maturities_grid
            )
            
            # Create samples from patches
            for patch_result in extracted_patches:
                hf_point = patch_result.hf_point
                
                # Get LF volatility at this point
                lf_vol = self._interpolate_lf_volatility(
                    lf_surface, strikes_grid, maturities_grid,
                    hf_point.strike, hf_point.maturity
                )
                
                # Create and normalize point features
                point_features = self.feature_engineer.create_point_features(
                    sabr_params, hf_point.strike, hf_point.maturity, lf_vol
                )
                
                # Calculate residual
                residual = hf_point.volatility - lf_vol
                
                # Create sample
                sample = DataSample(
                    patch=patch_result.patch.astype(np.float32),
                    point_features=point_features.normalized_features.astype(np.float32),
                    target_residual=float(residual),
                    sample_id=f"surface_{i:04d}_point_{sample_id_counter:06d}",
                    sabr_params=sabr_params,
                    strike=hf_point.strike,
                    maturity=hf_point.maturity,
                    hf_volatility=hf_point.volatility,
                    lf_volatility=lf_vol
                )
                
                all_samples.append(sample)
                sample_id_counter += 1
        
        return all_samples
    
    def _interpolate_lf_volatility(self, lf_surface: np.ndarray,
                                 strikes_grid: np.ndarray,
                                 maturities_grid: np.ndarray,
                                 target_strike: float,
                                 target_maturity: float) -> float:
        """
        Interpolate LF volatility at target strike/maturity.
        
        Args:
            lf_surface: LF surface array
            strikes_grid: Strike grid
            maturities_grid: Maturity grid
            target_strike: Target strike price
            target_maturity: Target maturity
            
        Returns:
            Interpolated LF volatility
        """
        from scipy.interpolate import RegularGridInterpolator
        
        # Create interpolator
        interpolator = RegularGridInterpolator(
            (maturities_grid, strikes_grid[0]),  # Assuming rectangular grid
            lf_surface,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        
        # Interpolate
        result = interpolator((target_maturity, target_strike))
        
        if np.isnan(result):
            # Fallback to nearest neighbor
            mat_idx = np.argmin(np.abs(maturities_grid - target_maturity))
            strike_idx = np.argmin(np.abs(strikes_grid[mat_idx] - target_strike))
            result = lf_surface[mat_idx, strike_idx]
        
        return float(result)


class SABRDataLoader:
    """
    Main data loader class that provides efficient batching and iteration
    over preprocessed SABR volatility surface data.
    """
    
    def __init__(self, config: DataLoaderConfig = None):
        """
        Initialize SABR data loader.
        
        Args:
            config: Data loader configuration
        """
        self.config = config or DataLoaderConfig()
        self.storage = None
        self.splits = None
        self.current_split = 'train'
        self.current_indices = None
        self.epoch_indices = None
        self.batch_queue = []
        self.prefetch_executor = None
        
    def load_from_hdf5(self, filepath: str, splits_filepath: Optional[str] = None):
        """
        Load data from HDF5 file.
        
        Args:
            filepath: Path to HDF5 data file
            splits_filepath: Path to splits file (optional)
        """
        self.storage = HDF5DataStorage(filepath, mode='r')
        self.storage.open()
        
        if splits_filepath and os.path.exists(splits_filepath):
            splitter = DataSplitter(self.config.random_seed)
            self.splits = splitter.load_splits(splits_filepath)
        else:
            # Create default splits
            info = self.storage.get_dataset_info()
            splitter = DataSplitter(self.config.random_seed)
            self.splits = splitter.create_splits(
                info['n_samples'], 
                self.config.validation_split,
                self.config.test_split
            )
        
        self.set_split('train')
        
    def set_split(self, split_name: str):
        """
        Set current data split.
        
        Args:
            split_name: Name of split ('train', 'validation', 'test')
        """
        if self.splits is None:
            raise ValueError("No splits loaded")
            
        if split_name not in self.splits:
            raise ValueError(f"Split '{split_name}' not found")
            
        self.current_split = split_name
        self.current_indices = self.splits[split_name]
        self._prepare_epoch()
        
    def _prepare_epoch(self):
        """Prepare indices for new epoch."""
        self.epoch_indices = self.current_indices.copy()
        
        if self.config.shuffle and self.current_split == 'train':
            rng = np.random.RandomState(self.config.random_seed)
            rng.shuffle(self.epoch_indices)
    
    def __len__(self) -> int:
        """Return number of batches in current split."""
        if self.current_indices is None:
            return 0
        return (len(self.current_indices) + self.config.batch_size - 1) // self.config.batch_size
    
    def __iter__(self) -> Iterator[DataBatch]:
        """Iterate over batches."""
        self._prepare_epoch()
        
        for i in range(0, len(self.epoch_indices), self.config.batch_size):
            batch_indices = self.epoch_indices[i:i + self.config.batch_size]
            yield self._create_batch(batch_indices)
    
    def _create_batch(self, indices: np.ndarray) -> DataBatch:
        """
        Create a batch from given indices.
        
        Args:
            indices: Array of sample indices
            
        Returns:
            DataBatch object
        """
        samples = self.storage.read_samples(indices)
        
        # Stack arrays
        patches = np.stack([s.patch for s in samples])
        point_features = np.stack([s.point_features for s in samples])
        target_residuals = np.array([s.target_residual for s in samples])
        sample_ids = [s.sample_id for s in samples]
        
        # Create batch info
        batch_info = {
            'split': self.current_split,
            'batch_size': len(samples),
            'indices': indices.tolist()
        }
        
        return DataBatch(
            patches=patches,
            point_features=point_features,
            target_residuals=target_residuals,
            sample_ids=sample_ids,
            batch_info=batch_info
        )
    
    def get_sample(self, index: int) -> DataSample:
        """
        Get single sample by index.
        
        Args:
            index: Sample index
            
        Returns:
            DataSample object
        """
        return self.storage.read_samples([index])[0]
    
    def get_split_info(self) -> Dict[str, Any]:
        """
        Get information about current splits.
        
        Returns:
            Dictionary with split information
        """
        if self.splits is None:
            return {'error': 'No splits loaded'}
            
        info = {}
        for split_name, indices in self.splits.items():
            info[split_name] = {
                'n_samples': len(indices),
                'n_batches': (len(indices) + self.config.batch_size - 1) // self.config.batch_size
            }
        
        return info
    
    def close(self):
        """Close data loader and cleanup resources."""
        if self.storage:
            self.storage.close()
        if self.prefetch_executor:
            self.prefetch_executor.shutdown(wait=True)


def create_preprocessed_dataset(raw_data_dir: str, output_filepath: str,
                              patch_config: PatchConfig = None,
                              feature_config: FeatureConfig = None,
                              loader_config: DataLoaderConfig = None) -> Dict[str, Any]:
    """
    Create preprocessed HDF5 dataset from raw surface data.
    
    Args:
        raw_data_dir: Directory containing raw surface data
        output_filepath: Path for output HDF5 file
        patch_config: Patch extraction configuration
        feature_config: Feature engineering configuration
        loader_config: Data loader configuration
        
    Returns:
        Dictionary with preprocessing results and statistics
    """
    # Load raw data (implementation depends on raw data format)
    # This is a placeholder - actual implementation would load from
    # the data generation pipeline output
    
    preprocessor = DataPreprocessor(patch_config, feature_config)
    
    # Example preprocessing (would be replaced with actual data loading)
    # samples = preprocessor.preprocess_surface_data(...)
    
    # For now, return placeholder
    return {
        'status': 'placeholder',
        'message': 'Actual implementation depends on raw data format from data generation pipeline'
    }


class SABRDataset:
    """
    PyTorch-style dataset for SABR volatility surface data.
    
    Loads preprocessed data and provides samples for training.
    """
    
    def __init__(self, data_dir: str, split: str = 'train', patch_size: int = 9, 
                 normalize: bool = True):
        """
        Initialize SABR dataset.
        
        Args:
            data_dir: Directory containing preprocessed data
            split: Data split ('train', 'val', 'test')
            patch_size: Size of patches (for compatibility)
            normalize: Whether data is normalized (for compatibility)
        """
        self.data_dir = data_dir
        self.split = split
        self.patch_size = patch_size
        self.normalize = normalize
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load data from files."""
        import pickle
        
        # Try to load pickle file first
        pickle_path = os.path.join(self.data_dir, f"{self.split}_data.pkl")
        
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                self.samples = pickle.load(f)
        else:
            # Try HDF5 file
            hdf5_path = os.path.join(self.data_dir, f"{self.split}_data.h5")
            if os.path.exists(hdf5_path):
                self._load_from_hdf5(hdf5_path)
            else:
                raise FileNotFoundError(f"No data files found for split '{self.split}' in {self.data_dir}")
    
    def _load_from_hdf5(self, hdf5_path: str):
        """Load data from HDF5 file."""
        self.samples = []
        
        with h5py.File(hdf5_path, 'r') as f:
            n_samples = f.attrs['n_samples']
            
            for i in range(n_samples):
                sample = {
                    'patch': f['patches'][i],
                    'features': f['features'][i],
                    'target': f['targets'][i]
                }
                self.samples.append(sample)
    
    def __len__(self):
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Convert to numpy arrays and handle NaN values
        patch = np.array(sample['patch'], dtype=np.float32)
        features = np.array(sample['features'], dtype=np.float32)
        target = float(sample['target'])
        
        # Replace NaN with zeros in patches
        patch = np.nan_to_num(patch, nan=0.0)
        
        return patch, features, target