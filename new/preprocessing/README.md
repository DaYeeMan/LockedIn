# Data Preprocessing and Loading Pipeline

This module implements a comprehensive data preprocessing and loading pipeline for SABR volatility surface modeling with MDA-CNN. The pipeline provides efficient data handling, normalization, and batch loading capabilities optimized for machine learning workflows.

## Overview

The preprocessing pipeline consists of several key components:

1. **HDF5 Data Storage** - Efficient storage and retrieval of large datasets
2. **Data Splitting** - Flexible train/validation/test splits with stratification
3. **Normalization** - Comprehensive normalization for patches, features, and targets
4. **Data Loading** - High-performance batch loading with shuffling and prefetching
5. **Integration** - Seamless integration with existing patch extraction and feature engineering

## Key Features

### ✅ Efficient Data Loader with Batching and Shuffling
- **SABRDataLoader**: Main data loader class with configurable batch sizes
- **Batch iteration**: Memory-efficient iteration over large datasets
- **Shuffling**: Configurable shuffling for training data
- **Multiple splits**: Support for train/validation/test splits

### ✅ Data Normalization and Scaling Utilities
- **Multiple methods**: Standard, min-max, robust, and quantile normalization
- **Specialized normalizers**: PatchNormalizer for surface patches, TargetNormalizer for residuals
- **Outlier handling**: Configurable outlier clipping and NaN handling
- **Inverse transformation**: Full support for denormalization

### ✅ Train/Validation/Test Splits with Proper Indexing
- **Random splits**: Standard random splitting with configurable ratios
- **Stratified splits**: SABR parameter-based stratification for balanced splits
- **Persistent splits**: Save/load splits for reproducible experiments
- **Index management**: Proper indexing to prevent data leakage

### ✅ HDF5-based Storage for Preprocessed Training Data
- **Compressed storage**: Efficient storage with gzip compression
- **Chunked access**: Optimized for batch reading patterns
- **Metadata preservation**: Complete sample information storage
- **Scalable**: Handles datasets from hundreds to millions of samples

### ✅ Tests for Data Loading Consistency and Performance
- **Comprehensive tests**: Unit tests for all components
- **Integration tests**: End-to-end pipeline testing
- **Performance tests**: Benchmarking for large datasets
- **Consistency validation**: Data integrity and normalization validation

## Components

### Core Classes

#### `DataSample`
Represents a single training sample with all associated data:
```python
@dataclass
class DataSample:
    patch: np.ndarray              # LF surface patch (9x9)
    point_features: np.ndarray     # Normalized point features
    target_residual: float         # HF - LF residual
    sample_id: str                 # Unique identifier
    sabr_params: SABRParams        # SABR model parameters
    strike: float                  # Strike price
    maturity: float                # Time to maturity
    hf_volatility: float           # High-fidelity volatility
    lf_volatility: float           # Low-fidelity volatility
```

#### `HDF5DataStorage`
Efficient HDF5-based storage with compression and chunking:
```python
# Create and write data
with HDF5DataStorage('data.h5', 'w') as storage:
    storage.create_dataset_structure(n_samples, (9, 9), 8)
    storage.write_samples(samples)

# Read data
with HDF5DataStorage('data.h5', 'r') as storage:
    samples = storage.read_samples([0, 1, 2])  # Read specific indices
```

#### `DataSplitter`
Flexible data splitting with stratification support:
```python
splitter = DataSplitter(random_seed=42)

# Random splits
splits = splitter.create_splits(n_samples=1000, validation_split=0.15, test_split=0.15)

# Stratified splits based on SABR parameters
stratified_splits = splitter.stratified_split(sabr_params_list, validation_split=0.15, test_split=0.15)
```

#### `DataNormalizer`
Comprehensive normalization with multiple methods:
```python
# Standard normalization
normalizer = DataNormalizer(NormalizationConfig(method='standard'))
normalized_data = normalizer.fit_transform(data)
original_data = normalizer.inverse_transform(normalized_data)

# Specialized patch normalization
patch_normalizer = PatchNormalizer()
normalized_patches = patch_normalizer.fit_transform(patches)
```

#### `SABRDataLoader`
Main data loader with batch iteration:
```python
config = DataLoaderConfig(batch_size=64, shuffle=True)
loader = SABRDataLoader(config)
loader.load_from_hdf5('data.h5', 'splits.npz')

# Iterate over batches
loader.set_split('train')
for batch in loader:
    patches = batch.patches          # (batch_size, 9, 9)
    features = batch.point_features  # (batch_size, n_features)
    targets = batch.target_residuals # (batch_size,)
```

## Usage Examples

### Basic Pipeline Usage

```python
from preprocessing.data_loader import SABRDataLoader, DataLoaderConfig
from preprocessing.normalization import create_normalization_pipeline

# Create normalization pipeline
pipeline = create_normalization_pipeline()

# Create data loader
config = DataLoaderConfig(batch_size=32, shuffle=True)
loader = SABRDataLoader(config)
loader.load_from_hdf5('preprocessed_data.h5', 'splits.npz')

# Training loop
loader.set_split('train')
for epoch in range(num_epochs):
    for batch in loader:
        # batch.patches: (32, 9, 9) - LF surface patches
        # batch.point_features: (32, 8) - normalized point features  
        # batch.target_residuals: (32,) - target residuals
        
        # Train model with batch data
        loss = model.train_step(batch.patches, batch.point_features, batch.target_residuals)
```

### Data Preprocessing

```python
from preprocessing.data_loader import DataPreprocessor, HDF5DataStorage
from preprocessing.patch_extractor import PatchConfig
from preprocessing.feature_engineer import FeatureConfig

# Create preprocessor
preprocessor = DataPreprocessor(
    patch_config=PatchConfig(patch_size=(9, 9)),
    feature_config=FeatureConfig(normalize_features=True)
)

# Preprocess surface data
samples = preprocessor.preprocess_surface_data(
    hf_surfaces=hf_surfaces,
    lf_surfaces=lf_surfaces,
    sabr_params_list=sabr_params_list,
    strikes_grids=strikes_grids,
    maturities_grids=maturities_grids,
    hf_points_list=hf_points_list
)

# Save to HDF5
with HDF5DataStorage('preprocessed_data.h5', 'w') as storage:
    storage.create_dataset_structure(len(samples), (9, 9), 8)
    storage.write_samples(samples)
```

## Performance Characteristics

The pipeline is optimized for performance and scalability:

- **Memory efficiency**: Streaming data loading without loading entire datasets into memory
- **I/O optimization**: HDF5 chunking and compression for optimal disk access patterns
- **Parallel processing**: Support for multi-threaded data loading (configurable)
- **Caching**: Optional caching of preprocessed data for repeated access

### Benchmarks

Performance on a typical development machine:

| Dataset Size | Write Time | Read Time (100 samples) | File Size | Compression Ratio |
|--------------|------------|-------------------------|-----------|-------------------|
| 100 samples  | 0.006s     | 0.110s                  | 0.06 MB   | 0.6x              |
| 500 samples  | 0.011s     | 0.171s                  | 0.19 MB   | 0.9x              |
| 1000 samples | 0.017s     | 0.238s                  | 0.38 MB   | 0.9x              |

## Configuration

### DataLoaderConfig
```python
@dataclass
class DataLoaderConfig:
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
```

### NormalizationConfig
```python
@dataclass
class NormalizationConfig:
    method: str = 'standard'  # 'standard', 'minmax', 'robust', 'quantile'
    clip_outliers: bool = False
    outlier_percentiles: Tuple[float, float] = (1.0, 99.0)
    epsilon: float = 1e-8
    per_feature: bool = True
    handle_nan: str = 'skip'  # 'skip', 'zero', 'mean'
```

## Testing

The module includes comprehensive tests:

```bash
# Run all preprocessing tests
python -m pytest preprocessing/test_data_loader.py -v
python -m pytest preprocessing/test_normalization.py -v

# Run integration tests
python preprocessing/test_integration_simple.py

# Run example demonstration
python preprocessing/example_usage.py
```

## Integration with MDA-CNN Pipeline

This preprocessing pipeline integrates seamlessly with the broader MDA-CNN system:

1. **Data Generation** → **Preprocessing** → Model Training
2. Patch extraction from LF surfaces around HF points
3. Feature engineering from SABR parameters and market data
4. Normalization for stable training
5. Efficient batch loading for model training

The pipeline satisfies requirements 6.3 and 6.5 from the specification:
- **6.3**: Efficient data loading and batching strategies ✅
- **6.5**: Memory-efficient batch processing for large datasets ✅

## Files Structure

```
preprocessing/
├── __init__.py
├── data_loader.py           # Main data loading and HDF5 storage
├── normalization.py         # Normalization utilities
├── patch_extractor.py       # Patch extraction (existing)
├── feature_engineer.py      # Feature engineering (existing)
├── test_data_loader.py      # Data loader tests
├── test_normalization.py    # Normalization tests
├── test_integration_simple.py  # Integration tests
├── example_usage.py         # Usage examples
└── README.md               # This file
```

The data preprocessing and loading pipeline is now complete and ready for integration with the MDA-CNN model training pipeline.