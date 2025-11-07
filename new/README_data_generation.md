# SABR Volatility Surface Data Generation

This guide explains how to generate the training data needed for the SABR volatility surface modeling project.

## Overview

The data generation process creates:
1. **High-fidelity data**: Monte Carlo simulated SABR volatility surfaces
2. **Low-fidelity data**: Hagan analytical SABR volatility surfaces  
3. **Training samples**: Preprocessed patches and features for MDA-CNN training

## Quick Start

### 1. Generate Data with Default Settings

```bash
cd new
python generate_training_data.py
```

This will:
- Generate 1,000 SABR parameter sets using Latin Hypercube Sampling
- Create Monte Carlo surfaces with 100,000 paths each
- Generate corresponding Hagan analytical surfaces
- Extract training patches and features
- Save data to `data/processed/` for training

### 2. Run Training

After data generation completes:

```bash
python main_training.py --data-dir data/processed
```

## Configuration

### Using Configuration File

Edit `config/data_generation_config.yaml` to customize:

```yaml
# Key parameters
n_parameter_sets: 1000          # Number of surfaces to generate
mc_paths: 100000               # Monte Carlo paths per surface
sampling_strategy: "lhs"        # Parameter sampling method
patch_size: 9                  # CNN patch size
parallel_processing: true      # Use multiprocessing
```

### Command Line Options

Override config file settings:

```bash
# Generate more data
python generate_training_data.py --n-parameter-sets 2000

# Use different output directory
python generate_training_data.py --output-dir my_data

# Enable visualizations (for debugging)
python generate_training_data.py --create-visualizations

# Use custom config file
python generate_training_data.py --config my_config.yaml
```

## Data Generation Process

### Step 1: Parameter Sampling
- Generates SABR parameter sets within Funahashi's ranges:
  - α (alpha): 0.05-0.6
  - β (beta): 0.3-0.9  
  - ν (nu): 0.05-0.9
  - ρ (rho): -0.75-0.75
- Uses Latin Hypercube Sampling for better parameter space coverage

### Step 2: Surface Generation
- **Monte Carlo**: High-fidelity surfaces using log-Euler scheme
- **Hagan**: Low-fidelity analytical approximation
- Grid: 21 strikes × 5 maturities (following Funahashi's setup)
- Parallel processing for faster generation

### Step 3: Data Preprocessing
- Extracts local surface patches around high-fidelity points
- Creates point features (SABR parameters, strike, maturity, moneyness)
- Computes residuals: D(ξ) = σ_MC(ξ) - σ_Hagan(ξ)
- Normalizes patches and features
- Splits into train/validation/test sets (70%/15%/15%)

### Step 4: Data Storage
- Saves as both pickle files (simple) and HDF5 files (efficient)
- Stores normalization statistics for consistent preprocessing
- Creates organized directory structure

## Output Structure

```
data/
├── raw/                        # Raw generated surfaces
│   ├── data_generation.log     # Generation log
│   ├── parameter_sets.pkl      # SABR parameters
│   ├── mc_results.pkl         # Monte Carlo surfaces
│   ├── hagan_results.pkl      # Hagan surfaces
│   └── validation_results.json # Quality validation
└── processed/                  # Training-ready data
    ├── train_data.pkl         # Training samples
    ├── val_data.pkl           # Validation samples  
    ├── test_data.pkl          # Test samples
    ├── train_data.h5          # HDF5 training data
    ├── val_data.h5            # HDF5 validation data
    ├── test_data.h5           # HDF5 test data
    └── normalization_stats.json # Normalization parameters
```

## Performance Tips

### For Faster Generation
- Reduce `n_parameter_sets` for testing (e.g., 100)
- Reduce `mc_paths` for faster MC simulation (e.g., 10,000)
- Enable `parallel_processing: true`
- Set `create_visualizations: false`

### For Higher Quality
- Increase `mc_paths` for more accurate MC surfaces (e.g., 500,000)
- Increase `n_parameter_sets` for more training data (e.g., 5,000)
- Use `sampling_strategy: "lhs"` for better parameter coverage

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce `n_parameter_sets` or `mc_paths`
2. **Slow Generation**: Enable parallel processing, reduce visualization
3. **Import Errors**: Ensure you're running from the `new/` directory
4. **Missing Dependencies**: Install required packages (numpy, scipy, h5py, etc.)

### Data Quality

The system automatically validates generated data:
- Checks volatility bounds and numerical stability
- Validates ATM volatility consistency with alpha parameter
- Detects outliers in MC-Hagan residuals
- Reports overall quality score

### Debugging

Enable visualizations to inspect generated surfaces:

```bash
python generate_training_data.py --create-visualizations
```

This creates plots in `data/raw/plots/` showing:
- MC vs Hagan surface comparisons
- Residual distributions
- Parameter space coverage

## Integration with Training

The generated data is automatically compatible with:
- `main_training.py`: Main training script
- `run_experiment.py`: Complete experiment runner
- `main_evaluation.py`: Model evaluation script

Simply ensure the `--data-dir` parameter points to your processed data directory.

## Requirements Compliance

This data generation process implements:
- **Requirement 1**: Complete data generation infrastructure
- **Requirement 5**: Organized project structure and reproducibility  
- **Requirement 6**: Identical datasets for fair model comparison

The generated data enables direct comparison between MDA-CNN and Funahashi baseline models using the same high-fidelity data budget, as specified in the requirements.