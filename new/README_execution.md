# SABR Volatility Surface Model Execution Scripts

This directory contains the main execution scripts for training and evaluating SABR volatility surface models, specifically comparing the MDA-CNN approach with the Funahashi baseline model.

## Scripts Overview

### 1. `main_training.py`
Main training script that trains both MDA-CNN and Funahashi baseline models with identical data and configuration.

**Usage:**
```bash
python main_training.py --config config/training_config.yaml --data-dir data/processed
```

**Key Features:**
- Trains both models with same data for fair comparison
- Automatic experiment directory creation with timestamps
- Comprehensive logging and progress tracking
- Model checkpointing and best model saving
- Initial evaluation on test set

### 2. `main_evaluation.py`
Comprehensive evaluation script that loads trained models and generates detailed comparison analysis.

**Usage:**
```bash
python main_evaluation.py --experiment-dir results/sabr_comparison_20241106_143022
```

**Key Features:**
- Loads trained models from experiment directory
- Generates all comparison metrics specified in requirements
- Creates comprehensive visualization plots
- Produces detailed analysis reports
- Supports both quick and detailed analysis modes

### 3. `run_experiment.py`
Convenience script that orchestrates complete experiments from training through evaluation.

**Usage:**
```bash
# Run complete experiment
python run_experiment.py --config config/training_config.yaml

# Run with custom settings
python run_experiment.py --experiment-name my_experiment --device cuda --detailed-analysis

# Run only evaluation on existing experiment
python run_experiment.py --skip-training --experiment-name existing_experiment
```

## Configuration

All scripts use the same configuration file format. Key parameters:

- `batch_size`: Training batch size
- `learning_rate`: Learning rate for optimization
- `num_epochs`: Maximum training epochs
- `patch_size`: Size of surface patches for MDA-CNN
- `point_features_dim`: Dimension of point features
- Model architecture parameters (CNN channels, MLP dimensions, etc.)

## Output Structure

Each experiment creates a structured output directory:

```
results/sabr_comparison_YYYYMMDD_HHMMSS/
├── config.yaml                 # Experiment configuration
├── training.log                # Training process log
├── mda_cnn/                    # MDA-CNN model files
│   ├── best_model.pth         # Best model checkpoint
│   ├── training_history.json  # Training metrics history
│   └── checkpoints/           # Training checkpoints
├── funahashi/                  # Funahashi model files
│   ├── best_model.pth         # Best model checkpoint
│   ├── training_history.json  # Training metrics history
│   └── checkpoints/           # Training checkpoints
├── evaluation/                 # Evaluation results
│   ├── evaluation.log         # Evaluation process log
│   ├── metrics_comparison.txt # Detailed metrics comparison
│   ├── detailed_results.npz   # Raw evaluation data
│   ├── final_comparison_report.md # Comprehensive report
│   └── plots/                 # All visualization plots
└── comparison_report.txt       # Initial comparison from training
```

## Requirements Compliance

These scripts implement all requirements specified in the SABR volatility surface modeling specification:

- **Requirement 5.4**: Main execution scripts for training and evaluation
- **Requirement 6.1**: Direct comparison with Funahashi baseline using identical data
- **Requirement 3.3**: Comprehensive evaluation metrics (MSE, RMSE, MAE, relative error)
- **Requirement 4.1, 4.3**: Complete visualization suite for model comparison

## Example Workflow

1. **Prepare data**: Ensure preprocessed data is available in `data/processed/`
2. **Configure experiment**: Edit `config/training_config.yaml` as needed
3. **Run experiment**: `python run_experiment.py --detailed-analysis`
4. **Review results**: Check the generated experiment directory for all outputs

## Performance Notes

- Training time depends on dataset size and model complexity
- GPU acceleration recommended for faster training
- Evaluation is typically much faster than training
- Detailed analysis mode generates additional surface reconstruction plots