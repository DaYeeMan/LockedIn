# SABR Training Infrastructure

This module provides comprehensive training infrastructure for SABR volatility surface models, including both MDA-CNN and Funahashi baseline models.

## Components Implemented

### 1. Main Training Loop (`trainer.py`)

The `ModelTrainer` class provides:
- **Validation and Early Stopping**: Configurable early stopping with patience and metric monitoring
- **MSE Loss Function**: Default MSE loss for residual learning (D(ξ) = σ_MC - σ_Hagan)
- **Model Checkpointing**: Automatic saving of best models based on validation metrics
- **Progress Monitoring**: Comprehensive logging and progress tracking

Key features:
- Reproducible training with fixed random seeds
- Automatic directory creation for organized output
- Support for both MDA-CNN and baseline models
- Comprehensive error handling and logging

### 2. Training Configuration (`training_config.py`)

Provides structured configuration management:
- `TrainingConfig`: General training parameters
- `BaselineTrainingConfig`: Funahashi-specific settings
- `HyperparameterSearchConfig`: For automated tuning

### 3. Training Utilities (`training_utils.py`)

Helper functions for:
- Optimizer creation (Adam, AdamW, SGD, RMSprop)
- Loss function setup (MSE, weighted MSE, Huber, combined)
- Callback configuration (early stopping, checkpointing, TensorBoard)
- Mixed precision training setup
- Model compilation and summary logging

### 4. Main Training Script (`train_model.py`)

Command-line interface for training:
```bash
python training/train_model.py --model_type mda_cnn --experiment_name my_experiment
```

## Usage Examples

### Basic Training

```python
from training import create_trainer, TrainingConfig
from config.config_manager import ConfigManager

# Setup configuration
config_manager = ConfigManager()
trainer = create_trainer(
    config_manager=config_manager,
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset
)

# Train model
history = trainer.train()

# Evaluate
results = trainer.evaluate(test_dataset)
```

### Advanced Configuration

```python
from training import TrainingConfig, create_mda_cnn_training_config

# Use pre-configured settings
config = create_mda_cnn_training_config()

# Or customize
config = TrainingConfig(
    epochs=200,
    batch_size=64,
    learning_rate=3e-4,
    early_stopping_patience=20,
    loss_function="mse"  # For residual learning
)
```

## Key Features

### MSE Loss for Residual Learning
- Default MSE loss function optimized for residual prediction
- Learns D(ξ) = σ_MC(ξ) - σ_Hagan(ξ) as specified in requirements
- Additional loss functions available (weighted MSE, Huber, combined)

### Model Checkpointing
- Automatic saving of best models based on validation loss
- Saves both complete models (.h5) and weights (.weights.h5)
- Organized directory structure with timestamps

### Comprehensive Logging
- Multi-level logging (DEBUG, INFO, WARNING, ERROR)
- Training progress with epoch timing and metrics
- TensorBoard integration for visual monitoring
- CSV export of training history
- Model architecture summaries

### Early Stopping
- Configurable patience and monitoring metrics
- Automatic restoration of best weights
- Prevents overfitting and saves training time

## Output Structure

```
results/
├── experiment_name_timestamp/
│   ├── models/
│   │   ├── best_model.h5
│   │   ├── final_model.h5
│   │   └── final.weights.h5
│   ├── logs/
│   │   ├── training_history.csv
│   │   ├── training_history.json
│   │   └── tensorboard/
│   ├── checkpoints/
│   │   └── best_model.h5
│   └── configs/
│       └── training_config.json
```

## Requirements Satisfied

✅ **Requirement 2.5**: Training and evaluation framework with proper loss functions
✅ **Requirement 3.3**: Model training with validation and metrics computation

The training infrastructure provides a complete, production-ready system for training SABR volatility surface models with proper validation, checkpointing, and monitoring capabilities.