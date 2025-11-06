"""
Training module for SABR volatility surface models.

This module provides comprehensive training infrastructure including:
- Main training loop with validation and early stopping
- MSE loss function for residual learning
- Model checkpointing and best model saving
- Training progress monitoring and logging
- Configuration management for training parameters
"""

from training.trainer import ModelTrainer, TrainingProgressCallback, create_trainer
from training.training_config import (
    TrainingConfig,
    BaselineTrainingConfig,
    HyperparameterSearchConfig,
    create_funahashi_training_config,
    create_mda_cnn_training_config
)
from training.training_utils import (
    create_optimizer,
    create_loss_function,
    create_metrics,
    create_callbacks,
    setup_mixed_precision,
    set_random_seeds,
    compile_model,
    save_training_config,
    load_training_config,
    plot_training_history,
    calculate_model_size,
    log_model_summary,
    create_experiment_directory
)
from training.train_model import train_model, main

__all__ = [
    # Main trainer classes
    'ModelTrainer',
    'TrainingProgressCallback',
    'create_trainer',
    
    # Configuration classes
    'TrainingConfig',
    'BaselineTrainingConfig', 
    'HyperparameterSearchConfig',
    'create_funahashi_training_config',
    'create_mda_cnn_training_config',
    
    # Utility functions
    'create_optimizer',
    'create_loss_function',
    'create_metrics',
    'create_callbacks',
    'setup_mixed_precision',
    'set_random_seeds',
    'compile_model',
    'save_training_config',
    'load_training_config',
    'plot_training_history',
    'calculate_model_size',
    'log_model_summary',
    'create_experiment_directory',
    
    # Main training function
    'train_model',
    'main'
]