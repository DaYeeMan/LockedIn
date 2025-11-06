"""
Training utilities and helper functions.

This module provides utility functions for model training,
including optimizer creation, callback setup, and training monitoring.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from training.training_config import TrainingConfig
from models.loss_functions import (
    WeightedMSELoss,
    HuberLoss,
    RelativePercentageErrorLoss,
    CombinedLoss,
    create_wing_weighted_mse,
    create_robust_huber_loss,
    create_combined_mse_rpe_loss,
    relative_percentage_error_metric,
    rmse_metric
)


def create_optimizer(config: TrainingConfig) -> keras.optimizers.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Configured optimizer
    """
    optimizer_config = config.get_optimizer_config()
    
    if config.optimizer.lower() == "adam":
        optimizer = keras.optimizers.Adam(**optimizer_config)
    elif config.optimizer.lower() == "adamw":
        optimizer = keras.optimizers.AdamW(**optimizer_config)
    elif config.optimizer.lower() == "sgd":
        optimizer = keras.optimizers.SGD(**optimizer_config)
    elif config.optimizer.lower() == "rmsprop":
        optimizer = keras.optimizers.RMSprop(**optimizer_config)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    
    # Apply gradient clipping if enabled
    if config.gradient_clipping_enabled:
        optimizer = keras.optimizers.experimental.enable_mixed_precision_graph_rewrite(
            optimizer, loss_scale="dynamic"
        ) if config.mixed_precision_enabled else optimizer
    
    return optimizer


def create_loss_function(config: TrainingConfig) -> keras.losses.Loss:
    """
    Create loss function based on configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Configured loss function
    """
    loss_config = config.get_loss_config()
    
    if config.loss_function == "mse":
        return keras.losses.MeanSquaredError()
    elif config.loss_function == "mae":
        return keras.losses.MeanAbsoluteError()
    elif config.loss_function == "weighted_mse":
        return create_wing_weighted_mse(**config.loss_params)
    elif config.loss_function == "huber":
        return create_robust_huber_loss(**config.loss_params)
    elif config.loss_function == "combined":
        return create_combined_mse_rpe_loss(**config.loss_params)
    else:
        raise ValueError(f"Unsupported loss function: {config.loss_function}")


def create_metrics() -> List[keras.metrics.Metric]:
    """
    Create standard metrics for SABR volatility surface modeling.
    
    Returns:
        List of metrics for model compilation
    """
    return [
        keras.metrics.MeanAbsoluteError(name='mae'),
        rmse_metric,
        relative_percentage_error_metric
    ]


def create_callbacks(config: TrainingConfig, 
                    model_name: str,
                    output_dir: Path) -> List[keras.callbacks.Callback]:
    """
    Create training callbacks based on configuration.
    
    Args:
        config: Training configuration
        model_name: Name of the model for file naming
        output_dir: Output directory for saving files
        
    Returns:
        List of configured callbacks
    """
    callbacks = []
    callback_configs = config.get_callback_configs()
    
    # Early stopping
    if 'early_stopping' in callback_configs:
        early_stopping = keras.callbacks.EarlyStopping(**callback_configs['early_stopping'])
        callbacks.append(early_stopping)
    
    # Model checkpointing
    if 'model_checkpoint' in callback_configs:
        checkpoint_path = output_dir / config.checkpoint_dir / f"{model_name}_best_model.h5"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            **callback_configs['model_checkpoint']
        )
        callbacks.append(model_checkpoint)
    
    # Learning rate scheduling
    if 'lr_scheduler' in callback_configs:
        lr_config = callback_configs['lr_scheduler']
        
        if lr_config['type'] == 'reduce_on_plateau':
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor=lr_config['monitor'],
                factor=lr_config['factor'],
                patience=lr_config['patience'],
                min_lr=lr_config['min_lr'],
                verbose=1
            )
        elif lr_config['type'] == 'exponential':
            lr_scheduler = keras.callbacks.ExponentialDecay(
                initial_learning_rate=config.learning_rate,
                decay_steps=lr_config.get('decay_steps', 1000),
                decay_rate=lr_config.get('decay_rate', 0.96)
            )
        else:
            # Default to ReduceLROnPlateau
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        
        callbacks.append(lr_scheduler)
    
    # TensorBoard logging
    if 'tensorboard' in callback_configs:
        tensorboard_dir = output_dir / config.log_dir / f"tensorboard_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            **callback_configs['tensorboard']
        )
        callbacks.append(tensorboard)
    
    # CSV logger
    csv_log_path = output_dir / config.log_dir / f"{model_name}_training_log.csv"
    csv_log_path.parent.mkdir(parents=True, exist_ok=True)
    csv_logger = keras.callbacks.CSVLogger(str(csv_log_path))
    callbacks.append(csv_logger)
    
    return callbacks


def setup_mixed_precision(enabled: bool = False) -> None:
    """
    Setup mixed precision training if enabled.
    
    Args:
        enabled: Whether to enable mixed precision training
    """
    if enabled:
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        logging.info("Mixed precision training enabled")
    else:
        logging.info("Mixed precision training disabled")


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducible training.
    
    Args:
        seed: Random seed value
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For deterministic behavior (may reduce performance)
    tf.config.experimental.enable_op_determinism()
    
    logging.info(f"Random seeds set to {seed}")


def compile_model(model: keras.Model, config: TrainingConfig) -> None:
    """
    Compile model with appropriate optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        config: Training configuration
    """
    optimizer = create_optimizer(config)
    loss_function = create_loss_function(config)
    metrics = create_metrics()
    
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics
    )
    
    logging.info(f"Model compiled with {config.optimizer} optimizer and {config.loss_function} loss")


def save_training_config(config: TrainingConfig, output_path: Path) -> None:
    """
    Save training configuration to JSON file.
    
    Args:
        config: Training configuration to save
        output_path: Path to save configuration
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dataclass to dictionary
    config_dict = {
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'validation_split': config.validation_split,
        'optimizer': config.optimizer,
        'optimizer_params': config.optimizer_params,
        'loss_function': config.loss_function,
        'loss_params': config.loss_params,
        'dropout_rate': config.dropout_rate,
        'l2_regularization': config.l2_regularization,
        'early_stopping_enabled': config.early_stopping_enabled,
        'early_stopping_patience': config.early_stopping_patience,
        'lr_scheduling_enabled': config.lr_scheduling_enabled,
        'lr_schedule_type': config.lr_schedule_type,
        'lr_schedule_params': config.lr_schedule_params,
        'random_seed': config.random_seed,
        'mixed_precision_enabled': config.mixed_precision_enabled,
        'gradient_clipping_enabled': config.gradient_clipping_enabled,
        'gradient_clip_norm': config.gradient_clip_norm
    }
    
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logging.info(f"Training configuration saved to {output_path}")


def load_training_config(config_path: Path) -> TrainingConfig:
    """
    Load training configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded training configuration
    """
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create TrainingConfig from dictionary
    config = TrainingConfig(**config_dict)
    
    logging.info(f"Training configuration loaded from {config_path}")
    return config


def plot_training_history(history: keras.callbacks.History, 
                         output_path: Optional[Path] = None,
                         show_plot: bool = True) -> None:
    """
    Plot training history curves.
    
    Args:
        history: Training history object
        output_path: Optional path to save plot
        show_plot: Whether to display plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss curves
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # MAE curves
    if 'mae' in history.history:
        axes[0, 1].plot(history.history['mae'], label='Training MAE')
        axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # RMSE curves
    if 'rmse_metric' in history.history:
        axes[1, 0].plot(history.history['rmse_metric'], label='Training RMSE')
        axes[1, 0].plot(history.history['val_rmse_metric'], label='Validation RMSE')
        axes[1, 0].set_title('Root Mean Squared Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # RPE curves
    if 'relative_percentage_error_metric' in history.history:
        axes[1, 1].plot(history.history['relative_percentage_error_metric'], label='Training RPE')
        axes[1, 1].plot(history.history['val_relative_percentage_error_metric'], label='Validation RPE')
        axes[1, 1].set_title('Relative Percentage Error')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('RPE (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Training history plot saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def calculate_model_size(model: keras.Model) -> Dict[str, int]:
    """
    Calculate model size statistics.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with model size information
    """
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params
    }


def log_model_summary(model: keras.Model, logger: logging.Logger) -> None:
    """
    Log detailed model summary.
    
    Args:
        model: Keras model
        logger: Logger instance
    """
    logger.info("Model Architecture Summary:")
    
    # Model size
    size_info = calculate_model_size(model)
    logger.info(f"  - Total parameters: {size_info['total_parameters']:,}")
    logger.info(f"  - Trainable parameters: {size_info['trainable_parameters']:,}")
    logger.info(f"  - Non-trainable parameters: {size_info['non_trainable_parameters']:,}")
    
    # Model structure
    logger.info("  - Model layers:")
    for i, layer in enumerate(model.layers):
        logger.info(f"    {i+1}. {layer.name} ({layer.__class__.__name__})")
        if hasattr(layer, 'output_shape'):
            logger.info(f"       Output shape: {layer.output_shape}")


def create_experiment_directory(base_dir: str, experiment_name: str) -> Path:
    """
    Create experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to created experiment directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['models', 'logs', 'checkpoints', 'plots', 'configs']
    for subdir in subdirs:
        (experiment_dir / subdir).mkdir(exist_ok=True)
    
    logging.info(f"Experiment directory created: {experiment_dir}")
    return experiment_dir