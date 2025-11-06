"""
Training-specific configuration classes and utilities.

This module provides detailed configuration options for training,
including hyperparameter settings, callback configurations, and
training strategy options.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Comprehensive training configuration for SABR volatility surface models.
    
    This class encapsulates all training-related parameters including
    optimization settings, regularization, callbacks, and logging options.
    """
    
    # Basic training parameters
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 3e-4
    validation_split: float = 0.15
    
    # Optimizer settings
    optimizer: str = "adam"
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-7
    })
    
    # Loss function settings
    loss_function: str = "mse"  # "mse", "weighted_mse", "huber", "combined"
    loss_params: Dict[str, Any] = field(default_factory=dict)
    
    # Regularization
    dropout_rate: float = 0.2
    l2_regularization: float = 1e-4
    
    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 20
    early_stopping_monitor: str = "val_loss"
    early_stopping_min_delta: float = 1e-6
    restore_best_weights: bool = True
    
    # Learning rate scheduling
    lr_scheduling_enabled: bool = True
    lr_schedule_type: str = "reduce_on_plateau"  # "reduce_on_plateau", "exponential", "cosine"
    lr_schedule_params: Dict[str, Any] = field(default_factory=lambda: {
        'factor': 0.5,
        'patience': 10,
        'min_lr': 1e-7
    })
    
    # Model checkpointing
    checkpointing_enabled: bool = True
    checkpoint_monitor: str = "val_loss"
    checkpoint_mode: str = "min"
    save_best_only: bool = True
    save_weights_only: bool = False
    
    # Logging and monitoring
    tensorboard_enabled: bool = True
    tensorboard_histogram_freq: int = 1
    tensorboard_write_graph: bool = True
    tensorboard_write_images: bool = True
    
    verbose_training: bool = True
    log_frequency: int = 1  # Log every N epochs
    
    # Advanced training options
    mixed_precision_enabled: bool = False
    gradient_clipping_enabled: bool = False
    gradient_clip_norm: float = 1.0
    
    # Data augmentation (if applicable)
    data_augmentation_enabled: bool = False
    augmentation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Reproducibility
    random_seed: int = 42
    deterministic_training: bool = True
    
    # Output directories
    output_dir: str = "results"
    model_save_dir: str = "models"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate parameters
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.validation_split < 1:
            raise ValueError("validation_split must be in [0, 1)")
        
        # Setup default loss parameters based on loss function
        if self.loss_function == "weighted_mse" and not self.loss_params:
            self.loss_params = {
                'wing_weight': 2.0,
                'atm_threshold': 0.1
            }
        elif self.loss_function == "huber" and not self.loss_params:
            self.loss_params = {
                'delta': 1.0
            }
        elif self.loss_function == "combined" and not self.loss_params:
            self.loss_params = {
                'mse_weight': 0.7,
                'rpe_weight': 0.3
            }
    
    def get_optimizer_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        config = {
            'learning_rate': self.learning_rate,
            **self.optimizer_params
        }
        return config
    
    def get_loss_config(self) -> Dict[str, Any]:
        """Get loss function configuration."""
        return {
            'loss_function': self.loss_function,
            **self.loss_params
        }
    
    def get_callback_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get callback configurations."""
        configs = {}
        
        if self.early_stopping_enabled:
            configs['early_stopping'] = {
                'monitor': self.early_stopping_monitor,
                'patience': self.early_stopping_patience,
                'min_delta': self.early_stopping_min_delta,
                'restore_best_weights': self.restore_best_weights,
                'verbose': 1
            }
        
        if self.lr_scheduling_enabled:
            configs['lr_scheduler'] = {
                'type': self.lr_schedule_type,
                'monitor': 'val_loss',
                **self.lr_schedule_params
            }
        
        if self.checkpointing_enabled:
            configs['model_checkpoint'] = {
                'monitor': self.checkpoint_monitor,
                'mode': self.checkpoint_mode,
                'save_best_only': self.save_best_only,
                'save_weights_only': self.save_weights_only,
                'verbose': 1
            }
        
        if self.tensorboard_enabled:
            configs['tensorboard'] = {
                'histogram_freq': self.tensorboard_histogram_freq,
                'write_graph': self.tensorboard_write_graph,
                'write_images': self.tensorboard_write_images
            }
        
        return configs
    
    def create_output_directories(self) -> None:
        """Create necessary output directories."""
        dirs = [
            self.output_dir,
            f"{self.output_dir}/{self.model_save_dir}",
            f"{self.output_dir}/{self.log_dir}",
            f"{self.output_dir}/{self.checkpoint_dir}"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class BaselineTrainingConfig:
    """
    Training configuration specifically for Funahashi baseline model.
    
    This ensures exact replication of Funahashi's training setup
    for fair comparison with MDA-CNN.
    """
    
    # Funahashi's exact configuration
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 3e-4
    
    # Network architecture (fixed for Funahashi baseline)
    hidden_layers: int = 5
    neurons_per_layer: int = 32
    activation: str = "relu"
    
    # Training strategy
    residual_learning: bool = True  # Learn D(ξ) = σ_MC - σ_Hagan
    
    # Regularization (minimal for baseline)
    dropout_rate: float = 0.0  # Funahashi doesn't mention dropout
    l2_regularization: float = 0.0
    
    # Early stopping (conservative for baseline)
    early_stopping_patience: int = 30
    
    # Reproducibility
    random_seed: int = 42
    
    def to_training_config(self) -> TrainingConfig:
        """Convert to general TrainingConfig for compatibility."""
        return TrainingConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            dropout_rate=self.dropout_rate,
            l2_regularization=self.l2_regularization,
            early_stopping_patience=self.early_stopping_patience,
            random_seed=self.random_seed,
            # Disable advanced features for baseline
            mixed_precision_enabled=False,
            gradient_clipping_enabled=False,
            data_augmentation_enabled=False
        )


@dataclass
class HyperparameterSearchConfig:
    """
    Configuration for hyperparameter optimization.
    
    Defines search spaces and strategies for automated
    hyperparameter tuning.
    """
    
    # Search strategy
    search_strategy: str = "random"  # "random", "grid", "bayesian"
    max_trials: int = 50
    max_epochs_per_trial: int = 100
    
    # Search spaces (define ranges for hyperparameters)
    learning_rate_range: tuple = (1e-5, 1e-2)
    batch_size_options: List[int] = field(default_factory=lambda: [32, 64, 128])
    dropout_rate_range: tuple = (0.0, 0.5)
    
    # Architecture search (for MDA-CNN)
    cnn_filters_options: List[List[int]] = field(default_factory=lambda: [
        [32, 64, 128],
        [16, 32, 64],
        [64, 128, 256]
    ])
    mlp_hidden_dims_options: List[List[int]] = field(default_factory=lambda: [
        [64, 64],
        [128, 64],
        [64, 32]
    ])
    
    # Objective metric
    objective_metric: str = "val_loss"
    objective_direction: str = "minimize"
    
    # Early stopping for trials
    trial_early_stopping_patience: int = 10
    
    # Output
    results_dir: str = "hyperparameter_search"


def create_funahashi_training_config() -> TrainingConfig:
    """
    Create training configuration that exactly matches Funahashi's setup.
    
    Returns:
        TrainingConfig configured for Funahashi baseline replication
    """
    baseline_config = BaselineTrainingConfig()
    return baseline_config.to_training_config()


def create_mda_cnn_training_config() -> TrainingConfig:
    """
    Create optimized training configuration for MDA-CNN.
    
    Returns:
        TrainingConfig optimized for MDA-CNN architecture
    """
    return TrainingConfig(
        epochs=200,
        batch_size=64,
        learning_rate=3e-4,
        
        # Enhanced regularization for MDA-CNN
        dropout_rate=0.2,
        l2_regularization=1e-4,
        
        # Advanced training features
        early_stopping_patience=20,
        lr_scheduling_enabled=True,
        gradient_clipping_enabled=True,
        gradient_clip_norm=1.0,
        
        # Enhanced monitoring
        tensorboard_enabled=True,
        verbose_training=True,
        
        # Loss function optimized for residual learning
        loss_function="mse",  # Start with MSE as per requirements
        
        random_seed=42
    )