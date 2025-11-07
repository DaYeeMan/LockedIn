"""
Training configuration for SABR volatility surface modeling.

This module provides the TrainingConfig class that main_training.py expects.
It integrates with the existing config system.
"""

import yaml
from dataclasses import dataclass
from typing import List, Optional, Union
from pathlib import Path

from config.config_manager import ConfigManager, ExperimentConfig, ModelConfig, DataConfig


@dataclass
class TrainingConfig:
    """
    Training configuration that matches what main_training.py expects.
    
    This class provides a unified interface for all training parameters.
    """
    # Model architecture
    patch_size: int = 9
    point_features_dim: int = 10  # Updated to match our feature extraction
    cnn_channels: List[int] = None
    cnn_kernel_sizes: List[int] = None
    mlp_hidden_dims: List[int] = None
    fusion_dim: int = 128
    dropout_rate: float = 0.2
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 3e-4
    num_epochs: int = 200
    early_stopping_patience: int = 20
    validation_split: float = 0.15
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    
    # Optimization
    weight_decay: float = 1e-5
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.5
    
    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 50
    
    # Device and reproducibility
    device: str = 'auto'
    random_seed: int = 42
    
    def __post_init__(self):
        """Initialize default values for lists."""
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 128]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 3, 3]
        if self.mlp_hidden_dims is None:
            self.mlp_hidden_dims = [64, 64]


def load_config(config_path: str) -> TrainingConfig:
    """
    Load training configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        TrainingConfig instance
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        # Return default config if file doesn't exist
        return TrainingConfig()
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract training-related parameters
        training_params = {}
        
        # Map from config file structure to TrainingConfig
        if 'experiment' in config_dict:
            exp_config = config_dict['experiment']
            training_params.update({
                'batch_size': exp_config.get('batch_size', 64),
                'learning_rate': exp_config.get('learning_rate', 3e-4),
                'num_epochs': exp_config.get('epochs', 200),
                'early_stopping_patience': exp_config.get('early_stopping_patience', 20),
                'validation_split': exp_config.get('validation_split', 0.15),
                'random_seed': exp_config.get('random_seed', 42),
                'patch_size': exp_config.get('patch_size', [9, 9])[0] if isinstance(exp_config.get('patch_size', 9), list) else exp_config.get('patch_size', 9),
                'point_features_dim': exp_config.get('n_point_features', 10)
            })
        
        if 'model' in config_dict:
            model_config = config_dict['model']
            training_params.update({
                'cnn_channels': model_config.get('cnn_filters', [32, 64, 128]),
                'cnn_kernel_sizes': model_config.get('cnn_kernel_sizes', [3, 3, 3]),
                'mlp_hidden_dims': model_config.get('mlp_hidden_dims', [64, 64]),
                'fusion_dim': model_config.get('fusion_dim', 128),
                'dropout_rate': model_config.get('dropout_rate', 0.2)
            })
        
        if 'data' in config_dict:
            data_config = config_dict['data']
            training_params.update({
                'num_workers': data_config.get('num_workers', 4),
                'pin_memory': data_config.get('pin_memory', True)
            })
        
        return TrainingConfig(**training_params)
        
    except Exception as e:
        print(f"Warning: Error loading config file {config_path}: {e}")
        print("Using default configuration.")
        return TrainingConfig()


def save_config(config: TrainingConfig, output_path: str):
    """
    Save training configuration to YAML file.
    
    Args:
        config: TrainingConfig instance
        output_path: Path to save configuration
    """
    config_dict = {
        'experiment': {
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'epochs': config.num_epochs,
            'early_stopping_patience': config.early_stopping_patience,
            'validation_split': config.validation_split,
            'random_seed': config.random_seed,
            'patch_size': [config.patch_size, config.patch_size],
            'n_point_features': config.point_features_dim
        },
        'model': {
            'cnn_filters': config.cnn_channels,
            'cnn_kernel_sizes': config.cnn_kernel_sizes,
            'mlp_hidden_dims': config.mlp_hidden_dims,
            'fusion_dim': config.fusion_dim,
            'dropout_rate': config.dropout_rate
        },
        'data': {
            'num_workers': config.num_workers,
            'pin_memory': config.pin_memory
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)