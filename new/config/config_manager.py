"""
Configuration management system for experiments and hyperparameters.
Supports YAML and JSON configuration files with validation and defaults.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for experiment settings."""
    name: str = "sabr_mdacnn_experiment"
    description: str = "SABR MDA-CNN volatility surface modeling"
    output_dir: str = "results"
    random_seed: int = 42
    
    # Data generation settings
    n_parameter_sets: int = 1000
    hf_budget: int = 200
    mc_paths: int = 100000
    
    # Model architecture settings
    patch_size: tuple = (9, 9)
    n_point_features: int = 8
    
    # Training settings
    batch_size: int = 64
    epochs: int = 200
    learning_rate: float = 3e-4
    validation_split: float = 0.15
    early_stopping_patience: int = 20
    
    # SABR parameter ranges (Funahashi's ranges)
    alpha_range: tuple = (0.05, 0.6)
    beta_range: tuple = (0.3, 0.9)
    nu_range: tuple = (0.05, 0.9)
    rho_range: tuple = (-0.75, 0.75)
    maturity_range: tuple = (1.0, 10.0)
    
    # Grid configuration
    n_strikes: int = 21  # Funahashi's 21 strikes
    n_maturities: int = 10


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    model_type: str = "mda_cnn"
    
    # CNN branch settings
    cnn_filters: list = None
    cnn_kernel_sizes: list = None
    cnn_activation: str = "relu"
    
    # MLP branch settings
    mlp_hidden_dims: list = None
    mlp_activation: str = "relu"
    
    # Fusion settings
    fusion_dim: int = 128
    dropout_rate: float = 0.2
    
    def __post_init__(self):
        if self.cnn_filters is None:
            self.cnn_filters = [32, 64, 128]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 3, 3]
        if self.mlp_hidden_dims is None:
            self.mlp_hidden_dims = [64, 64]


@dataclass
class DataConfig:
    """Configuration for data processing."""
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    
    # Normalization settings
    normalize_features: bool = True
    normalization_method: str = "standard"  # "standard", "minmax", "robust"
    
    # Data loading settings
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True


class ConfigManager:
    """
    Configuration manager for handling experiment configurations.
    Supports loading from YAML/JSON files and environment variables.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self.config_path = Path(config_path) if config_path else None
        self.experiment_config = ExperimentConfig()
        self.model_config = ModelConfig()
        self.data_config = DataConfig()
        
        if self.config_path and self.config_path.exists():
            self.load_config()
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path or not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        # Load configuration based on file extension
        if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif self.config_path.suffix.lower() == '.json':
            with open(self.config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {self.config_path.suffix}")
        
        # Update configurations
        if 'experiment' in config_dict:
            self._update_config(self.experiment_config, config_dict['experiment'])
        
        if 'model' in config_dict:
            self._update_config(self.model_config, config_dict['model'])
        
        if 'data' in config_dict:
            self._update_config(self.data_config, config_dict['data'])
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        
        config_dict = {
            'experiment': asdict(self.experiment_config),
            'model': asdict(self.model_config),
            'data': asdict(self.data_config)
        }
        
        # Save based on file extension
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif output_path.suffix.lower() == '.json':
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
    
    def _update_config(self, config_obj: Any, config_dict: Dict[str, Any]) -> None:
        """Update configuration object with dictionary values."""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
    
    def get_experiment_config(self) -> ExperimentConfig:
        """Get experiment configuration."""
        return self.experiment_config
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self.model_config
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration."""
        return self.data_config
    
    def update_from_env(self, prefix: str = "SABR_") -> None:
        """
        Update configuration from environment variables.
        
        Args:
            prefix: Prefix for environment variables
        """
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                
                # Try to update experiment config
                if hasattr(self.experiment_config, config_key):
                    # Convert string to appropriate type
                    current_value = getattr(self.experiment_config, config_key)
                    converted_value = self._convert_env_value(value, type(current_value))
                    setattr(self.experiment_config, config_key, converted_value)
    
    def _convert_env_value(self, value: str, target_type: type) -> Any:
        """Convert environment variable string to target type."""
        if target_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == tuple:
            # Parse tuple from string like "(1,2)" or "1,2"
            value = value.strip('()')
            return tuple(map(float, value.split(',')))
        else:
            return value
    
    def create_output_dirs(self) -> None:
        """Create necessary output directories."""
        dirs_to_create = [
            self.experiment_config.output_dir,
            self.data_config.data_dir,
            self.data_config.raw_data_dir,
            self.data_config.processed_data_dir,
            f"{self.experiment_config.output_dir}/models",
            f"{self.experiment_config.output_dir}/plots",
            f"{self.experiment_config.output_dir}/logs"
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


def load_default_config() -> ConfigManager:
    """Load default configuration."""
    return ConfigManager()


def load_config_from_file(config_path: Union[str, Path]) -> ConfigManager:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigManager instance with loaded configuration
    """
    return ConfigManager(config_path)