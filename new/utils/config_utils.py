"""
Configuration utilities for loading and saving experiment configurations.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union

from config.training_config import TrainingConfig


def load_config(config_path: str) -> TrainingConfig:
    """
    Load training configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        TrainingConfig instance
    """
    from config.training_config import load_config as load_training_config
    return load_training_config(config_path)


def save_config(config: TrainingConfig, output_path: str):
    """
    Save training configuration to file.
    
    Args:
        config: TrainingConfig instance
        output_path: Path to save configuration
    """
    from config.training_config import save_config as save_training_config
    save_training_config(config, output_path)


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dictionary with loaded data
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]):
    """
    Save data to YAML file.
    
    Args:
        data: Data to save
        file_path: Path to save file
    """
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary with loaded data
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Union[str, Path]):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save file
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)