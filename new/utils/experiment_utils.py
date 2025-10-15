"""
Experiment utilities for managing the complete experimental workflow.
Combines configuration, logging, and reproducibility management.
"""

import os
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime

from new.utils.logging_utils import get_experiment_logger, setup_logger
from new.utils.seed_utils import ReproducibilityManager, setup_reproducibility
from new.config.config_manager import ConfigManager, load_config_from_file


class ExperimentManager:
    """
    Central manager for SABR MDA-CNN experiments.
    Handles configuration, logging, reproducibility, and output organization.
    """
    
    def __init__(
        self,
        experiment_name: str,
        config_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        seed: int = 42
    ):
        """
        Initialize experiment manager.
        
        Args:
            experiment_name: Name of the experiment
            config_path: Path to configuration file (optional)
            output_dir: Output directory for results (optional)
            seed: Random seed for reproducibility
        """
        self.experiment_name = experiment_name
        self.seed = seed
        self.start_time = datetime.now()
        
        # Load configuration
        if config_path:
            self.config_manager = load_config_from_file(config_path)
        else:
            self.config_manager = ConfigManager()
        
        # Override output directory if provided
        if output_dir:
            self.config_manager.experiment_config.output_dir = str(output_dir)
        
        # Update experiment name in config
        self.config_manager.experiment_config.name = experiment_name
        
        # Set up reproducibility
        self.reproducibility_manager = ReproducibilityManager(seed, deterministic=True)
        
        # Initialize logger (will be set up in setup())
        self.logger = None
        
        # Track experiment state
        self.is_setup = False
        self.experiment_id = None
    
    def setup(self) -> None:
        """Set up the experiment environment."""
        if self.is_setup:
            return
        
        # Generate unique experiment ID
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{self.experiment_name}_{timestamp}"
        
        # Create output directories
        self.config_manager.create_output_dirs()
        
        # Set up reproducibility
        self.reproducibility_manager.setup()
        
        # Set up logging
        self.logger = get_experiment_logger(
            self.experiment_id,
            self.config_manager.experiment_config.output_dir
        )
        
        # Log experiment start
        self.logger.info(f"Starting experiment: {self.experiment_id}")
        self.logger.info(f"Random seed: {self.seed}")
        self.logger.info(f"Output directory: {self.config_manager.experiment_config.output_dir}")
        
        # Save configuration
        config_path = Path(self.config_manager.experiment_config.output_dir) / f"config_{self.experiment_id}.yaml"
        self.config_manager.save_config(config_path)
        self.logger.info(f"Configuration saved to: {config_path}")
        
        self.is_setup = True
    
    def get_config(self) -> ConfigManager:
        """Get the configuration manager."""
        return self.config_manager
    
    def get_logger(self):
        """Get the experiment logger."""
        if not self.is_setup:
            self.setup()
        return self.logger
    
    def get_output_path(self, *path_parts: str) -> Path:
        """
        Get path within the experiment output directory.
        
        Args:
            *path_parts: Path components to join
            
        Returns:
            Full path within output directory
        """
        base_path = Path(self.config_manager.experiment_config.output_dir)
        return base_path.joinpath(*path_parts)
    
    def create_subdirectory(self, *path_parts: str) -> Path:
        """
        Create a subdirectory within the experiment output directory.
        
        Args:
            *path_parts: Path components to join
            
        Returns:
            Created directory path
        """
        dir_path = self.get_output_path(*path_parts)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        if not self.is_setup:
            self.setup()
        
        extra_fields = {'metric_name': name, 'metric_value': value}
        if step is not None:
            extra_fields['step'] = step
        
        self.logger.info(f"Metric {name}: {value}", extra=extra_fields)
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters.
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        if not self.is_setup:
            self.setup()
        
        self.logger.info("Hyperparameters:", extra={'hyperparameters': hyperparams})
        for name, value in hyperparams.items():
            self.logger.info(f"  {name}: {value}")
    
    def save_artifact(self, artifact: Any, filename: str, subdir: str = "artifacts") -> Path:
        """
        Save an experiment artifact.
        
        Args:
            artifact: Artifact to save (will be pickled)
            filename: Filename for the artifact
            subdir: Subdirectory within output directory
            
        Returns:
            Path where artifact was saved
        """
        import pickle
        
        artifact_dir = self.create_subdirectory(subdir)
        artifact_path = artifact_dir / filename
        
        with open(artifact_path, 'wb') as f:
            pickle.dump(artifact, f)
        
        if self.logger:
            self.logger.info(f"Artifact saved: {artifact_path}")
        
        return artifact_path
    
    def finalize(self) -> None:
        """Finalize the experiment and log summary."""
        if not self.is_setup:
            return
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.logger.info(f"Experiment completed: {self.experiment_id}")
        self.logger.info(f"Total duration: {duration}")
        self.logger.info(f"Results saved in: {self.config_manager.experiment_config.output_dir}")
    
    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None and self.logger:
            self.logger.error(f"Experiment failed with {exc_type.__name__}: {exc_val}")
        self.finalize()


def create_experiment(
    name: str,
    config_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    seed: int = 42
) -> ExperimentManager:
    """
    Create a new experiment manager.
    
    Args:
        name: Experiment name
        config_path: Path to configuration file
        output_dir: Output directory for results
        seed: Random seed
        
    Returns:
        Configured experiment manager
    """
    return ExperimentManager(name, config_path, output_dir, seed)


def setup_basic_experiment(seed: int = 42, log_level: str = "INFO") -> None:
    """
    Set up basic experiment environment without full experiment manager.
    Useful for quick scripts and testing.
    
    Args:
        seed: Random seed
        log_level: Logging level
    """
    # Set up reproducibility
    setup_reproducibility(seed, deterministic=True)
    
    # Set up basic logging
    logger = setup_logger("sabr_mdacnn", log_level=log_level)
    logger.info(f"Basic experiment setup complete with seed {seed}")
    
    return logger