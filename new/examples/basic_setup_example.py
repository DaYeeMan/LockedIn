"""
Basic example demonstrating the project setup and core utilities.
This script shows how to use the configuration, logging, and reproducibility systems.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import with absolute paths from the new directory
sys.path.insert(0, str(project_root.parent))

from new.utils.experiment_utils import create_experiment, setup_basic_experiment
from new.config.config_manager import load_default_config


def basic_setup_example():
    """Demonstrate basic project setup."""
    print("=== Basic Setup Example ===")
    
    # Method 1: Quick setup for simple scripts
    print("\n1. Quick setup:")
    logger = setup_basic_experiment(seed=42, log_level="INFO")
    logger.info("This is a basic setup example")
    
    # Method 2: Full experiment manager
    print("\n2. Full experiment manager:")
    with create_experiment("basic_example", seed=42) as experiment:
        logger = experiment.get_logger()
        config = experiment.get_config()
        
        logger.info("Experiment started successfully")
        logger.info(f"Random seed: {config.experiment_config.random_seed}")
        logger.info(f"Batch size: {config.experiment_config.batch_size}")
        
        # Log some example metrics
        experiment.log_metric("example_loss", 0.123)
        experiment.log_metric("example_accuracy", 0.95)
        
        # Log hyperparameters
        experiment.log_hyperparameters({
            "learning_rate": config.experiment_config.learning_rate,
            "batch_size": config.experiment_config.batch_size,
            "epochs": config.experiment_config.epochs
        })
        
        # Create some output directories
        model_dir = experiment.create_subdirectory("models")
        plot_dir = experiment.create_subdirectory("plots")
        
        logger.info(f"Created model directory: {model_dir}")
        logger.info(f"Created plot directory: {plot_dir}")
        
        # Save a simple artifact
        example_data = {"message": "Hello from SABR MDA-CNN!"}
        artifact_path = experiment.save_artifact(example_data, "example.pkl")
        logger.info(f"Saved example artifact to: {artifact_path}")


def config_example():
    """Demonstrate configuration management."""
    print("\n=== Configuration Example ===")
    
    # Load default configuration
    config_manager = load_default_config()
    
    print(f"Experiment name: {config_manager.experiment_config.name}")
    print(f"Random seed: {config_manager.experiment_config.random_seed}")
    print(f"Batch size: {config_manager.experiment_config.batch_size}")
    print(f"Learning rate: {config_manager.experiment_config.learning_rate}")
    print(f"SABR alpha range: {config_manager.experiment_config.alpha_range}")
    print(f"Model type: {config_manager.model_config.model_type}")
    print(f"CNN filters: {config_manager.model_config.cnn_filters}")
    
    # Save configuration to file
    output_path = Path("example_config.yaml")
    config_manager.save_config(output_path)
    print(f"Configuration saved to: {output_path}")
    
    # Clean up
    if output_path.exists():
        output_path.unlink()
        print("Cleaned up example configuration file")


if __name__ == "__main__":
    basic_setup_example()
    config_example()
    print("\n=== Setup Complete ===")
    print("The project structure and core utilities are ready!")
    print("Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run this example: python examples/basic_setup_example.py")
    print("3. Start implementing the next task in the implementation plan")