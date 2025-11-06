"""
Main training script for SABR volatility surface models.

This script provides a complete training pipeline that can be used
to train both MDA-CNN and baseline models with proper configuration,
logging, and model saving.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf
import numpy as np

from training.trainer import ModelTrainer, create_trainer
from training.training_config import TrainingConfig, create_mda_cnn_training_config, create_funahashi_training_config
from training.training_utils import (
    setup_mixed_precision,
    set_random_seeds,
    save_training_config,
    create_experiment_directory,
    log_model_summary
)
from config.config_manager import ConfigManager, load_default_config
from models.mda_cnn import MDACNN
from models.baseline_models import FunahashiBaseline


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    return logging.getLogger(__name__)


def load_datasets(data_dir: str, 
                 batch_size: int = 64,
                 validation_split: float = 0.15) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Load and prepare datasets for training.
    
    Args:
        data_dir: Directory containing processed data
        batch_size: Batch size for datasets
        validation_split: Fraction of data for validation
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # This is a placeholder implementation
    # In practice, this would load the actual preprocessed data
    # from the data generation and preprocessing modules
    
    logger = logging.getLogger(__name__)
    logger.warning("Using dummy datasets - replace with actual data loading")
    
    # Create dummy datasets for demonstration
    # Replace this with actual data loading logic
    def create_dummy_dataset(size: int) -> tf.data.Dataset:
        # Dummy patch data (batch_size, 9, 9, 1)
        patches = tf.random.normal((size, 9, 9, 1))
        # Dummy point features (batch_size, 8)
        features = tf.random.normal((size, 8))
        # Dummy residual targets (batch_size, 1)
        targets = tf.random.normal((size, 1))
        
        dataset = tf.data.Dataset.from_tensor_slices(((patches, features), targets))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create datasets with different sizes
    train_size = 8000
    val_size = int(train_size * validation_split)
    test_size = 2000
    
    train_dataset = create_dummy_dataset(train_size)
    val_dataset = create_dummy_dataset(val_size)
    test_dataset = create_dummy_dataset(test_size)
    
    logger.info(f"Datasets created - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    return train_dataset, val_dataset, test_dataset


def create_model(model_type: str, config_manager: ConfigManager) -> tf.keras.Model:
    """
    Create model based on type and configuration.
    
    Args:
        model_type: Type of model ("mda_cnn" or "funahashi_baseline")
        config_manager: Configuration manager
        
    Returns:
        Keras model instance
    """
    model_config = config_manager.get_model_config()
    experiment_config = config_manager.get_experiment_config()
    
    if model_type == "mda_cnn":
        model = MDACNN(
            patch_size=experiment_config.patch_size,
            n_point_features=experiment_config.n_point_features,
            cnn_filters=model_config.cnn_filters,
            mlp_hidden_dims=model_config.mlp_hidden_dims,
            fusion_dim=model_config.fusion_dim,
            dropout_rate=model_config.dropout_rate
        )
    elif model_type == "funahashi_baseline":
        model = FunahashiBaseline(
            input_dim=experiment_config.n_point_features,
            hidden_layers=5,  # Funahashi's exact architecture
            neurons_per_layer=32,
            activation='relu',
            dropout_rate=0.0  # No dropout in baseline
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def train_model(model_type: str,
               config_path: Optional[str] = None,
               data_dir: str = "data/processed",
               output_dir: str = "results",
               experiment_name: Optional[str] = None) -> ModelTrainer:
    """
    Main training function.
    
    Args:
        model_type: Type of model to train ("mda_cnn" or "funahashi_baseline")
        config_path: Optional path to configuration file
        data_dir: Directory containing processed data
        output_dir: Output directory for results
        experiment_name: Optional experiment name
        
    Returns:
        Trained ModelTrainer instance
    """
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting training for {model_type}")
    
    # Load configuration
    if config_path:
        config_manager = ConfigManager(config_path)
    else:
        config_manager = load_default_config()
    
    # Create experiment directory
    if experiment_name is None:
        experiment_name = f"{model_type}_experiment"
    
    experiment_dir = create_experiment_directory(output_dir, experiment_name)
    
    # Update config with experiment directory
    config_manager.experiment_config.output_dir = str(experiment_dir)
    
    # Create training configuration
    if model_type == "mda_cnn":
        training_config = create_mda_cnn_training_config()
    elif model_type == "funahashi_baseline":
        training_config = create_funahashi_training_config()
    else:
        training_config = TrainingConfig()
    
    # Setup training environment
    setup_mixed_precision(training_config.mixed_precision_enabled)
    set_random_seeds(training_config.random_seed)
    
    # Save training configuration
    config_path = experiment_dir / "configs" / "training_config.json"
    save_training_config(training_config, config_path)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset, val_dataset, test_dataset = load_datasets(
        data_dir=data_dir,
        batch_size=training_config.batch_size,
        validation_split=training_config.validation_split
    )
    
    # Create model
    logger.info(f"Creating {model_type} model...")
    model = create_model(model_type, config_manager)
    
    # Log model summary
    log_model_summary(model, logger)
    
    # Create trainer
    trainer = create_trainer(
        config_manager=config_manager,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )
    
    # Update trainer with training config
    trainer.config = config_manager.get_experiment_config()
    trainer.config.epochs = training_config.epochs
    trainer.config.batch_size = training_config.batch_size
    trainer.config.learning_rate = training_config.learning_rate
    trainer.config.early_stopping_patience = training_config.early_stopping_patience
    
    # Train model
    logger.info("Starting model training...")
    history = trainer.train()
    
    # Evaluate model
    if test_dataset is not None:
        logger.info("Evaluating model on test dataset...")
        test_results = trainer.evaluate(test_dataset)
        logger.info(f"Test results: {test_results}")
    
    # Log training summary
    summary = trainer.get_training_summary()
    logger.info("Training completed successfully!")
    logger.info(f"Training summary: {summary}")
    
    return trainer


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description="Train SABR volatility surface models")
    
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["mda_cnn", "funahashi_baseline"],
        default="mda_cnn",
        help="Type of model to train"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for the experiment"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging with specified level
    setup_logging(args.log_level)
    
    try:
        # Train model
        trainer = train_model(
            model_type=args.model_type,
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name
        )
        
        print(f"Training completed successfully!")
        print(f"Results saved to: {trainer.output_dir}")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()