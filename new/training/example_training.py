"""
Example script demonstrating the training infrastructure.

This script shows how to use the training infrastructure to train
both MDA-CNN and Funahashi baseline models with proper configuration.
"""

import logging
from pathlib import Path

import tensorflow as tf
import numpy as np

from training.trainer import create_trainer
from training.training_config import create_mda_cnn_training_config, create_funahashi_training_config
from training.training_utils import set_random_seeds, create_experiment_directory
from config.config_manager import load_default_config
from models.mda_cnn import MDACNN
from models.baseline_models import FunahashiBaseline


def create_dummy_datasets(batch_size: int = 64) -> tuple:
    """
    Create dummy datasets for demonstration purposes.
    
    Args:
        batch_size: Batch size for datasets
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    def create_dataset(size: int) -> tf.data.Dataset:
        # Dummy patch data (9x9 patches)
        patches = tf.random.normal((size, 9, 9, 1))
        # Dummy point features (8 features: SABR params + strike + maturity + hagan_vol)
        features = tf.random.normal((size, 8))
        # Dummy residual targets
        targets = tf.random.normal((size, 1)) * 0.01  # Small residuals
        
        dataset = tf.data.Dataset.from_tensor_slices(((patches, features), targets))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    train_dataset = create_dataset(1000)
    val_dataset = create_dataset(200)
    test_dataset = create_dataset(200)
    
    return train_dataset, val_dataset, test_dataset


def example_mda_cnn_training():
    """Example of training MDA-CNN model."""
    print("=" * 60)
    print("Training MDA-CNN Model Example")
    print("=" * 60)
    
    # Setup
    set_random_seeds(42)
    config_manager = load_default_config()
    training_config = create_mda_cnn_training_config()
    
    # Reduce epochs for quick demonstration
    training_config.epochs = 5
    training_config.early_stopping_patience = 3
    
    # Create experiment directory
    experiment_dir = create_experiment_directory("results", "mda_cnn_example")
    config_manager.experiment_config.output_dir = str(experiment_dir)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_dummy_datasets(
        batch_size=training_config.batch_size
    )
    
    # Create MDA-CNN model
    model = MDACNN(
        patch_size=(9, 9),
        n_point_features=8,
        cnn_filters=[32, 64, 128],
        mlp_hidden_dims=[64, 64],
        fusion_dim=128,
        dropout_rate=0.2
    )
    
    # Create trainer
    trainer = create_trainer(
        config_manager=config_manager,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )
    
    # Update trainer config with training parameters
    trainer.config.epochs = training_config.epochs
    trainer.config.learning_rate = training_config.learning_rate
    trainer.config.early_stopping_patience = training_config.early_stopping_patience
    
    print(f"Training MDA-CNN for {training_config.epochs} epochs...")
    
    # Train model
    history = trainer.train()
    
    # Evaluate model
    test_results = trainer.evaluate()
    
    print(f"Training completed!")
    print(f"Final test loss: {test_results['loss']:.6f}")
    print(f"Final test RMSE: {test_results['rmse_metric']:.6f}")
    print(f"Results saved to: {experiment_dir}")
    
    return trainer


def example_funahashi_baseline_training():
    """Example of training Funahashi baseline model."""
    print("=" * 60)
    print("Training Funahashi Baseline Model Example")
    print("=" * 60)
    
    # Setup
    set_random_seeds(42)
    config_manager = load_default_config()
    training_config = create_funahashi_training_config()
    
    # Reduce epochs for quick demonstration
    training_config.epochs = 5
    training_config.early_stopping_patience = 3
    
    # Create experiment directory
    experiment_dir = create_experiment_directory("results", "funahashi_baseline_example")
    config_manager.experiment_config.output_dir = str(experiment_dir)
    
    # Create datasets (only point features for baseline)
    def create_baseline_dataset(size: int) -> tf.data.Dataset:
        # Only point features for baseline model
        features = tf.random.normal((size, 8))
        targets = tf.random.normal((size, 1)) * 0.01
        
        dataset = tf.data.Dataset.from_tensor_slices((features, targets))
        return dataset.batch(training_config.batch_size).prefetch(tf.data.AUTOTUNE)
    
    train_dataset = create_baseline_dataset(1000)
    val_dataset = create_baseline_dataset(200)
    test_dataset = create_baseline_dataset(200)
    
    # Create Funahashi baseline model
    model = FunahashiBaseline(
        input_dim=8,
        hidden_layers=5,
        neurons_per_layer=32,
        activation='relu',
        dropout_rate=0.0  # No dropout in baseline
    )
    
    # Create trainer
    trainer = create_trainer(
        config_manager=config_manager,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )
    
    # Update trainer config
    trainer.config.epochs = training_config.epochs
    trainer.config.learning_rate = training_config.learning_rate
    trainer.config.early_stopping_patience = training_config.early_stopping_patience
    
    print(f"Training Funahashi baseline for {training_config.epochs} epochs...")
    
    # Train model
    history = trainer.train()
    
    # Evaluate model
    test_results = trainer.evaluate()
    
    print(f"Training completed!")
    print(f"Final test loss: {test_results['loss']:.6f}")
    print(f"Final test RMSE: {test_results['rmse_metric']:.6f}")
    print(f"Results saved to: {experiment_dir}")
    
    return trainer


def run_training_comparison():
    """Run both models for comparison."""
    print("Running training comparison between MDA-CNN and Funahashi baseline...")
    
    # Train MDA-CNN
    mda_trainer = example_mda_cnn_training()
    
    print("\n" + "="*60 + "\n")
    
    # Train Funahashi baseline
    funahashi_trainer = example_funahashi_baseline_training()
    
    # Compare results
    print("\n" + "="*60)
    print("Training Comparison Results")
    print("="*60)
    
    mda_summary = mda_trainer.get_training_summary()
    funahashi_summary = funahashi_trainer.get_training_summary()
    
    print(f"MDA-CNN:")
    print(f"  - Best validation loss: {mda_summary['best_val_loss']:.6f}")
    print(f"  - Training time: {mda_summary.get('training_time_seconds', 0):.2f}s")
    
    print(f"Funahashi Baseline:")
    print(f"  - Best validation loss: {funahashi_summary['best_val_loss']:.6f}")
    print(f"  - Training time: {funahashi_summary.get('training_time_seconds', 0):.2f}s")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run examples
    print("SABR Volatility Surface Model Training Examples")
    print("=" * 60)
    
    try:
        # Run individual examples
        print("\n1. MDA-CNN Training Example:")
        example_mda_cnn_training()
        
        print("\n2. Funahashi Baseline Training Example:")
        example_funahashi_baseline_training()
        
        print("\n3. Training Comparison:")
        run_training_comparison()
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise