"""
Test script for training infrastructure.

This script tests the core training functionality to ensure
all components work together correctly.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from new.training.trainer import ModelTrainer, create_trainer
from new.training.training_config import TrainingConfig
from new.config.config_manager import ConfigManager, ExperimentConfig
from new.models.mda_cnn import MDACNN


def test_training_infrastructure():
    """Test the training infrastructure with a simple model and data."""
    print("Testing SABR Training Infrastructure")
    print("=" * 50)
    
    # Create simple configuration
    config_manager = ConfigManager()
    config_manager.experiment_config = ExperimentConfig(
        name="test_experiment",
        output_dir="test_results",
        epochs=2,  # Very short for testing
        batch_size=32,
        learning_rate=1e-3,
        early_stopping_patience=1
    )
    
    # Create simple datasets
    def create_test_dataset(size: int = 100) -> tf.data.Dataset:
        # Simple synthetic data
        patches = tf.random.normal((size, 9, 9, 1))
        features = tf.random.normal((size, 8))
        targets = tf.random.normal((size, 1)) * 0.01
        
        dataset = tf.data.Dataset.from_tensor_slices(((patches, features), targets))
        return dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    
    train_dataset = create_test_dataset(100)
    val_dataset = create_test_dataset(50)
    test_dataset = create_test_dataset(50)
    
    print("✓ Test datasets created")
    
    # Create simple MDA-CNN model
    model = MDACNN(
        patch_size=(9, 9),
        n_point_features=8,
        cnn_filters=[16, 32],  # Smaller for testing
        mlp_hidden_dims=[32],
        fusion_dim=64,
        dropout_rate=0.1
    )
    
    print("✓ MDA-CNN model created")
    
    # Create trainer
    trainer = create_trainer(
        config_manager=config_manager,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )
    
    print("✓ Trainer created")
    
    # Test model compilation
    print(f"✓ Model compiled with {trainer.model.optimizer.__class__.__name__} optimizer")
    print(f"✓ Loss function: {trainer.model.loss.__class__.__name__}")
    
    # Test training for a few epochs
    print("\nStarting training test...")
    history = trainer.train()
    
    print("✓ Training completed successfully")
    
    # Test evaluation
    test_results = trainer.evaluate()
    print(f"✓ Evaluation completed - Test loss: {test_results['loss']:.6f}")
    
    # Test model saving/loading
    model_path = Path("test_results/models/test_model.h5")
    if model_path.exists():
        print("✓ Model saved successfully")
    
    # Test training summary
    summary = trainer.get_training_summary()
    print(f"✓ Training summary generated - Best val loss: {summary['best_val_loss']:.6f}")
    
    print("\n" + "=" * 50)
    print("All training infrastructure tests passed! ✓")
    
    return trainer


def test_training_config():
    """Test training configuration classes."""
    print("\nTesting Training Configuration")
    print("-" * 30)
    
    # Test basic training config
    config = TrainingConfig(
        epochs=100,
        batch_size=64,
        learning_rate=3e-4
    )
    
    print(f"✓ Basic config created - Epochs: {config.epochs}")
    
    # Test optimizer config
    optimizer_config = config.get_optimizer_config()
    print(f"✓ Optimizer config: {optimizer_config}")
    
    # Test callback configs
    callback_configs = config.get_callback_configs()
    print(f"✓ Callback configs: {list(callback_configs.keys())}")
    
    print("✓ Training configuration tests passed!")


if __name__ == "__main__":
    # Set random seed for reproducible testing
    tf.random.set_seed(42)
    np.random.seed(42)
    
    try:
        # Test configuration
        test_training_config()
        
        # Test full training infrastructure
        trainer = test_training_infrastructure()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise