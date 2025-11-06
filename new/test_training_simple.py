"""
Simple test for training infrastructure.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path

# Import training components
from training.trainer import ModelTrainer
from training.training_config import TrainingConfig
from config.config_manager import ConfigManager, ExperimentConfig


def test_basic_training():
    """Test basic training functionality."""
    print("Testing Basic Training Infrastructure")
    print("=" * 40)
    
    # Create minimal configuration
    config_manager = ConfigManager()
    config_manager.experiment_config = ExperimentConfig(
        name="simple_test",
        output_dir="test_output",
        epochs=1,
        batch_size=16
    )
    
    # Create simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    # Create simple dataset
    x_train = tf.random.normal((100, 8))
    y_train = tf.random.normal((100, 1))
    x_val = tf.random.normal((20, 8))
    y_val = tf.random.normal((20, 1))
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(16)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(16)
    
    # Create trainer
    trainer = ModelTrainer(
        config_manager=config_manager,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    print("✓ Trainer created successfully")
    
    # Test training
    history = trainer.train()
    
    print("✓ Training completed successfully")
    print(f"✓ Final training loss: {history.history['loss'][-1]:.6f}")
    
    return trainer


if __name__ == "__main__":
    tf.random.set_seed(42)
    
    try:
        trainer = test_basic_training()
        print("\n🎉 Training infrastructure test passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()