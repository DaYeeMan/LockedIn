"""
Basic test script to verify MDA-CNN model functionality.
"""

import tensorflow as tf
import numpy as np
from models.mda_cnn import MDACNN, create_mda_cnn_model
from models.loss_functions import WeightedMSELoss, relative_percentage_error_metric


def test_basic_functionality():
    """Test basic MDA-CNN functionality."""
    print("Testing MDA-CNN basic functionality...")
    
    # Create model
    model = MDACNN(patch_size=(9, 9), n_point_features=8, dropout_rate=0.2)
    print(f"✓ Model created: {model.name}")
    
    # Create sample data
    batch_size = 4
    patches = tf.random.normal((batch_size, 9, 9, 1))
    features = tf.random.normal((batch_size, 8))
    
    print(f"✓ Sample data created - patches: {patches.shape}, features: {features.shape}")
    
    # Test forward pass
    output = model([patches, features], training=True)
    print(f"✓ Forward pass successful - output shape: {output.shape}")
    
    # Test with dictionary input
    dict_input = {'patches': patches, 'features': features}
    output_dict = model(dict_input, training=False)
    print(f"✓ Dictionary input successful - output shape: {output_dict.shape}")
    
    # Test model compilation
    compiled_model = create_mda_cnn_model(learning_rate=1e-3)
    print(f"✓ Model compilation successful")
    
    # Test training step
    targets = tf.random.normal((batch_size, 1))
    
    with tf.GradientTape() as tape:
        predictions = compiled_model([patches, features], training=True)
        loss = tf.reduce_mean(tf.keras.losses.mse(targets, predictions))
    
    gradients = tape.gradient(loss, compiled_model.trainable_variables)
    print(f"✓ Training step successful - loss: {loss.numpy():.6f}")
    
    # Test custom loss function
    custom_loss = WeightedMSELoss(wing_weight=2.0, atm_threshold=0.1)
    loss_value = custom_loss(targets, predictions)
    print(f"✓ Custom loss function works - loss: {loss_value.numpy():.6f}")
    
    # Test custom metric
    rpe = relative_percentage_error_metric(targets, predictions)
    print(f"✓ Custom metric works - RPE: {rpe.numpy():.2f}%")
    
    print("\n🎉 All tests passed successfully!")


def test_different_configurations():
    """Test different model configurations."""
    print("\nTesting different model configurations...")
    
    configs = [
        {'patch_size': (5, 5), 'n_point_features': 6, 'dropout_rate': 0.1},
        {'patch_size': (7, 7), 'n_point_features': 10, 'dropout_rate': 0.3},
        {'patch_size': (11, 11), 'n_point_features': 12, 'dropout_rate': 0.0}
    ]
    
    for i, config in enumerate(configs):
        model = MDACNN(**config)
        
        # Create appropriate input data
        patches = tf.random.normal((2, *config['patch_size'], 1))
        features = tf.random.normal((2, config['n_point_features']))
        
        output = model([patches, features])
        print(f"✓ Config {i+1}: patch_size={config['patch_size']}, "
              f"n_features={config['n_point_features']}, output_shape={output.shape}")
    
    print("✓ All configurations work correctly!")


if __name__ == "__main__":
    test_basic_functionality()
    test_different_configurations()