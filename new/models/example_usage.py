"""
Example usage of MDA-CNN model for SABR volatility surface modeling.

This script demonstrates how to create, configure, and use the MDA-CNN model
for predicting residuals between Monte Carlo and Hagan surfaces.
"""

import tensorflow as tf
import numpy as np
from mda_cnn import MDACNN, create_mda_cnn_model, build_model_with_inputs
from loss_functions import WeightedMSELoss, create_wing_weighted_mse, relative_percentage_error_metric, rmse_metric


def create_sample_data(batch_size=32, patch_size=(9, 9), n_features=8):
    """Create sample training data for demonstration."""
    
    # Sample LF surface patches (normalized Hagan volatility patches)
    patches = np.random.normal(0.2, 0.05, (batch_size, *patch_size, 1))
    patches = np.clip(patches, 0.01, 1.0)  # Realistic volatility range
    
    # Sample point features: [alpha, beta, nu, rho, strike, maturity, forward, hagan_vol]
    features = np.random.uniform(0, 1, (batch_size, n_features))
    
    # Normalize features to realistic SABR parameter ranges
    features[:, 0] = features[:, 0] * 0.55 + 0.05  # alpha: [0.05, 0.6]
    features[:, 1] = features[:, 1] * 0.6 + 0.3     # beta: [0.3, 0.9]
    features[:, 2] = features[:, 2] * 0.85 + 0.05   # nu: [0.05, 0.9]
    features[:, 3] = features[:, 3] * 1.5 - 0.75    # rho: [-0.75, 0.75]
    features[:, 4] = features[:, 4] * 1.6 + 0.4     # strike: [0.4, 2.0]
    features[:, 5] = features[:, 5] * 9 + 1         # maturity: [1, 10]
    features[:, 6] = 1.0                            # forward: fixed at 1.0
    features[:, 7] = features[:, 7] * 0.5 + 0.1     # hagan_vol: [0.1, 0.6]
    
    # Sample residuals (MC - Hagan differences)
    # Wing regions typically have larger residuals
    residuals = np.random.normal(0, 0.02, (batch_size, 1))
    
    # Add some wing effects (larger residuals for extreme strikes)
    wing_mask = (features[:, 4] < 0.7) | (features[:, 4] > 1.3)  # Moneyness < 0.7 or > 1.3
    residuals[wing_mask] *= 3  # Amplify wing residuals
    
    return patches.astype(np.float32), features.astype(np.float32), residuals.astype(np.float32)


def demonstrate_basic_usage():
    """Demonstrate basic MDA-CNN model usage."""
    print("=== MDA-CNN Basic Usage Demo ===\n")
    
    # 1. Create model using factory function
    print("1. Creating MDA-CNN model...")
    model = create_mda_cnn_model(
        patch_size=(9, 9),
        n_point_features=8,
        dropout_rate=0.2,
        learning_rate=3e-4
    )
    
    # Build the model by running a forward pass
    dummy_patches = tf.random.normal((1, 9, 9, 1))
    dummy_features = tf.random.normal((1, 8))
    _ = model([dummy_patches, dummy_features])
    
    print(f"   ✓ Model created with {model.count_params():,} parameters")
    
    # 2. Create sample data
    print("\n2. Creating sample training data...")
    patches, features, targets = create_sample_data(batch_size=64)
    
    print(f"   ✓ Patches shape: {patches.shape}")
    print(f"   ✓ Features shape: {features.shape}")
    print(f"   ✓ Targets shape: {targets.shape}")
    
    # 3. Test forward pass
    print("\n3. Testing forward pass...")
    predictions = model([patches, features], training=False)
    
    print(f"   ✓ Predictions shape: {predictions.shape}")
    print(f"   ✓ Sample predictions: {predictions[:5].numpy().flatten()}")
    
    # 4. Calculate metrics
    print("\n4. Calculating evaluation metrics...")
    mse = tf.reduce_mean(tf.square(targets - predictions))
    mae = tf.reduce_mean(tf.abs(targets - predictions))
    rpe = relative_percentage_error_metric(targets, predictions)
    rmse = rmse_metric(targets, predictions)
    
    print(f"   ✓ MSE: {mse.numpy():.6f}")
    print(f"   ✓ MAE: {mae.numpy():.6f}")
    print(f"   ✓ RMSE: {rmse.numpy():.6f}")
    print(f"   ✓ RPE: {rpe.numpy():.2f}%")
    
    return model, patches, features, targets


def demonstrate_custom_loss():
    """Demonstrate custom loss functions for wing emphasis."""
    print("\n=== Custom Loss Functions Demo ===\n")
    
    # Create model with custom loss
    print("1. Creating model with weighted MSE loss...")
    model = MDACNN(patch_size=(9, 9), n_point_features=8, dropout_rate=0.2)
    
    # Use custom weighted loss for wing emphasis
    wing_loss = create_wing_weighted_mse(wing_weight=2.5, atm_threshold=0.01)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=wing_loss,
        metrics=[rmse_metric, relative_percentage_error_metric]
    )
    
    print("   ✓ Model compiled with weighted MSE loss")
    
    # Create data with pronounced wing effects
    patches, features, targets = create_sample_data(batch_size=32)
    
    # Test loss calculation
    predictions = model([patches, features])
    loss_value = wing_loss(targets, predictions)
    
    print(f"   ✓ Weighted MSE loss: {loss_value.numpy():.6f}")
    
    # Compare with standard MSE
    standard_mse = tf.reduce_mean(tf.square(targets - predictions))
    print(f"   ✓ Standard MSE: {standard_mse.numpy():.6f}")
    print(f"   ✓ Loss ratio (weighted/standard): {loss_value.numpy()/standard_mse.numpy():.2f}")


def demonstrate_training_step():
    """Demonstrate a complete training step."""
    print("\n=== Training Step Demo ===\n")
    
    # Create model
    model = create_mda_cnn_model(learning_rate=1e-3)
    
    # Create training data
    patches, features, targets = create_sample_data(batch_size=16)
    
    print("1. Performing training step...")
    
    # Training step with gradient tape
    with tf.GradientTape() as tape:
        predictions = model([patches, features], training=True)
        loss = tf.reduce_mean(tf.square(targets - predictions))
    
    # Calculate gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Apply gradients
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    print(f"   ✓ Training loss: {loss.numpy():.6f}")
    print(f"   ✓ Gradients computed for {len(gradients)} variables")
    print(f"   ✓ Gradient norms: min={min(tf.norm(g).numpy() for g in gradients if g is not None):.6f}, "
          f"max={max(tf.norm(g).numpy() for g in gradients if g is not None):.6f}")
    
    # Test inference after training step
    new_predictions = model([patches, features], training=False)
    new_loss = tf.reduce_mean(tf.square(targets - new_predictions))
    
    print(f"   ✓ Post-training loss: {new_loss.numpy():.6f}")
    print(f"   ✓ Loss change: {new_loss.numpy() - loss.numpy():.6f}")


def demonstrate_different_architectures():
    """Demonstrate different model configurations."""
    print("\n=== Different Architecture Configurations ===\n")
    
    configs = [
        {"name": "Small", "patch_size": (5, 5), "n_point_features": 6, "dropout_rate": 0.1},
        {"name": "Standard", "patch_size": (9, 9), "n_point_features": 8, "dropout_rate": 0.2},
        {"name": "Large", "patch_size": (13, 13), "n_point_features": 12, "dropout_rate": 0.3}
    ]
    
    for config in configs:
        print(f"Testing {config['name']} configuration...")
        
        model = MDACNN(
            patch_size=config['patch_size'],
            n_point_features=config['n_point_features'],
            dropout_rate=config['dropout_rate']
        )
        
        # Create appropriate test data
        patches = tf.random.normal((4, *config['patch_size'], 1))
        features = tf.random.normal((4, config['n_point_features']))
        
        output = model([patches, features])
        param_count = model.count_params()
        
        print(f"   ✓ Patch size: {config['patch_size']}, Features: {config['n_point_features']}")
        print(f"   ✓ Parameters: {param_count:,}, Output shape: {output.shape}")
        print()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Run demonstrations
    model, patches, features, targets = demonstrate_basic_usage()
    demonstrate_custom_loss()
    demonstrate_training_step()
    demonstrate_different_architectures()
    
    print("🎉 All demonstrations completed successfully!")
    print("\nThe MDA-CNN model is ready for SABR volatility surface modeling!")
    print("Next steps:")
    print("- Integrate with data generation pipeline")
    print("- Implement training loop with validation")
    print("- Add evaluation against Funahashi baseline")
    print("- Create visualization tools for results")