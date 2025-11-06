"""
Example usage of baseline models for SABR volatility surface modeling.
"""

import numpy as np
import tensorflow as tf
from baseline_models import create_funahashi_baseline, get_all_baseline_models


def demonstrate_baseline_models():
    """Demonstrate baseline model usage."""
    print("=== Baseline Models Demo ===")
    
    # Create Funahashi baseline
    model = create_funahashi_baseline(n_point_features=8)
    print(f"Funahashi model created with {model.count_params():,} parameters")
    
    # Test prediction
    dummy_input = np.random.randn(5, 8)
    predictions = model.predict(dummy_input, verbose=0)
    print(f"Predictions shape: {predictions.shape}")
    
    # Create all models
    models = get_all_baseline_models()
    print(f"Created {len(models)} baseline models: {list(models.keys())}")
    
    print("Baseline models working correctly!")


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    demonstrate_baseline_models()