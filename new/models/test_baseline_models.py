"""
Tests for baseline models in SABR volatility surface modeling.

This module tests all baseline model implementations including:
- Funahashi baseline model
- Direct MLP model
- Residual MLP model
- CNN-only model
- Model interfaces and factory functions
"""

import unittest
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tempfile
import os

import sys
import os
sys.path.append(os.path.dirname(__file__))

from baseline_models import (
    FunahashiBaseline,
    DirectMLPModel,
    ResidualMLPModel,
    CNNOnlyModel,
    create_funahashi_baseline,
    create_direct_mlp_model,
    create_residual_mlp_model,
    create_cnn_only_model,
    get_all_baseline_models,
    BaselineModelInterface,
    create_baseline_interface
)


class TestFunahashiBaseline(unittest.TestCase):
    """Test cases for Funahashi baseline model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_point_features = 8
        self.batch_size = 32
        self.model = FunahashiBaseline(n_point_features=self.n_point_features)
        
    def test_model_creation(self):
        """Test model creation and architecture."""
        self.assertIsInstance(self.model, FunahashiBaseline)
        self.assertEqual(self.model.n_point_features, self.n_point_features)
        
    def test_model_forward_pass(self):
        """Test forward pass with dummy data."""
        # Create dummy input data
        dummy_input = tf.random.normal((self.batch_size, self.n_point_features))
        
        # Forward pass
        output = self.model(dummy_input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
        
    def test_model_layers(self):
        """Test model has correct layer structure."""
        # Build model with dummy input
        dummy_input = tf.random.normal((1, self.n_point_features))
        _ = self.model(dummy_input)
        
        # Check layer count (5 dense + 1 output = 6 total)
        self.assertEqual(len(self.model.layers), 6)
        
        # Check layer configurations
        for i in range(5):  # First 5 layers should have 32 units and ReLU
            layer = self.model.layers[i]
            self.assertEqual(layer.units, 32)
            self.assertEqual(layer.activation.__name__, 'relu')
            
        # Output layer should have 1 unit and linear activation
        output_layer = self.model.layers[-1]
        self.assertEqual(output_layer.units, 1)
        self.assertEqual(output_layer.activation.__name__, 'linear')
        
    def test_model_config(self):
        """Test model configuration serialization."""
        config = self.model.get_config()
        self.assertIn('n_point_features', config)
        self.assertEqual(config['n_point_features'], self.n_point_features)


class TestDirectMLPModel(unittest.TestCase):
    """Test cases for Direct MLP model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_point_features = 7  # No Hagan vol
        self.batch_size = 32
        self.hidden_units = [64, 64, 32]
        self.model = DirectMLPModel(
            n_point_features=self.n_point_features,
            hidden_units=self.hidden_units
        )
        
    def test_model_creation(self):
        """Test model creation and architecture."""
        self.assertIsInstance(self.model, DirectMLPModel)
        self.assertEqual(self.model.n_point_features, self.n_point_features)
        self.assertEqual(self.model.hidden_units, self.hidden_units)
        
    def test_model_forward_pass(self):
        """Test forward pass with dummy data."""
        dummy_input = tf.random.normal((self.batch_size, self.n_point_features))
        output = self.model(dummy_input)
        
        # Check output shape and positive values (softplus activation)
        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertTrue(tf.reduce_all(output >= 0))  # Softplus ensures positive
        
    def test_model_config(self):
        """Test model configuration serialization."""
        config = self.model.get_config()
        self.assertIn('n_point_features', config)
        self.assertIn('hidden_units', config)
        self.assertIn('dropout_rate', config)


class TestResidualMLPModel(unittest.TestCase):
    """Test cases for Residual MLP model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_point_features = 8
        self.batch_size = 32
        self.hidden_units = [128, 64, 32]
        self.model = ResidualMLPModel(
            n_point_features=self.n_point_features,
            hidden_units=self.hidden_units
        )
        
    def test_model_creation(self):
        """Test model creation and architecture."""
        self.assertIsInstance(self.model, ResidualMLPModel)
        self.assertEqual(self.model.n_point_features, self.n_point_features)
        
    def test_model_forward_pass(self):
        """Test forward pass with dummy data."""
        dummy_input = tf.random.normal((self.batch_size, self.n_point_features))
        output = self.model(dummy_input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
        
    def test_residual_output_range(self):
        """Test that residual outputs can be negative (linear activation)."""
        dummy_input = tf.random.normal((self.batch_size, self.n_point_features))
        output = self.model(dummy_input)
        
        # With random weights, should get both positive and negative values
        # This is a probabilistic test, but very likely to pass
        self.assertTrue(len(tf.unique(tf.sign(output))[0]) > 1)


class TestCNNOnlyModel(unittest.TestCase):
    """Test cases for CNN-only model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.patch_size = (9, 9)
        self.batch_size = 16
        self.filters = [32, 64, 128]
        self.model = CNNOnlyModel(
            patch_size=self.patch_size,
            filters=self.filters
        )
        
    def test_model_creation(self):
        """Test model creation and architecture."""
        self.assertIsInstance(self.model, CNNOnlyModel)
        self.assertEqual(self.model.patch_size, self.patch_size)
        self.assertEqual(self.model.filters, self.filters)
        
    def test_model_forward_pass(self):
        """Test forward pass with dummy patch data."""
        # Create dummy patch data
        dummy_patches = tf.random.normal((self.batch_size, *self.patch_size, 1))
        output = self.model(dummy_patches)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
        
    def test_conv_layers(self):
        """Test CNN layers are created correctly."""
        # Build model with dummy input
        dummy_patches = tf.random.normal((1, *self.patch_size, 1))
        _ = self.model(dummy_patches)
        
        # Check that we have the expected number of conv layers
        conv_layers = [layer for layer in self.model.layers if 'conv' in layer.name]
        self.assertEqual(len(conv_layers), len(self.filters))


class TestFactoryFunctions(unittest.TestCase):
    """Test cases for model factory functions."""
    
    def test_create_funahashi_baseline(self):
        """Test Funahashi baseline factory function."""
        model = create_funahashi_baseline(n_point_features=8)
        
        self.assertIsInstance(model, FunahashiBaseline)
        self.assertIsNotNone(model.optimizer)
        self.assertEqual(model.loss, 'mse')
        
    def test_create_direct_mlp_model(self):
        """Test direct MLP factory function."""
        model = create_direct_mlp_model(n_point_features=7)
        
        self.assertIsInstance(model, DirectMLPModel)
        self.assertIsNotNone(model.optimizer)
        
    def test_create_residual_mlp_model(self):
        """Test residual MLP factory function."""
        model = create_residual_mlp_model(n_point_features=8)
        
        self.assertIsInstance(model, ResidualMLPModel)
        self.assertIsNotNone(model.optimizer)
        
    def test_create_cnn_only_model(self):
        """Test CNN-only factory function."""
        model = create_cnn_only_model(patch_size=(9, 9))
        
        self.assertIsInstance(model, CNNOnlyModel)
        self.assertIsNotNone(model.optimizer)
        
    def test_get_all_baseline_models(self):
        """Test function that creates all baseline models."""
        models = get_all_baseline_models(n_point_features=8, patch_size=(9, 9))
        
        expected_models = ['funahashi_baseline', 'direct_mlp', 'residual_mlp', 'cnn_only']
        self.assertEqual(set(models.keys()), set(expected_models))
        
        # Check each model is properly compiled
        for model_name, model in models.items():
            self.assertIsNotNone(model.optimizer)
            self.assertIsNotNone(model.loss)


class TestBaselineModelInterface(unittest.TestCase):
    """Test cases for baseline model interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = create_funahashi_baseline(n_point_features=8)
        self.interface = BaselineModelInterface(self.model, 'funahashi_baseline')
        
    def test_interface_creation(self):
        """Test interface creation."""
        self.assertEqual(self.interface.model_type, 'funahashi_baseline')
        self.assertIsNone(self.interface.history)
        
    def test_prediction_interface(self):
        """Test prediction through interface."""
        dummy_input = np.random.randn(10, 8)
        predictions = self.interface.predict(dummy_input)
        
        self.assertEqual(predictions.shape, (10, 1))
        
    def test_parameter_counting(self):
        """Test parameter counting."""
        param_count = self.interface.count_parameters()
        self.assertIsInstance(param_count, int)
        self.assertGreater(param_count, 0)
        
    def test_weight_save_load(self):
        """Test weight saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            weight_path = os.path.join(temp_dir, 'test_weights.h5')
            
            # Make a prediction to initialize weights
            dummy_input = np.random.randn(1, 8)
            original_pred = self.interface.predict(dummy_input)
            
            # Save weights
            self.interface.save_weights(weight_path)
            self.assertTrue(os.path.exists(weight_path))
            
            # Modify weights (reinitialize)
            self.interface.model.build((None, 8))
            
            # Load weights back
            self.interface.load_weights(weight_path)
            
            # Prediction should be the same
            loaded_pred = self.interface.predict(dummy_input)
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)


class TestCreateBaselineInterface(unittest.TestCase):
    """Test cases for baseline interface factory function."""
    
    def test_create_funahashi_interface(self):
        """Test creating Funahashi interface."""
        interface = create_baseline_interface('funahashi_baseline', n_point_features=8)
        
        self.assertIsInstance(interface, BaselineModelInterface)
        self.assertEqual(interface.model_type, 'funahashi_baseline')
        self.assertIsInstance(interface.model, FunahashiBaseline)
        
    def test_create_direct_mlp_interface(self):
        """Test creating direct MLP interface."""
        interface = create_baseline_interface('direct_mlp', n_point_features=7)
        
        self.assertIsInstance(interface, BaselineModelInterface)
        self.assertEqual(interface.model_type, 'direct_mlp')
        self.assertIsInstance(interface.model, DirectMLPModel)
        
    def test_create_residual_mlp_interface(self):
        """Test creating residual MLP interface."""
        interface = create_baseline_interface('residual_mlp', n_point_features=8)
        
        self.assertIsInstance(interface, BaselineModelInterface)
        self.assertEqual(interface.model_type, 'residual_mlp')
        self.assertIsInstance(interface.model, ResidualMLPModel)
        
    def test_create_cnn_only_interface(self):
        """Test creating CNN-only interface."""
        interface = create_baseline_interface('cnn_only', patch_size=(9, 9))
        
        self.assertIsInstance(interface, BaselineModelInterface)
        self.assertEqual(interface.model_type, 'cnn_only')
        self.assertIsInstance(interface.model, CNNOnlyModel)
        
    def test_invalid_model_name(self):
        """Test error handling for invalid model names."""
        with self.assertRaises(ValueError):
            create_baseline_interface('invalid_model')


class TestModelCompatibility(unittest.TestCase):
    """Test cases for model compatibility and integration."""
    
    def test_all_models_same_interface(self):
        """Test that all models work with the same interface."""
        models = get_all_baseline_models(n_point_features=8, patch_size=(9, 9))
        
        # Test that all models can make predictions
        point_features = np.random.randn(5, 8)
        patches = np.random.randn(5, 9, 9, 1)
        
        for model_name, model in models.items():
            if model_name == 'cnn_only':
                # CNN-only model takes patches
                predictions = model.predict(patches, verbose=0)
            elif model_name == 'direct_mlp':
                # Direct MLP doesn't use Hagan vol
                predictions = model.predict(point_features[:, :-1], verbose=0)
            else:
                # Other models use full point features
                predictions = model.predict(point_features, verbose=0)
                
            self.assertEqual(predictions.shape, (5, 1))
            
    def test_residual_vs_absolute_predictions(self):
        """Test difference between residual and absolute prediction models."""
        # Create models
        direct_mlp = create_direct_mlp_model(n_point_features=7)
        residual_mlp = create_residual_mlp_model(n_point_features=8)
        
        # Create test data
        point_features_no_hagan = np.random.randn(10, 7)
        hagan_vols = np.random.uniform(0.1, 0.5, (10, 1))  # Realistic vol range
        point_features_with_hagan = np.concatenate([point_features_no_hagan, hagan_vols], axis=1)
        
        # Get predictions
        absolute_preds = direct_mlp.predict(point_features_no_hagan, verbose=0)
        residual_preds = residual_mlp.predict(point_features_with_hagan, verbose=0)
        
        # Absolute predictions should be positive (volatilities)
        self.assertTrue(np.all(absolute_preds > 0))
        
        # Residual predictions can be negative
        # (This is probabilistic but very likely with random weights)
        self.assertTrue(len(np.unique(np.sign(residual_preds))) > 1)


if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run tests
    unittest.main(verbosity=2)