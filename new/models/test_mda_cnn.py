"""
Unit tests for MDA-CNN model architecture components.
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from models.mda_cnn import (
    MDACNN, CNNBranch, MLPBranch, FusionHead,
    create_mda_cnn_model, build_model_with_inputs
)
from models.model_utils import (
    count_trainable_parameters, validate_input_shapes,
    create_custom_metrics, ResidualBlock
)
from models.loss_functions import (
    WeightedMSELoss, HuberLoss, RelativePercentageErrorLoss,
    CombinedLoss, relative_percentage_error_metric, rmse_metric
)


class TestCNNBranch:
    """Test cases for CNN branch component."""
    
    def test_cnn_branch_initialization(self):
        """Test CNN branch initialization with default parameters."""
        cnn_branch = CNNBranch()
        
        assert cnn_branch.patch_size == (9, 9)
        assert cnn_branch.name == "cnn_branch"
        
    def test_cnn_branch_forward_pass(self):
        """Test CNN branch forward pass with sample data."""
        cnn_branch = CNNBranch(patch_size=(9, 9))
        
        # Create sample input: batch_size=4, height=9, width=9, channels=1
        sample_input = tf.random.normal((4, 9, 9, 1))
        
        output = cnn_branch(sample_input)
        
        # Output should be (batch_size, 128)
        assert output.shape == (4, 128)
        assert output.dtype == tf.float32


class TestMLPBranch:
    """Test cases for MLP branch component."""
    
    def test_mlp_branch_initialization(self):
        """Test MLP branch initialization with default parameters."""
        mlp_branch = MLPBranch()
        
        assert mlp_branch.n_point_features == 8
        assert mlp_branch.name == "mlp_branch"
        
    def test_mlp_branch_forward_pass(self):
        """Test MLP branch forward pass with sample data."""
        mlp_branch = MLPBranch(n_point_features=8)
        
        # Create sample input: batch_size=4, features=8
        sample_input = tf.random.normal((4, 8))
        
        output = mlp_branch(sample_input)
        
        # Output should be (batch_size, 64)
        assert output.shape == (4, 64)
        assert output.dtype == tf.float32


class TestFusionHead:
    """Test cases for fusion head component."""
    
    def test_fusion_head_forward_pass(self):
        """Test fusion head forward pass with sample data."""
        fusion_head = FusionHead(dropout_rate=0.1)
        
        # Create sample input: batch_size=4, combined_features=192 (128+64)
        sample_input = tf.random.normal((4, 192))
        
        output = fusion_head(sample_input, training=True)
        
        # Output should be (batch_size, 1) for residual prediction
        assert output.shape == (4, 1)
        assert output.dtype == tf.float32


class TestMDACNN:
    """Test cases for complete MDA-CNN model."""
    
    def test_mdacnn_initialization(self):
        """Test MDA-CNN initialization with default parameters."""
        model = MDACNN()
        
        assert model.patch_size == (9, 9)
        assert model.n_point_features == 8
        assert model.dropout_rate == 0.2
        assert model.name == "mda_cnn"
        
    def test_mdacnn_forward_pass_dict_input(self):
        """Test MDA-CNN forward pass with dictionary input."""
        model = MDACNN(patch_size=(9, 9), n_point_features=8)
        
        # Create sample inputs
        patches = tf.random.normal((4, 9, 9, 1))
        features = tf.random.normal((4, 8))
        
        inputs = {'patches': patches, 'features': features}
        output = model(inputs, training=True)
        
        assert output.shape == (4, 1)
        assert output.dtype == tf.float32


class TestModelFactoryFunctions:
    """Test cases for model factory functions."""
    
    def test_create_mda_cnn_model(self):
        """Test create_mda_cnn_model factory function."""
        model = create_mda_cnn_model(
            patch_size=(9, 9),
            n_point_features=8,
            dropout_rate=0.2,
            learning_rate=1e-3
        )
        
        assert isinstance(model, MDACNN)
        assert model.patch_size == (9, 9)
        assert model.n_point_features == 8
        
        # Check that model is compiled
        assert model.optimizer is not None
        assert model.loss is not None


class TestLossFunctions:
    """Test cases for custom loss functions."""
    
    def test_weighted_mse_loss(self):
        """Test WeightedMSELoss functionality."""
        loss_fn = WeightedMSELoss(wing_weight=2.0, atm_threshold=0.1)
        
        # Create sample data with some wing regions (high residuals)
        y_true = tf.constant([0.05, 0.15, -0.2, 0.08])  # Mix of ATM and wing
        y_pred = tf.constant([0.04, 0.12, -0.18, 0.09])
        
        loss = loss_fn(y_true, y_pred)
        
        assert tf.is_tensor(loss)
        assert loss.numpy() > 0
        
    def test_custom_metrics(self):
        """Test custom metric functions."""
        y_true = tf.constant([1.0, 2.0, 3.0])
        y_pred = tf.constant([1.05, 1.9, 3.1])
        
        rpe = relative_percentage_error_metric(y_true, y_pred)
        rmse = rmse_metric(y_true, y_pred)
        
        assert tf.is_tensor(rpe)
        assert tf.is_tensor(rmse)
        assert rpe.numpy() > 0
        assert rmse.numpy() > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])