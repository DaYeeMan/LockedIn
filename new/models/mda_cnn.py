"""
Multi-fidelity Data Aggregation CNN (MDA-CNN) for SABR volatility surface modeling.

This module implements the MDA-CNN architecture that combines:
1. CNN branch for processing local LF surface patches
2. MLP branch for processing point features (SABR parameters, strike, maturity)
3. Fusion layer to combine representations
4. Residual prediction head for D(ξ) = σ_MC(ξ) - σ_Hagan(ξ)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


class CNNBranch(keras.Model):
    """CNN branch for processing local LF surface patches."""
    
    def __init__(self, patch_size: Tuple[int, int] = (9, 9), name: str = "cnn_branch"):
        super(CNNBranch, self).__init__(name=name)
        
        self.patch_size = patch_size
        
        # CNN layers for patch processing
        self.conv1 = layers.Conv2D(32, 3, activation='relu', padding='same', name='conv1')
        self.conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', name='conv2')
        self.conv3 = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv3')
        
        # Global pooling to reduce spatial dimensions
        self.global_pool = layers.GlobalAveragePooling2D(name='global_pool')
        
        # Dense layer for feature extraction
        self.dense = layers.Dense(128, activation='relu', name='cnn_dense')
        
    def call(self, inputs, training=None):
        """Forward pass through CNN branch.
        
        Args:
            inputs: Tensor of shape (batch_size, patch_height, patch_width, 1)
            training: Boolean indicating training mode
            
        Returns:
            Tensor of shape (batch_size, 128) - CNN features
        """
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = self.dense(x)
        return x


class MLPBranch(keras.Model):
    """MLP branch for processing point features."""
    
    def __init__(self, n_point_features: int = 8, name: str = "mlp_branch"):
        super(MLPBranch, self).__init__(name=name)
        
        self.n_point_features = n_point_features
        
        # MLP layers for point feature processing
        self.dense1 = layers.Dense(64, activation='relu', name='mlp_dense1')
        self.dense2 = layers.Dense(64, activation='relu', name='mlp_dense2')
        
    def call(self, inputs, training=None):
        """Forward pass through MLP branch.
        
        Args:
            inputs: Tensor of shape (batch_size, n_point_features)
            training: Boolean indicating training mode
            
        Returns:
            Tensor of shape (batch_size, 64) - MLP features
        """
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x


class FusionHead(keras.Model):
    """Fusion layer to combine CNN and MLP representations."""
    
    def __init__(self, dropout_rate: float = 0.2, name: str = "fusion_head"):
        super(FusionHead, self).__init__(name=name)
        
        self.dropout_rate = dropout_rate
        
        # Fusion layers
        self.fusion_dense1 = layers.Dense(128, activation='relu', name='fusion_dense1')
        self.dropout = layers.Dropout(dropout_rate, name='fusion_dropout')
        self.fusion_dense2 = layers.Dense(64, activation='relu', name='fusion_dense2')
        
        # Residual prediction head with linear activation
        self.residual_head = layers.Dense(1, activation='linear', name='residual_output')
        
    def call(self, inputs, training=None):
        """Forward pass through fusion head.
        
        Args:
            inputs: Tensor of shape (batch_size, cnn_features + mlp_features)
            training: Boolean indicating training mode
            
        Returns:
            Tensor of shape (batch_size, 1) - Residual predictions
        """
        x = self.fusion_dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.fusion_dense2(x)
        x = self.residual_head(x)
        return x


class MDACNN(keras.Model):
    """
    Multi-fidelity Data Aggregation CNN for SABR volatility surface modeling.
    
    This model combines CNN processing of local surface patches with MLP processing
    of point features to predict residuals between MC and Hagan surfaces.
    """
    
    def __init__(self, 
                 patch_size: Tuple[int, int] = (9, 9),
                 n_point_features: int = 8,
                 dropout_rate: float = 0.2,
                 name: str = "mda_cnn"):
        super(MDACNN, self).__init__(name=name)
        
        self.patch_size = patch_size
        self.n_point_features = n_point_features
        self.dropout_rate = dropout_rate
        
        # Initialize branches
        self.cnn_branch = CNNBranch(patch_size=patch_size)
        self.mlp_branch = MLPBranch(n_point_features=n_point_features)
        self.fusion_head = FusionHead(dropout_rate=dropout_rate)
        
        # Concatenation layer
        self.concat = layers.Concatenate(axis=-1, name='feature_concat')
        
    def call(self, inputs, training=None):
        """Forward pass through MDA-CNN.
        
        Args:
            inputs: Dictionary or tuple containing:
                - 'patches' or inputs[0]: Tensor of shape (batch_size, patch_height, patch_width, 1)
                - 'features' or inputs[1]: Tensor of shape (batch_size, n_point_features)
            training: Boolean indicating training mode
            
        Returns:
            Tensor of shape (batch_size, 1) - Residual predictions D(ξ) = σ_MC(ξ) - σ_Hagan(ξ)
        """
        if isinstance(inputs, dict):
            patches = inputs['patches']
            features = inputs['features']
        else:
            patches, features = inputs
            
        # Process through branches
        cnn_features = self.cnn_branch(patches, training=training)
        mlp_features = self.mlp_branch(features, training=training)
        
        # Concatenate features
        combined_features = self.concat([cnn_features, mlp_features])
        
        # Generate residual predictions
        residuals = self.fusion_head(combined_features, training=training)
        
        return residuals
    
    def get_config(self):
        """Get model configuration for serialization."""
        config = super(MDACNN, self).get_config()
        config.update({
            'patch_size': self.patch_size,
            'n_point_features': self.n_point_features,
            'dropout_rate': self.dropout_rate
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)


def create_mda_cnn_model(patch_size: Tuple[int, int] = (9, 9),
                        n_point_features: int = 8,
                        dropout_rate: float = 0.2,
                        learning_rate: float = 3e-4) -> MDACNN:
    """
    Factory function to create and compile MDA-CNN model.
    
    Args:
        patch_size: Size of input patches (height, width)
        n_point_features: Number of point features
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled MDA-CNN model
    """
    model = MDACNN(
        patch_size=patch_size,
        n_point_features=n_point_features,
        dropout_rate=dropout_rate
    )
    
    # Compile model with appropriate loss and metrics
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


def build_model_with_inputs(patch_size: Tuple[int, int] = (9, 9),
                           n_point_features: int = 8,
                           dropout_rate: float = 0.2) -> keras.Model:
    """
    Build MDA-CNN model using functional API for explicit input definition.
    
    This is useful when you need explicit input layers for model visualization
    or when working with tf.data pipelines.
    
    Args:
        patch_size: Size of input patches (height, width)
        n_point_features: Number of point features
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Keras Model with explicit inputs
    """
    # Define inputs
    patch_input = keras.Input(shape=(*patch_size, 1), name='patches')
    feature_input = keras.Input(shape=(n_point_features,), name='features')
    
    # Create MDA-CNN instance
    mda_cnn = MDACNN(
        patch_size=patch_size,
        n_point_features=n_point_features,
        dropout_rate=dropout_rate
    )
    
    # Get outputs
    outputs = mda_cnn([patch_input, feature_input])
    
    # Create functional model
    model = keras.Model(inputs=[patch_input, feature_input], outputs=outputs, name='mda_cnn_functional')
    
    return model