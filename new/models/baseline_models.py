"""
Baseline models for SABR volatility surface modeling comparison.

This module implements various baseline models to compare against the MDA-CNN:
1. Funahashi baseline model (exact 5-layer, 32 neurons, ReLU, residual learning)
2. Direct MLP model (point features → volatility)
3. Residual MLP model (point features → residual, no patches)
4. Simple CNN-only model for ablation studies
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional, List
# Note: model_utils functions not needed for baseline models


class FunahashiBaseline(keras.Model):
    """
    Exact implementation of Funahashi's baseline model.
    
    Architecture: 5 layers, 32 neurons each, ReLU activation, residual learning
    Input: Point features (SABR parameters, strike, maturity, Hagan volatility)
    Output: Residual D(ξ) = σ_MC(ξ) - σ_Hagan(ξ)
    """
    
    def __init__(self, n_point_features: int = 8, name: str = "funahashi_baseline"):
        super(FunahashiBaseline, self).__init__(name=name)
        
        self.n_point_features = n_point_features
        
        # Exact Funahashi architecture: 5 layers, 32 neurons, ReLU
        self.dense1 = layers.Dense(32, activation='relu', name='funahashi_dense1')
        self.dense2 = layers.Dense(32, activation='relu', name='funahashi_dense2')
        self.dense3 = layers.Dense(32, activation='relu', name='funahashi_dense3')
        self.dense4 = layers.Dense(32, activation='relu', name='funahashi_dense4')
        self.dense5 = layers.Dense(32, activation='relu', name='funahashi_dense5')
        
        # Residual output layer (linear activation for residual learning)
        self.residual_output = layers.Dense(1, activation='linear', name='residual_output')
        
    def call(self, inputs, training=None):
        """
        Forward pass through Funahashi baseline model.
        
        Args:
            inputs: Tensor of shape (batch_size, n_point_features)
            training: Boolean indicating training mode
            
        Returns:
            Tensor of shape (batch_size, 1) - Residual predictions
        """
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.residual_output(x)
        return x
    
    def get_config(self):
        """Get model configuration for serialization."""
        config = super(FunahashiBaseline, self).get_config()
        config.update({
            'n_point_features': self.n_point_features
        })
        return config


class DirectMLPModel(keras.Model):
    """
    Direct MLP model that predicts absolute volatility values.
    
    Input: Point features (SABR parameters, strike, maturity)
    Output: Absolute volatility σ_MC(ξ)
    """
    
    def __init__(self, 
                 n_point_features: int = 7,  # No Hagan vol in input
                 hidden_units: List[int] = [64, 64, 32],
                 dropout_rate: float = 0.1,
                 name: str = "direct_mlp"):
        super(DirectMLPModel, self).__init__(name=name)
        
        self.n_point_features = n_point_features
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        
        # Build MLP layers
        self.mlp_layers = []
        for i, units in enumerate(hidden_units):
            self.mlp_layers.append(
                layers.Dense(units, activation='relu', name=f'direct_dense_{i+1}')
            )
            if dropout_rate > 0:
                self.mlp_layers.append(
                    layers.Dropout(dropout_rate, name=f'direct_dropout_{i+1}')
                )
        
        # Output layer for absolute volatility (positive values)
        self.volatility_output = layers.Dense(1, activation='softplus', name='volatility_output')
        
    def call(self, inputs, training=None):
        """
        Forward pass through direct MLP model.
        
        Args:
            inputs: Tensor of shape (batch_size, n_point_features)
            training: Boolean indicating training mode
            
        Returns:
            Tensor of shape (batch_size, 1) - Absolute volatility predictions
        """
        x = inputs
        for layer in self.mlp_layers:
            x = layer(x, training=training)
        x = self.volatility_output(x)
        return x
    
    def get_config(self):
        """Get model configuration for serialization."""
        config = super(DirectMLPModel, self).get_config()
        config.update({
            'n_point_features': self.n_point_features,
            'hidden_units': self.hidden_units,
            'dropout_rate': self.dropout_rate
        })
        return config


class ResidualMLPModel(keras.Model):
    """
    Residual MLP model that predicts residuals without using patches.
    
    Input: Point features (SABR parameters, strike, maturity, Hagan volatility)
    Output: Residual D(ξ) = σ_MC(ξ) - σ_Hagan(ξ)
    """
    
    def __init__(self, 
                 n_point_features: int = 8,
                 hidden_units: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 name: str = "residual_mlp"):
        super(ResidualMLPModel, self).__init__(name=name)
        
        self.n_point_features = n_point_features
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        
        # Build MLP layers
        self.mlp_layers = []
        for i, units in enumerate(hidden_units):
            self.mlp_layers.append(
                layers.Dense(units, activation='relu', name=f'residual_dense_{i+1}')
            )
            if dropout_rate > 0:
                self.mlp_layers.append(
                    layers.Dropout(dropout_rate, name=f'residual_dropout_{i+1}')
                )
        
        # Residual output layer (linear activation)
        self.residual_output = layers.Dense(1, activation='linear', name='residual_output')
        
    def call(self, inputs, training=None):
        """
        Forward pass through residual MLP model.
        
        Args:
            inputs: Tensor of shape (batch_size, n_point_features)
            training: Boolean indicating training mode
            
        Returns:
            Tensor of shape (batch_size, 1) - Residual predictions
        """
        x = inputs
        for layer in self.mlp_layers:
            x = layer(x, training=training)
        x = self.residual_output(x)
        return x
    
    def get_config(self):
        """Get model configuration for serialization."""
        config = super(ResidualMLPModel, self).get_config()
        config.update({
            'n_point_features': self.n_point_features,
            'hidden_units': self.hidden_units,
            'dropout_rate': self.dropout_rate
        })
        return config


class CNNOnlyModel(keras.Model):
    """
    CNN-only model for ablation studies.
    
    Input: Local surface patches only (no point features)
    Output: Residual D(ξ) = σ_MC(ξ) - σ_Hagan(ξ)
    """
    
    def __init__(self, 
                 patch_size: Tuple[int, int] = (9, 9),
                 filters: List[int] = [32, 64, 128],
                 dense_units: List[int] = [128, 64],
                 dropout_rate: float = 0.2,
                 name: str = "cnn_only"):
        super(CNNOnlyModel, self).__init__(name=name)
        
        self.patch_size = patch_size
        self.filters = filters
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        
        # CNN layers for patch processing
        self.conv_layers = []
        for i, filter_count in enumerate(filters):
            self.conv_layers.append(
                layers.Conv2D(filter_count, 3, activation='relu', padding='same', 
                            name=f'cnn_conv_{i+1}')
            )
        
        # Global pooling to reduce spatial dimensions
        self.global_pool = layers.GlobalAveragePooling2D(name='cnn_global_pool')
        
        # Dense layers
        self.dense_layers = []
        for i, units in enumerate(dense_units):
            self.dense_layers.append(
                layers.Dense(units, activation='relu', name=f'cnn_dense_{i+1}')
            )
            if dropout_rate > 0:
                self.dense_layers.append(
                    layers.Dropout(dropout_rate, name=f'cnn_dropout_{i+1}')
                )
        
        # Residual output layer
        self.residual_output = layers.Dense(1, activation='linear', name='residual_output')
        
    def call(self, inputs, training=None):
        """
        Forward pass through CNN-only model.
        
        Args:
            inputs: Tensor of shape (batch_size, patch_height, patch_width, 1)
            training: Boolean indicating training mode
            
        Returns:
            Tensor of shape (batch_size, 1) - Residual predictions
        """
        x = inputs
        
        # Process through CNN layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Process through dense layers
        for layer in self.dense_layers:
            x = layer(x, training=training)
        
        # Generate residual predictions
        x = self.residual_output(x)
        return x
    
    def get_config(self):
        """Get model configuration for serialization."""
        config = super(CNNOnlyModel, self).get_config()
        config.update({
            'patch_size': self.patch_size,
            'filters': self.filters,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate
        })
        return config


# Factory functions for creating baseline models

def create_funahashi_baseline(n_point_features: int = 8,
                            learning_rate: float = 3e-4) -> FunahashiBaseline:
    """
    Factory function to create and compile Funahashi baseline model.
    
    Args:
        n_point_features: Number of point features
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Funahashi baseline model
    """
    model = FunahashiBaseline(n_point_features=n_point_features)
    
    # Compile with MSE loss for residual learning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


def create_direct_mlp_model(n_point_features: int = 7,
                          hidden_units: List[int] = [64, 64, 32],
                          dropout_rate: float = 0.1,
                          learning_rate: float = 3e-4) -> DirectMLPModel:
    """
    Factory function to create and compile direct MLP model.
    
    Args:
        n_point_features: Number of point features (excluding Hagan vol)
        hidden_units: List of hidden layer units
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled direct MLP model
    """
    model = DirectMLPModel(
        n_point_features=n_point_features,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate
    )
    
    # Compile with MSE loss for absolute volatility prediction
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


def create_residual_mlp_model(n_point_features: int = 8,
                            hidden_units: List[int] = [128, 64, 32],
                            dropout_rate: float = 0.2,
                            learning_rate: float = 3e-4) -> ResidualMLPModel:
    """
    Factory function to create and compile residual MLP model.
    
    Args:
        n_point_features: Number of point features
        hidden_units: List of hidden layer units
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled residual MLP model
    """
    model = ResidualMLPModel(
        n_point_features=n_point_features,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate
    )
    
    # Compile with MSE loss for residual learning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


def create_cnn_only_model(patch_size: Tuple[int, int] = (9, 9),
                        filters: List[int] = [32, 64, 128],
                        dense_units: List[int] = [128, 64],
                        dropout_rate: float = 0.2,
                        learning_rate: float = 3e-4) -> CNNOnlyModel:
    """
    Factory function to create and compile CNN-only model.
    
    Args:
        patch_size: Size of input patches (height, width)
        filters: List of filter counts for CNN layers
        dense_units: List of dense layer units
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled CNN-only model
    """
    model = CNNOnlyModel(
        patch_size=patch_size,
        filters=filters,
        dense_units=dense_units,
        dropout_rate=dropout_rate
    )
    
    # Compile with MSE loss for residual learning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


def get_all_baseline_models(n_point_features: int = 8,
                          patch_size: Tuple[int, int] = (9, 9),
                          learning_rate: float = 3e-4) -> dict:
    """
    Create all baseline models for comparison studies.
    
    Args:
        n_point_features: Number of point features
        patch_size: Size of input patches
        learning_rate: Learning rate for all models
        
    Returns:
        Dictionary of compiled baseline models
    """
    models = {
        'funahashi_baseline': create_funahashi_baseline(
            n_point_features=n_point_features,
            learning_rate=learning_rate
        ),
        'direct_mlp': create_direct_mlp_model(
            n_point_features=n_point_features - 1,  # Exclude Hagan vol
            learning_rate=learning_rate
        ),
        'residual_mlp': create_residual_mlp_model(
            n_point_features=n_point_features,
            learning_rate=learning_rate
        ),
        'cnn_only': create_cnn_only_model(
            patch_size=patch_size,
            learning_rate=learning_rate
        )
    }
    
    return models


class BaselineModelInterface:
    """
    Consistent interface for all baseline models to ensure compatibility.
    
    This class provides a unified interface for training, evaluation, and
    prediction across all baseline model architectures.
    """
    
    def __init__(self, model: keras.Model, model_type: str):
        self.model = model
        self.model_type = model_type
        self.history = None
        
    def train(self, 
             train_data,
             validation_data=None,
             epochs: int = 100,
             batch_size: int = 64,
             callbacks: List = None,
             verbose: int = 1):
        """
        Train the baseline model.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        self.history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks or [],
            verbose=verbose
        )
        return self.history
    
    def evaluate(self, test_data, batch_size: int = 64, verbose: int = 0):
        """
        Evaluate the baseline model.
        
        Args:
            test_data: Test dataset
            batch_size: Batch size for evaluation
            verbose: Verbosity level
            
        Returns:
            Evaluation metrics
        """
        return self.model.evaluate(test_data, batch_size=batch_size, verbose=verbose)
    
    def predict(self, inputs, batch_size: int = 64):
        """
        Make predictions with the baseline model.
        
        Args:
            inputs: Input data
            batch_size: Batch size for prediction
            
        Returns:
            Model predictions
        """
        return self.model.predict(inputs, batch_size=batch_size)
    
    def save_weights(self, filepath: str):
        """Save model weights."""
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath: str):
        """Load model weights."""
        self.model.load_weights(filepath)
    
    def get_model_summary(self):
        """Get model summary."""
        return self.model.summary()
    
    def count_parameters(self):
        """Count trainable parameters."""
        return self.model.count_params()


def create_baseline_interface(model_name: str, **kwargs) -> BaselineModelInterface:
    """
    Factory function to create baseline model with consistent interface.
    
    Args:
        model_name: Name of the baseline model
        **kwargs: Model-specific arguments
        
    Returns:
        BaselineModelInterface instance
    """
    model_factories = {
        'funahashi_baseline': create_funahashi_baseline,
        'direct_mlp': create_direct_mlp_model,
        'residual_mlp': create_residual_mlp_model,
        'cnn_only': create_cnn_only_model
    }
    
    if model_name not in model_factories:
        raise ValueError(f"Unknown model name: {model_name}. "
                        f"Available models: {list(model_factories.keys())}")
    
    model = model_factories[model_name](**kwargs)
    return BaselineModelInterface(model, model_name)