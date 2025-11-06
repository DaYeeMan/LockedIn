"""
Utility functions and common components for SABR volatility surface models.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, List, Optional, Dict, Any


def get_model_summary_string(model: keras.Model) -> str:
    """
    Get model summary as a string for logging.
    
    Args:
        model: Keras model
        
    Returns:
        String representation of model summary
    """
    import io
    import contextlib
    
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        model.summary()
    return f.getvalue()


def count_trainable_parameters(model: keras.Model) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: Keras model
        
    Returns:
        Number of trainable parameters
    """
    return sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])


def create_residual_block(filters: int, 
                         kernel_size: int = 3,
                         activation: str = 'relu',
                         name: str = None) -> keras.Sequential:
    """
    Create a residual block for CNN architectures.
    
    Args:
        filters: Number of filters
        kernel_size: Convolution kernel size
        activation: Activation function
        name: Block name
        
    Returns:
        Sequential model representing residual block
    """
    block = keras.Sequential(name=name)
    block.add(layers.Conv2D(filters, kernel_size, padding='same', activation=activation))
    block.add(layers.BatchNormalization())
    block.add(layers.Conv2D(filters, kernel_size, padding='same', activation=None))
    block.add(layers.BatchNormalization())
    
    return block


class ResidualBlock(layers.Layer):
    """
    Residual block layer with skip connections.
    """
    
    def __init__(self, filters: int, kernel_size: int = 3, activation: str = 'relu', **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same', activation=activation)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same', activation=None)
        self.bn2 = layers.BatchNormalization()
        self.add = layers.Add()
        self.final_activation = layers.Activation(activation)
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Skip connection
        x = self.add([x, inputs])
        x = self.final_activation(x)
        
        return x
    
    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': self.activation
        })
        return config


def create_mlp_block(units: List[int],
                    activation: str = 'relu',
                    dropout_rate: float = 0.0,
                    batch_norm: bool = False,
                    name: str = None) -> keras.Sequential:
    """
    Create an MLP block with specified architecture.
    
    Args:
        units: List of units for each dense layer
        activation: Activation function
        dropout_rate: Dropout rate (0 means no dropout)
        batch_norm: Whether to use batch normalization
        name: Block name
        
    Returns:
        Sequential model representing MLP block
    """
    block = keras.Sequential(name=name)
    
    for i, unit in enumerate(units):
        block.add(layers.Dense(unit, activation=activation, name=f'dense_{i}'))
        
        if batch_norm:
            block.add(layers.BatchNormalization(name=f'bn_{i}'))
            
        if dropout_rate > 0:
            block.add(layers.Dropout(dropout_rate, name=f'dropout_{i}'))
    
    return block


def validate_input_shapes(patches: tf.Tensor, 
                         features: tf.Tensor,
                         expected_patch_shape: Tuple[int, int],
                         expected_feature_dim: int) -> None:
    """
    Validate input tensor shapes for MDA-CNN.
    
    Args:
        patches: Patch tensor
        features: Feature tensor
        expected_patch_shape: Expected patch shape (height, width)
        expected_feature_dim: Expected feature dimension
        
    Raises:
        ValueError: If shapes don't match expectations
    """
    patch_shape = patches.shape
    feature_shape = features.shape
    
    if len(patch_shape) != 4:
        raise ValueError(f"Patches should be 4D (batch, height, width, channels), got shape {patch_shape}")
    
    if patch_shape[1:3] != expected_patch_shape:
        raise ValueError(f"Expected patch shape {expected_patch_shape}, got {patch_shape[1:3]}")
    
    if len(feature_shape) != 2:
        raise ValueError(f"Features should be 2D (batch, features), got shape {feature_shape}")
    
    if feature_shape[1] != expected_feature_dim:
        raise ValueError(f"Expected {expected_feature_dim} features, got {feature_shape[1]}")


def create_custom_metrics() -> Dict[str, keras.metrics.Metric]:
    """
    Create custom metrics for SABR volatility surface evaluation.
    
    Returns:
        Dictionary of custom metrics
    """
    metrics = {}
    
    # Relative percentage error
    def relative_percentage_error(y_true, y_pred):
        """Relative percentage error: |y_true - y_pred| / |y_true| * 100"""
        return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + 1e-8)) * 100)
    
    # Root mean squared error
    def rmse(y_true, y_pred):
        """Root mean squared error"""
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    
    metrics['relative_percentage_error'] = relative_percentage_error
    metrics['rmse'] = rmse
    
    return metrics


class ModelCheckpointCallback(keras.callbacks.Callback):
    """
    Custom callback for model checkpointing with additional logging.
    """
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', save_best_only: bool = True):
        super(ModelCheckpointCallback, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_value = float('inf') if 'loss' in monitor else float('-inf')
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            return
        
        # Check if this is the best model so far
        is_better = (current_value < self.best_value if 'loss' in self.monitor 
                    else current_value > self.best_value)
        
        if not self.save_best_only or is_better:
            if is_better:
                self.best_value = current_value
                
            self.model.save_weights(self.filepath)
            print(f"Epoch {epoch + 1}: {self.monitor} = {current_value:.6f}, saved model to {self.filepath}")


def create_learning_rate_scheduler(initial_lr: float = 3e-4,
                                 decay_steps: int = 1000,
                                 decay_rate: float = 0.9) -> keras.optimizers.schedules.LearningRateSchedule:
    """
    Create exponential decay learning rate scheduler.
    
    Args:
        initial_lr: Initial learning rate
        decay_steps: Steps between decay
        decay_rate: Decay rate
        
    Returns:
        Learning rate schedule
    """
    return keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )


def get_model_flops(model: keras.Model, input_shapes: List[Tuple]) -> int:
    """
    Estimate FLOPs (Floating Point Operations) for a model.
    
    Note: This is a rough estimation and may not be perfectly accurate.
    
    Args:
        model: Keras model
        input_shapes: List of input shapes (without batch dimension)
        
    Returns:
        Estimated number of FLOPs
    """
    # This is a simplified FLOP estimation
    # For more accurate measurements, consider using tf.profiler
    
    total_flops = 0
    
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            # Dense layer: input_dim * output_dim * 2 (multiply + add)
            total_flops += layer.input_shape[-1] * layer.units * 2
            
        elif isinstance(layer, layers.Conv2D):
            # Conv2D: kernel_size * kernel_size * input_channels * output_channels * output_height * output_width * 2
            kernel_size = layer.kernel_size[0] * layer.kernel_size[1]
            input_channels = layer.input_shape[-1]
            output_channels = layer.filters
            # Approximate output spatial dimensions (depends on padding/stride)
            output_spatial = layer.input_shape[1] * layer.input_shape[2]  # Simplified
            total_flops += kernel_size * input_channels * output_channels * output_spatial * 2
    
    return total_flops