"""
Custom loss functions for SABR volatility surface modeling.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Optional, Callable


class WeightedMSELoss(keras.losses.Loss):
    """
    Weighted Mean Squared Error loss for emphasizing wing regions.
    
    This loss function applies higher weights to points in the wing regions
    where MC-Hagan residuals are typically larger and more important for
    accurate volatility surface modeling.
    """
    
    def __init__(self, 
                 wing_weight: float = 2.0,
                 atm_threshold: float = 0.1,
                 name: str = "weighted_mse",
                 **kwargs):
        super(WeightedMSELoss, self).__init__(name=name, **kwargs)
        self.wing_weight = wing_weight
        self.atm_threshold = atm_threshold
        
    def call(self, y_true, y_pred):
        """
        Compute weighted MSE loss.
        
        Args:
            y_true: True residual values
            y_pred: Predicted residual values
            
        Returns:
            Weighted MSE loss
        """
        # Compute squared errors
        squared_errors = tf.square(y_true - y_pred)
        
        # Create weights based on residual magnitude
        # Higher weights for larger residuals (wing regions)
        residual_magnitude = tf.abs(y_true)
        weights = tf.where(
            residual_magnitude > self.atm_threshold,
            self.wing_weight,
            1.0
        )
        
        # Apply weights
        weighted_errors = weights * squared_errors
        
        return tf.reduce_mean(weighted_errors)
    
    def get_config(self):
        config = super(WeightedMSELoss, self).get_config()
        config.update({
            'wing_weight': self.wing_weight,
            'atm_threshold': self.atm_threshold
        })
        return config


class HuberLoss(keras.losses.Loss):
    """
    Huber loss for robust training against outliers.
    
    Combines MSE for small errors and MAE for large errors,
    making training more robust to outliers in the data.
    """
    
    def __init__(self, delta: float = 1.0, name: str = "huber_loss", **kwargs):
        super(HuberLoss, self).__init__(name=name, **kwargs)
        self.delta = delta
        
    def call(self, y_true, y_pred):
        """
        Compute Huber loss.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Huber loss
        """
        error = y_true - y_pred
        abs_error = tf.abs(error)
        
        # Use MSE for small errors, MAE for large errors
        quadratic = tf.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        
        loss = 0.5 * quadratic**2 + self.delta * linear
        
        return tf.reduce_mean(loss)
    
    def get_config(self):
        config = super(HuberLoss, self).get_config()
        config.update({'delta': self.delta})
        return config


class RelativePercentageErrorLoss(keras.losses.Loss):
    """
    Relative percentage error loss for volatility surface modeling.
    
    This loss function computes the relative percentage error,
    which is commonly used in financial modeling for comparing
    model performance across different volatility levels.
    """
    
    def __init__(self, epsilon: float = 1e-8, name: str = "relative_percentage_error", **kwargs):
        super(RelativePercentageErrorLoss, self).__init__(name=name, **kwargs)
        self.epsilon = epsilon
        
    def call(self, y_true, y_pred):
        """
        Compute relative percentage error loss.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Relative percentage error loss
        """
        # Avoid division by zero
        denominator = tf.maximum(tf.abs(y_true), self.epsilon)
        relative_error = tf.abs(y_true - y_pred) / denominator
        
        return tf.reduce_mean(relative_error * 100)
    
    def get_config(self):
        config = super(RelativePercentageErrorLoss, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config


class CombinedLoss(keras.losses.Loss):
    """
    Combined loss function that weights multiple loss components.
    
    This allows combining different loss functions (e.g., MSE + relative error)
    with different weights to balance different aspects of model performance.
    """
    
    def __init__(self, 
                 loss_functions: list,
                 loss_weights: list,
                 name: str = "combined_loss",
                 **kwargs):
        super(CombinedLoss, self).__init__(name=name, **kwargs)
        
        if len(loss_functions) != len(loss_weights):
            raise ValueError("Number of loss functions must match number of weights")
            
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights
        
    def call(self, y_true, y_pred):
        """
        Compute combined loss.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Weighted combination of losses
        """
        total_loss = 0.0
        
        for loss_fn, weight in zip(self.loss_functions, self.loss_weights):
            loss_value = loss_fn(y_true, y_pred)
            total_loss += weight * loss_value
            
        return total_loss
    
    def get_config(self):
        config = super(CombinedLoss, self).get_config()
        config.update({
            'loss_functions': [loss_fn.get_config() for loss_fn in self.loss_functions],
            'loss_weights': self.loss_weights
        })
        return config


def create_wing_weighted_mse(wing_weight: float = 2.0, 
                           atm_threshold: float = 0.1) -> WeightedMSELoss:
    """
    Factory function to create weighted MSE loss for wing emphasis.
    
    Args:
        wing_weight: Weight multiplier for wing regions
        atm_threshold: Threshold for determining wing vs ATM regions
        
    Returns:
        WeightedMSELoss instance
    """
    return WeightedMSELoss(wing_weight=wing_weight, atm_threshold=atm_threshold)


def create_robust_huber_loss(delta: float = 1.0) -> HuberLoss:
    """
    Factory function to create Huber loss for robust training.
    
    Args:
        delta: Threshold for switching between MSE and MAE
        
    Returns:
        HuberLoss instance
    """
    return HuberLoss(delta=delta)


def create_combined_mse_rpe_loss(mse_weight: float = 0.7, 
                                rpe_weight: float = 0.3) -> CombinedLoss:
    """
    Factory function to create combined MSE + Relative Percentage Error loss.
    
    Args:
        mse_weight: Weight for MSE component
        rpe_weight: Weight for RPE component
        
    Returns:
        CombinedLoss instance
    """
    mse_loss = keras.losses.MeanSquaredError()
    rpe_loss = RelativePercentageErrorLoss()
    
    return CombinedLoss(
        loss_functions=[mse_loss, rpe_loss],
        loss_weights=[mse_weight, rpe_weight]
    )


# Custom metrics for evaluation
def relative_percentage_error_metric(y_true, y_pred):
    """
    Relative percentage error metric for model evaluation.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Relative percentage error
    """
    epsilon = 1e-8
    denominator = tf.maximum(tf.abs(y_true), epsilon)
    relative_error = tf.abs(y_true - y_pred) / denominator
    return tf.reduce_mean(relative_error * 100)


def rmse_metric(y_true, y_pred):
    """
    Root Mean Squared Error metric.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def wing_region_mse(y_true, y_pred, threshold: float = 0.1):
    """
    MSE computed only on wing regions (high residual magnitude).
    
    Args:
        y_true: True residual values
        y_pred: Predicted residual values
        threshold: Threshold for wing region classification
        
    Returns:
        MSE for wing regions only
    """
    # Identify wing regions based on residual magnitude
    wing_mask = tf.abs(y_true) > threshold
    
    # Compute MSE only for wing regions
    wing_true = tf.boolean_mask(y_true, wing_mask)
    wing_pred = tf.boolean_mask(y_pred, wing_mask)
    
    # Handle case where no wing points exist
    wing_mse = tf.cond(
        tf.size(wing_true) > 0,
        lambda: tf.reduce_mean(tf.square(wing_true - wing_pred)),
        lambda: 0.0
    )
    
    return wing_mse


def atm_region_mse(y_true, y_pred, threshold: float = 0.1):
    """
    MSE computed only on ATM regions (low residual magnitude).
    
    Args:
        y_true: True residual values
        y_pred: Predicted residual values
        threshold: Threshold for ATM region classification
        
    Returns:
        MSE for ATM regions only
    """
    # Identify ATM regions based on residual magnitude
    atm_mask = tf.abs(y_true) <= threshold
    
    # Compute MSE only for ATM regions
    atm_true = tf.boolean_mask(y_true, atm_mask)
    atm_pred = tf.boolean_mask(y_pred, atm_mask)
    
    # Handle case where no ATM points exist
    atm_mse = tf.cond(
        tf.size(atm_true) > 0,
        lambda: tf.reduce_mean(tf.square(atm_true - atm_pred)),
        lambda: 0.0
    )
    
    return atm_mse