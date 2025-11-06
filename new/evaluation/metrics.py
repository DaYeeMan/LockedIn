"""
Evaluation metrics for SABR volatility surface models.

This module implements Funahashi's exact metrics and additional evaluation
metrics for comparing MDA-CNN against baseline models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import tensorflow as tf
from dataclasses import dataclass


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    mse: float
    rmse: float
    mae: float
    relative_percentage_error: float
    max_absolute_error: float
    mean_relative_error: float
    std_relative_error: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy serialization."""
        return {
            'MSE': self.mse,
            'RMSE': self.rmse,
            'MAE': self.mae,
            'Relative_Percentage_Error': self.relative_percentage_error,
            'Max_Absolute_Error': self.max_absolute_error,
            'Mean_Relative_Error': self.mean_relative_error,
            'Std_Relative_Error': self.std_relative_error
        }


class FunahashiMetrics:
    """
    Implementation of Funahashi's exact evaluation metrics.
    
    These metrics match the evaluation approach used in:
    "SABR Equipped with AI Wings" by Funahashi et al.
    """
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Squared Error as used in Funahashi's paper.
        
        Args:
            y_true: True volatility values
            y_pred: Predicted volatility values
            
        Returns:
            MSE value
        """
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.
        
        Args:
            y_true: True volatility values
            y_pred: Predicted volatility values
            
        Returns:
            RMSE value
        """
        return np.sqrt(FunahashiMetrics.mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            y_true: True volatility values
            y_pred: Predicted volatility values
            
        Returns:
            MAE value
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def relative_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, 
                                epsilon: float = 1e-8) -> float:
        """
        Calculate relative percentage error as used in Funahashi's paper.
        
        This is calculated as: mean(|y_true - y_pred| / (y_true + epsilon)) * 100
        
        Args:
            y_true: True volatility values
            y_pred: Predicted volatility values
            epsilon: Small value to avoid division by zero
            
        Returns:
            Relative percentage error
        """
        relative_errors = np.abs(y_true - y_pred) / (np.abs(y_true) + epsilon)
        return np.mean(relative_errors) * 100
    
    @staticmethod
    def max_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate maximum absolute error.
        
        Args:
            y_true: True volatility values
            y_pred: Predicted volatility values
            
        Returns:
            Maximum absolute error
        """
        return np.max(np.abs(y_true - y_pred))
    
    @staticmethod
    def relative_error_statistics(y_true: np.ndarray, y_pred: np.ndarray,
                                epsilon: float = 1e-8) -> Tuple[float, float]:
        """
        Calculate mean and standard deviation of relative errors.
        
        Args:
            y_true: True volatility values
            y_pred: Predicted volatility values
            epsilon: Small value to avoid division by zero
            
        Returns:
            Tuple of (mean_relative_error, std_relative_error)
        """
        relative_errors = (y_true - y_pred) / (np.abs(y_true) + epsilon)
        return np.mean(relative_errors), np.std(relative_errors)


class ModelEvaluator:
    """
    Comprehensive model evaluation class for SABR volatility surface models.
    """
    
    def __init__(self):
        self.funahashi_metrics = FunahashiMetrics()
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationResults:
        """
        Evaluate model predictions using all metrics.
        
        Args:
            y_true: True volatility values
            y_pred: Predicted volatility values
            
        Returns:
            EvaluationResults containing all metrics
        """
        # Flatten arrays if needed
        y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
        y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
        
        # Calculate all metrics
        mse = self.funahashi_metrics.mean_squared_error(y_true_flat, y_pred_flat)
        rmse = self.funahashi_metrics.root_mean_squared_error(y_true_flat, y_pred_flat)
        mae = self.funahashi_metrics.mean_absolute_error(y_true_flat, y_pred_flat)
        rel_pct_error = self.funahashi_metrics.relative_percentage_error(y_true_flat, y_pred_flat)
        max_abs_error = self.funahashi_metrics.max_absolute_error(y_true_flat, y_pred_flat)
        mean_rel_error, std_rel_error = self.funahashi_metrics.relative_error_statistics(
            y_true_flat, y_pred_flat
        )
        
        return EvaluationResults(
            mse=mse,
            rmse=rmse,
            mae=mae,
            relative_percentage_error=rel_pct_error,
            max_absolute_error=max_abs_error,
            mean_relative_error=mean_rel_error,
            std_relative_error=std_rel_error
        )
    
    def evaluate_by_region(self, y_true: np.ndarray, y_pred: np.ndarray,
                          strikes: np.ndarray, forward_price: float,
                          atm_threshold: float = 0.05) -> Dict[str, EvaluationResults]:
        """
        Evaluate model performance by moneyness regions (ITM, ATM, OTM).
        
        Args:
            y_true: True volatility values
            y_pred: Predicted volatility values
            strikes: Strike prices corresponding to predictions
            forward_price: Forward price for moneyness calculation
            atm_threshold: Threshold for ATM region (default 5%)
            
        Returns:
            Dictionary with evaluation results for each region
        """
        # Calculate moneyness
        moneyness = strikes / forward_price
        
        # Define regions
        atm_mask = np.abs(moneyness - 1.0) <= atm_threshold
        itm_mask = (moneyness < (1.0 - atm_threshold))
        otm_mask = (moneyness > (1.0 + atm_threshold))
        
        results = {}
        
        # Evaluate each region
        for region_name, mask in [('ITM', itm_mask), ('ATM', atm_mask), ('OTM', otm_mask)]:
            if np.any(mask):
                region_true = y_true[mask]
                region_pred = y_pred[mask]
                results[region_name] = self.evaluate_model(region_true, region_pred)
            else:
                # Handle case where region has no data points
                results[region_name] = EvaluationResults(
                    mse=np.nan, rmse=np.nan, mae=np.nan,
                    relative_percentage_error=np.nan, max_absolute_error=np.nan,
                    mean_relative_error=np.nan, std_relative_error=np.nan
                )
        
        return results
    
    def evaluate_tensorflow_model(self, model: tf.keras.Model, 
                                test_data: tf.data.Dataset) -> EvaluationResults:
        """
        Evaluate a TensorFlow model on test data.
        
        Args:
            model: Trained TensorFlow model
            test_data: Test dataset
            
        Returns:
            EvaluationResults
        """
        # Collect predictions and true values
        y_true_list = []
        y_pred_list = []
        
        for batch in test_data:
            if isinstance(batch, tuple) and len(batch) == 2:
                x_batch, y_batch = batch
            else:
                raise ValueError("Expected test_data to yield (x, y) tuples")
            
            # Get predictions
            y_pred_batch = model(x_batch, training=False)
            
            # Convert to numpy and collect
            y_true_list.append(y_batch.numpy())
            y_pred_list.append(y_pred_batch.numpy())
        
        # Concatenate all batches
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        
        return self.evaluate_model(y_true, y_pred)


class WingRegionAnalyzer:
    """
    Specialized analyzer for volatility wing regions where MC-Hagan residuals are largest.
    """
    
    def __init__(self, wing_threshold_percentile: float = 90):
        """
        Initialize wing region analyzer.
        
        Args:
            wing_threshold_percentile: Percentile threshold for identifying wing regions
        """
        self.wing_threshold_percentile = wing_threshold_percentile
    
    def identify_wing_regions(self, residuals: np.ndarray, 
                            strikes: np.ndarray, 
                            forward_price: float) -> np.ndarray:
        """
        Identify wing regions based on residual magnitude.
        
        Args:
            residuals: MC-Hagan residuals
            strikes: Strike prices
            forward_price: Forward price
            
        Returns:
            Boolean mask indicating wing regions
        """
        # Calculate absolute residuals
        abs_residuals = np.abs(residuals)
        
        # Find threshold based on percentile
        threshold = np.percentile(abs_residuals, self.wing_threshold_percentile)
        
        # Create wing mask
        wing_mask = abs_residuals >= threshold
        
        return wing_mask
    
    def evaluate_wing_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                residuals: np.ndarray, strikes: np.ndarray,
                                forward_price: float) -> Dict[str, EvaluationResults]:
        """
        Evaluate model performance specifically in wing regions.
        
        Args:
            y_true: True volatility values
            y_pred: Predicted volatility values
            residuals: MC-Hagan residuals for wing identification
            strikes: Strike prices
            forward_price: Forward price
            
        Returns:
            Dictionary with wing and non-wing evaluation results
        """
        evaluator = ModelEvaluator()
        
        # Identify wing regions
        wing_mask = self.identify_wing_regions(residuals, strikes, forward_price)
        
        results = {}
        
        # Evaluate wing regions
        if np.any(wing_mask):
            wing_true = y_true[wing_mask]
            wing_pred = y_pred[wing_mask]
            results['Wing_Regions'] = evaluator.evaluate_model(wing_true, wing_pred)
        
        # Evaluate non-wing regions
        non_wing_mask = ~wing_mask
        if np.any(non_wing_mask):
            non_wing_true = y_true[non_wing_mask]
            non_wing_pred = y_pred[non_wing_mask]
            results['Non_Wing_Regions'] = evaluator.evaluate_model(non_wing_true, non_wing_pred)
        
        return results