"""
Main training infrastructure for SABR volatility surface models.

This module provides the core training loop with validation, early stopping,
model checkpointing, and comprehensive logging for both MDA-CNN and baseline models.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

from config.config_manager import ConfigManager, ExperimentConfig
from models.loss_functions import (
    create_wing_weighted_mse, 
    relative_percentage_error_metric,
    rmse_metric,
    wing_region_mse,
    atm_region_mse
)


class ModelTrainer:
    """
    Main trainer class for SABR volatility surface models.
    
    Handles training loop, validation, early stopping, checkpointing,
    and comprehensive logging for reproducible experiments.
    """
    
    def __init__(self, 
                 config_manager: ConfigManager,
                 model: keras.Model,
                 train_dataset: tf.data.Dataset,
                 val_dataset: tf.data.Dataset,
                 test_dataset: Optional[tf.data.Dataset] = None):
        """
        Initialize trainer.
        
        Args:
            config_manager: Configuration manager instance
            model: Keras model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Optional test dataset for final evaluation
        """
        self.config = config_manager.get_experiment_config()
        self.model_config = config_manager.get_model_config()
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Setup output directories
        self.output_dir = Path(self.config.output_dir)
        self.model_dir = self.output_dir / "models"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        
        # Create directories
        for dir_path in [self.output_dir, self.model_dir, self.log_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.history = None
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.training_start_time = None
        
        # Setup model compilation
        self._compile_model()
        
        # Setup callbacks
        self.callbacks = self._setup_callbacks()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f"trainer_{self.config.name}")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _compile_model(self) -> None:
        """Compile model with appropriate loss function and metrics."""
        # Use MSE loss for residual learning as specified in requirements
        loss_function = keras.losses.MeanSquaredError()
        
        # Setup optimizer
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        
        # Setup metrics
        metrics = [
            'mae',
            rmse_metric,
            relative_percentage_error_metric
        ]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=metrics
        )
        
        self.logger.info(f"Model compiled with MSE loss and learning rate {self.config.learning_rate}")
    
    def _setup_callbacks(self) -> List[keras.callbacks.Callback]:
        """Setup training callbacks."""
        callbacks = []
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model checkpointing
        checkpoint_path = self.checkpoint_dir / f"{self.config.name}_best_model.h5"
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        # Learning rate reduction
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # TensorBoard logging
        tensorboard_dir = self.log_dir / f"tensorboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard)
        
        # Custom training progress callback
        progress_callback = TrainingProgressCallback(self.logger)
        callbacks.append(progress_callback)
        
        return callbacks
    
    def train(self) -> keras.callbacks.History:
        """
        Execute main training loop.
        
        Returns:
            Training history object
        """
        self.logger.info("Starting model training...")
        self.logger.info(f"Training configuration:")
        self.logger.info(f"  - Epochs: {self.config.epochs}")
        self.logger.info(f"  - Batch size: {self.config.batch_size}")
        self.logger.info(f"  - Learning rate: {self.config.learning_rate}")
        self.logger.info(f"  - Early stopping patience: {self.config.early_stopping_patience}")
        
        # Set random seed for reproducibility
        tf.random.set_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        self.training_start_time = time.time()
        
        try:
            # Train model
            self.history = self.model.fit(
                self.train_dataset,
                epochs=self.config.epochs,
                validation_data=self.val_dataset,
                callbacks=self.callbacks,
                verbose=1
            )
            
            training_time = time.time() - self.training_start_time
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save final model
            self._save_final_model()
            
            # Save training history
            self._save_training_history()
            
            # Log final metrics
            self._log_final_metrics()
            
        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            raise
        
        return self.history
    
    def evaluate(self, dataset: Optional[tf.data.Dataset] = None) -> Dict[str, float]:
        """
        Evaluate model on test dataset.
        
        Args:
            dataset: Dataset to evaluate on (uses test_dataset if None)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if dataset is None:
            dataset = self.test_dataset
        
        if dataset is None:
            raise ValueError("No test dataset provided for evaluation")
        
        self.logger.info("Evaluating model on test dataset...")
        
        # Evaluate model
        test_results = self.model.evaluate(dataset, verbose=1, return_dict=True)
        
        # Log results
        self.logger.info("Test evaluation results:")
        for metric_name, value in test_results.items():
            self.logger.info(f"  - {metric_name}: {value:.6f}")
        
        # Save evaluation results
        self._save_evaluation_results(test_results)
        
        return test_results
    
    def _save_final_model(self) -> None:
        """Save the final trained model."""
        model_path = self.model_dir / f"{self.config.name}_final_model.h5"
        self.model.save(str(model_path))
        self.logger.info(f"Final model saved to {model_path}")
        
        # Also save model weights separately
        weights_path = self.model_dir / f"{self.config.name}_final.weights.h5"
        self.model.save_weights(str(weights_path))
        self.logger.info(f"Final model weights saved to {weights_path}")
    
    def _save_training_history(self) -> None:
        """Save training history to CSV and JSON."""
        if self.history is None:
            return
        
        # Convert history to DataFrame
        history_df = pd.DataFrame(self.history.history)
        
        # Save as CSV
        csv_path = self.log_dir / f"{self.config.name}_training_history.csv"
        history_df.to_csv(csv_path, index=False)
        
        # Save as JSON for easy loading
        json_path = self.log_dir / f"{self.config.name}_training_history.json"
        history_df.to_json(json_path, orient='records', indent=2)
        
        self.logger.info(f"Training history saved to {csv_path} and {json_path}")
    
    def _save_evaluation_results(self, results: Dict[str, float]) -> None:
        """Save evaluation results."""
        results_df = pd.DataFrame([results])
        
        # Add metadata
        results_df['experiment_name'] = self.config.name
        results_df['model_type'] = self.model_config.model_type
        results_df['timestamp'] = datetime.now().isoformat()
        
        # Save results
        results_path = self.log_dir / f"{self.config.name}_evaluation_results.csv"
        results_df.to_csv(results_path, index=False)
        
        self.logger.info(f"Evaluation results saved to {results_path}")
    
    def _log_final_metrics(self) -> None:
        """Log final training metrics."""
        if self.history is None:
            return
        
        final_epoch = len(self.history.history['loss'])
        final_train_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        
        # Find best validation loss
        best_val_loss = min(self.history.history['val_loss'])
        best_epoch = self.history.history['val_loss'].index(best_val_loss) + 1
        
        self.logger.info("Final training metrics:")
        self.logger.info(f"  - Total epochs: {final_epoch}")
        self.logger.info(f"  - Final train loss: {final_train_loss:.6f}")
        self.logger.info(f"  - Final val loss: {final_val_loss:.6f}")
        self.logger.info(f"  - Best val loss: {best_val_loss:.6f} (epoch {best_epoch})")
        
        if 'val_rmse_metric' in self.history.history:
            best_val_rmse = min(self.history.history['val_rmse_metric'])
            self.logger.info(f"  - Best val RMSE: {best_val_rmse:.6f}")
        
        if 'val_relative_percentage_error_metric' in self.history.history:
            best_val_rpe = min(self.history.history['val_relative_percentage_error_metric'])
            self.logger.info(f"  - Best val RPE: {best_val_rpe:.2f}%")
    
    def load_best_model(self) -> None:
        """Load the best saved model."""
        checkpoint_path = self.checkpoint_dir / f"{self.config.name}_best_model.h5"
        
        if checkpoint_path.exists():
            self.model = keras.models.load_model(str(checkpoint_path))
            self.logger.info(f"Best model loaded from {checkpoint_path}")
        else:
            self.logger.warning(f"No checkpoint found at {checkpoint_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Dictionary containing training summary information
        """
        if self.history is None:
            return {}
        
        summary = {
            'experiment_name': self.config.name,
            'model_type': self.model_config.model_type,
            'total_epochs': len(self.history.history['loss']),
            'final_train_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history['val_loss'][-1],
            'best_val_loss': min(self.history.history['val_loss']),
            'best_epoch': self.history.history['val_loss'].index(min(self.history.history['val_loss'])) + 1,
            'training_time_seconds': time.time() - self.training_start_time if self.training_start_time else None,
            'config': {
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'epochs': self.config.epochs,
                'early_stopping_patience': self.config.early_stopping_patience
            }
        }
        
        return summary


class TrainingProgressCallback(keras.callbacks.Callback):
    """Custom callback for detailed training progress logging."""
    
    def __init__(self, logger: logging.Logger):
        super().__init__()
        self.logger = logger
        self.epoch_start_time = None
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch."""
        self.epoch_start_time = time.time()
        self.logger.info(f"Starting epoch {epoch + 1}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        if logs is None:
            logs = {}
        
        epoch_time = time.time() - self.epoch_start_time
        
        # Log epoch results
        self.logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        self.logger.info(f"  - Train loss: {logs.get('loss', 0):.6f}")
        self.logger.info(f"  - Val loss: {logs.get('val_loss', 0):.6f}")
        
        if 'rmse_metric' in logs:
            self.logger.info(f"  - Train RMSE: {logs['rmse_metric']:.6f}")
        if 'val_rmse_metric' in logs:
            self.logger.info(f"  - Val RMSE: {logs['val_rmse_metric']:.6f}")
        
        if 'relative_percentage_error_metric' in logs:
            self.logger.info(f"  - Train RPE: {logs['relative_percentage_error_metric']:.2f}%")
        if 'val_relative_percentage_error_metric' in logs:
            self.logger.info(f"  - Val RPE: {logs['val_relative_percentage_error_metric']:.2f}%")


def create_trainer(config_manager: ConfigManager,
                  model: keras.Model,
                  train_dataset: tf.data.Dataset,
                  val_dataset: tf.data.Dataset,
                  test_dataset: Optional[tf.data.Dataset] = None) -> ModelTrainer:
    """
    Factory function to create a ModelTrainer instance.
    
    Args:
        config_manager: Configuration manager
        model: Keras model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        
    Returns:
        ModelTrainer instance
    """
    return ModelTrainer(
        config_manager=config_manager,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )