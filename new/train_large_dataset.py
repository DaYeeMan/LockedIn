#!/usr/bin/env python3
"""
Training script optimized for larger dataset.

This script trains both MDA-CNN and Funahashi baseline models using the
larger dataset in data/processed for better performance comparison.
"""

import os
import sys
import logging
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.mdacnn_model import MDACNNModel
from models.baseline_models import FunahashiBaselineModel
from preprocessing.data_loader import SABRDataset
from config.training_config import TrainingConfig


def setup_logging(experiment_dir):
    """Setup logging for the experiment."""
    log_file = experiment_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def create_data_loaders(data_dir, batch_size=64):
    """Create data loaders for training with larger dataset."""
    # Load datasets
    train_dataset = SABRDataset(data_dir, split='train')
    val_dataset = SABRDataset(data_dir, split='val')
    test_dataset = SABRDataset(data_dir, split='test')
    
    # Create data loaders with larger batch size for efficiency
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    logging.info(f"Training samples: {len(train_dataset)}")
    logging.info(f"Validation samples: {len(val_dataset)}")
    logging.info(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, model_name, experiment_dir, epochs=200):
    """Train a single model with optimized settings for larger dataset."""
    logging.info(f"Training {model_name}...")
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logging.info(f"Using device: {device}")
    
    criterion = nn.MSELoss()
    
    # Optimized learning rate and scheduler for larger dataset
    if isinstance(model, MDACNNModel):
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    else:  # Funahashi baseline
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
        'epoch_times': []
    }
    
    best_val_loss = float('inf')
    patience = 30  # Increased patience for larger dataset
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch_idx, (patches, features, targets) in enumerate(train_loader):
            patches = patches.float().to(device)
            features = features.float().to(device)
            targets = targets.float().to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            if isinstance(model, MDACNNModel):
                outputs = model(patches, features)
            else:  # Funahashi baseline
                outputs = model(features)
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * len(targets)
            train_samples += len(targets)
        
        avg_train_loss = train_loss / train_samples
        history['train_loss'].append(avg_train_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for patches, features, targets in val_loader:
                patches = patches.float().to(device)
                features = features.float().to(device)
                targets = targets.float().to(device).unsqueeze(1)
                
                if isinstance(model, MDACNNModel):
                    outputs = model(patches, features)
                else:
                    outputs = model(features)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item() * len(targets)
                val_samples += len(targets)
        
        avg_val_loss = val_loss / val_samples
        history['val_loss'].append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), experiment_dir / f"{model_name.lower()}_best.pth")
        else:
            patience_counter += 1
        
        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)
        
        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logging.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}, "
                       f"Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}, "
                       f"Time: {epoch_time:.2f}s")
        
        # Early stopping
        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save final model
    torch.save(model.state_dict(), experiment_dir / f"{model_name.lower()}_final.pth")
    
    # Save training history
    with open(experiment_dir / f"{model_name.lower()}_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    logging.info(f"{model_name} training completed! Best val loss: {best_val_loss:.6f}")
    return model, history


def evaluate_models(mdacnn_model, funahashi_model, test_loader, experiment_dir):
    """Comprehensive evaluation of both models."""
    logging.info("Evaluating models on test set...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdacnn_model.eval()
    funahashi_model.eval()
    
    mdacnn_predictions = []
    funahashi_predictions = []
    targets_list = []
    
    with torch.no_grad():
        for patches, features, targets in test_loader:
            patches = patches.float().to(device)
            features = features.float().to(device)
            targets = targets.float().to(device)
            
            # MDA-CNN predictions
            mdacnn_pred = mdacnn_model(patches, features).cpu().numpy().flatten()
            mdacnn_predictions.extend(mdacnn_pred)
            
            # Funahashi predictions
            funahashi_pred = funahashi_model(features).cpu().numpy().flatten()
            funahashi_predictions.extend(funahashi_pred)
            
            targets_list.extend(targets.cpu().numpy())
    
    # Calculate comprehensive metrics
    mdacnn_predictions = np.array(mdacnn_predictions)
    funahashi_predictions = np.array(funahashi_predictions)
    targets_array = np.array(targets_list)
    
    # MSE, RMSE, MAE
    mdacnn_mse = np.mean((mdacnn_predictions - targets_array) ** 2)
    funahashi_mse = np.mean((funahashi_predictions - targets_array) ** 2)
    
    mdacnn_rmse = np.sqrt(mdacnn_mse)
    funahashi_rmse = np.sqrt(funahashi_mse)
    
    mdacnn_mae = np.mean(np.abs(mdacnn_predictions - targets_array))
    funahashi_mae = np.mean(np.abs(funahashi_predictions - targets_array))
    
    # R-squared
    ss_res_mdacnn = np.sum((targets_array - mdacnn_predictions) ** 2)
    ss_res_funahashi = np.sum((targets_array - funahashi_predictions) ** 2)
    ss_tot = np.sum((targets_array - np.mean(targets_array)) ** 2)
    
    mdacnn_r2 = 1 - (ss_res_mdacnn / ss_tot)
    funahashi_r2 = 1 - (ss_res_funahashi / ss_tot)
    
    # Log results
    logging.info("=" * 70)
    logging.info("COMPREHENSIVE MODEL COMPARISON RESULTS")
    logging.info("=" * 70)
    logging.info(f"Dataset size: {len(targets_array)} test samples")
    logging.info("")
    logging.info(f"MDA-CNN:")
    logging.info(f"  MSE:  {mdacnn_mse:.8f}")
    logging.info(f"  RMSE: {mdacnn_rmse:.8f}")
    logging.info(f"  MAE:  {mdacnn_mae:.8f}")
    logging.info(f"  R²:   {mdacnn_r2:.6f}")
    logging.info("")
    logging.info(f"Funahashi Baseline:")
    logging.info(f"  MSE:  {funahashi_mse:.8f}")
    logging.info(f"  RMSE: {funahashi_rmse:.8f}")
    logging.info(f"  MAE:  {funahashi_mae:.8f}")
    logging.info(f"  R²:   {funahashi_r2:.6f}")
    logging.info("")
    logging.info(f"Improvement (MDA-CNN vs Funahashi):")
    logging.info(f"  MSE:  {((funahashi_mse - mdacnn_mse) / funahashi_mse * 100):+.2f}%")
    logging.info(f"  RMSE: {((funahashi_rmse - mdacnn_rmse) / funahashi_rmse * 100):+.2f}%")
    logging.info(f"  MAE:  {((funahashi_mae - mdacnn_mae) / funahashi_mae * 100):+.2f}%")
    logging.info(f"  R²:   {((mdacnn_r2 - funahashi_r2) / abs(funahashi_r2) * 100):+.2f}%")
    
    # Save detailed results
    results = {
        'dataset_size': len(targets_array),
        'mdacnn': {
            'mse': float(mdacnn_mse),
            'rmse': float(mdacnn_rmse),
            'mae': float(mdacnn_mae),
            'r2': float(mdacnn_r2)
        },
        'funahashi': {
            'mse': float(funahashi_mse),
            'rmse': float(funahashi_rmse),
            'mae': float(funahashi_mae),
            'r2': float(funahashi_r2)
        },
        'improvement': {
            'mse_percent': float((funahashi_mse - mdacnn_mse) / funahashi_mse * 100),
            'rmse_percent': float((funahashi_rmse - mdacnn_rmse) / funahashi_rmse * 100),
            'mae_percent': float((funahashi_mae - mdacnn_mae) / funahashi_mae * 100),
            'r2_percent': float((mdacnn_r2 - funahashi_r2) / abs(funahashi_r2) * 100)
        }
    }
    
    with open(experiment_dir / "comprehensive_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """Main training function for larger dataset."""
    print("LARGE DATASET TRAINING - MDA-CNN vs FUNAHASHI")
    print("=" * 60)
    
    # Setup experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("results") / f"large_dataset_experiment_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(experiment_dir)
    
    logging.info("Starting large dataset training experiment...")
    
    try:
        # Create data loaders
        data_dir = "data/processed"
        train_loader, val_loader, test_loader = create_data_loaders(data_dir, batch_size=64)
        
        # Get sample to determine dimensions
        sample_patches, sample_features, _ = next(iter(train_loader))
        patch_size = sample_patches.shape[-1]
        n_features = sample_features.shape[-1]
        
        logging.info(f"Patch size: {patch_size}x{patch_size}")
        logging.info(f"Number of features: {n_features}")
        
        # Create models with optimized architectures for larger dataset
        mdacnn_model = MDACNNModel(
            patch_size=patch_size,
            point_features_dim=n_features,
            cnn_channels=[32, 64, 128, 256],  # Deeper CNN for more data
            mlp_hidden_dims=[128, 64, 32],    # Deeper MLP
            fusion_dim=256,                   # Larger fusion layer
            dropout_rate=0.3                  # Higher dropout for regularization
        )
        
        funahashi_model = FunahashiBaselineModel(n_point_features=n_features)
        
        logging.info(f"MDA-CNN parameters: {mdacnn_model.count_parameters():,}")
        logging.info(f"Funahashi parameters: {funahashi_model.count_parameters():,}")
        
        # Train MDA-CNN
        mdacnn_trained, mdacnn_history = train_model(
            mdacnn_model, train_loader, val_loader, "MDA-CNN", experiment_dir, epochs=200
        )
        
        # Train Funahashi baseline
        funahashi_trained, funahashi_history = train_model(
            funahashi_model, train_loader, val_loader, "Funahashi", experiment_dir, epochs=200
        )
        
        # Comprehensive evaluation
        results = evaluate_models(mdacnn_trained, funahashi_trained, test_loader, experiment_dir)
        
        print("\n" + "=" * 60)
        print("🎉 LARGE DATASET TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Results saved to: {experiment_dir}")
        print(f"Dataset size: {results['dataset_size']} test samples")
        print(f"MDA-CNN MSE: {results['mdacnn']['mse']:.8f}")
        print(f"Funahashi MSE: {results['funahashi']['mse']:.8f}")
        print(f"Improvement: {results['improvement']['mse_percent']:+.2f}%")
        
        if results['improvement']['mse_percent'] > 0:
            print("🎯 MDA-CNN OUTPERFORMED FUNAHASHI! 🎯")
        else:
            print("📊 Results show competitive performance")
        
        return 0
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())