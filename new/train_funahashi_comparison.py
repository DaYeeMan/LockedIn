#!/usr/bin/env python3
"""
Simple training script for Funahashi comparison.

This script trains both MDA-CNN and Funahashi baseline models using the
generated comparison data.
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


def create_data_loaders(data_dir, batch_size=32):
    """Create data loaders for training."""
    # Load datasets
    train_dataset = SABRDataset(data_dir, split='train')
    
    # Check if we have validation data
    try:
        val_dataset = SABRDataset(data_dir, split='val')
        if len(val_dataset) == 0:
            val_dataset = None
    except:
        val_dataset = None
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    
    logging.info(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        logging.info(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, model_name, experiment_dir, epochs=100):
    """Train a single model."""
    logging.info(f"Training {model_name}...")
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch_times': []
    }
    
    best_val_loss = float('inf')
    patience = 20
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
            optimizer.step()
            
            train_loss += loss.item() * len(targets)
            train_samples += len(targets)
        
        avg_train_loss = train_loss / train_samples
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        avg_val_loss = 0.0
        if val_loader:
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
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), experiment_dir / f"{model_name.lower()}_best.pth")
            else:
                patience_counter += 1
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)
        
        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if val_loader:
                logging.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}, "
                           f"Val Loss: {avg_val_loss:.6f}, Time: {epoch_time:.2f}s")
            else:
                logging.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}, "
                           f"Time: {epoch_time:.2f}s")
        
        # Early stopping
        if val_loader and patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save final model
    torch.save(model.state_dict(), experiment_dir / f"{model_name.lower()}_final.pth")
    
    # Save training history
    import json
    with open(experiment_dir / f"{model_name.lower()}_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    logging.info(f"{model_name} training completed!")
    return model, history


def evaluate_models(mdacnn_model, funahashi_model, train_loader, experiment_dir):
    """Quick evaluation of both models."""
    logging.info("Evaluating models...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdacnn_model.eval()
    funahashi_model.eval()
    
    mdacnn_predictions = []
    funahashi_predictions = []
    targets_list = []
    
    with torch.no_grad():
        for patches, features, targets in train_loader:
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
    
    # Calculate metrics
    mdacnn_predictions = np.array(mdacnn_predictions)
    funahashi_predictions = np.array(funahashi_predictions)
    targets_array = np.array(targets_list)
    
    # MSE
    mdacnn_mse = np.mean((mdacnn_predictions - targets_array) ** 2)
    funahashi_mse = np.mean((funahashi_predictions - targets_array) ** 2)
    
    # RMSE
    mdacnn_rmse = np.sqrt(mdacnn_mse)
    funahashi_rmse = np.sqrt(funahashi_mse)
    
    # MAE
    mdacnn_mae = np.mean(np.abs(mdacnn_predictions - targets_array))
    funahashi_mae = np.mean(np.abs(funahashi_predictions - targets_array))
    
    # Log results
    logging.info("=" * 60)
    logging.info("MODEL COMPARISON RESULTS")
    logging.info("=" * 60)
    logging.info(f"MDA-CNN:")
    logging.info(f"  MSE:  {mdacnn_mse:.6f}")
    logging.info(f"  RMSE: {mdacnn_rmse:.6f}")
    logging.info(f"  MAE:  {mdacnn_mae:.6f}")
    logging.info(f"")
    logging.info(f"Funahashi Baseline:")
    logging.info(f"  MSE:  {funahashi_mse:.6f}")
    logging.info(f"  RMSE: {funahashi_rmse:.6f}")
    logging.info(f"  MAE:  {funahashi_mae:.6f}")
    logging.info(f"")
    logging.info(f"Improvement (MDA-CNN vs Funahashi):")
    logging.info(f"  MSE:  {((funahashi_mse - mdacnn_mse) / funahashi_mse * 100):+.2f}%")
    logging.info(f"  RMSE: {((funahashi_rmse - mdacnn_rmse) / funahashi_rmse * 100):+.2f}%")
    logging.info(f"  MAE:  {((funahashi_mae - mdacnn_mae) / funahashi_mae * 100):+.2f}%")
    
    # Save results
    results = {
        'mdacnn': {
            'mse': float(mdacnn_mse),
            'rmse': float(mdacnn_rmse),
            'mae': float(mdacnn_mae)
        },
        'funahashi': {
            'mse': float(funahashi_mse),
            'rmse': float(funahashi_rmse),
            'mae': float(funahashi_mae)
        },
        'improvement': {
            'mse_percent': float((funahashi_mse - mdacnn_mse) / funahashi_mse * 100),
            'rmse_percent': float((funahashi_rmse - mdacnn_rmse) / funahashi_rmse * 100),
            'mae_percent': float((funahashi_mae - mdacnn_mae) / funahashi_mae * 100)
        }
    }
    
    with open(experiment_dir / "comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """Main training function."""
    print("FUNAHASHI COMPARISON TRAINING")
    print("=" * 50)
    
    # Setup experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("results") / f"funahashi_comparison_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(experiment_dir)
    
    logging.info("Starting Funahashi comparison training...")
    
    try:
        # Create data loaders - USE LARGER DATASET
        data_dir = "data/processed"
        train_loader, val_loader = create_data_loaders(data_dir, batch_size=32)
        
        # Get sample to determine dimensions
        sample_patches, sample_features, _ = next(iter(train_loader))
        patch_size = sample_patches.shape[-1]  # Assuming square patches
        n_features = sample_features.shape[-1]
        
        logging.info(f"Patch size: {patch_size}x{patch_size}")
        logging.info(f"Number of features: {n_features}")
        
        # Create models
        mdacnn_model = MDACNNModel(
            patch_size=patch_size,
            point_features_dim=n_features,
            cnn_channels=[32, 64, 128],
            mlp_hidden_dims=[64, 64],
            fusion_dim=128,
            dropout_rate=0.2
        )
        
        funahashi_model = FunahashiBaselineModel(n_point_features=n_features)
        
        logging.info(f"MDA-CNN parameters: {mdacnn_model.count_parameters():,}")
        logging.info(f"Funahashi parameters: {funahashi_model.count_parameters():,}")
        
        # Train MDA-CNN
        mdacnn_trained, mdacnn_history = train_model(
            mdacnn_model, train_loader, val_loader, "MDA-CNN", experiment_dir, epochs=100
        )
        
        # Train Funahashi baseline
        funahashi_trained, funahashi_history = train_model(
            funahashi_model, train_loader, val_loader, "Funahashi", experiment_dir, epochs=100
        )
        
        # Evaluate both models
        results = evaluate_models(mdacnn_trained, funahashi_trained, train_loader, experiment_dir)
        
        print("\n" + "=" * 50)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Results saved to: {experiment_dir}")
        print(f"MDA-CNN MSE: {results['mdacnn']['mse']:.6f}")
        print(f"Funahashi MSE: {results['funahashi']['mse']:.6f}")
        print(f"Improvement: {results['improvement']['mse_percent']:+.2f}%")
        
        return 0
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())