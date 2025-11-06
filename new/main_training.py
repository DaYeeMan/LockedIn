#!/usr/bin/env python3
"""
Main training script for SABR volatility surface modeling.

This script trains both MDA-CNN and Funahashi baseline models with the same data
and configuration, enabling direct performance comparison as specified in the requirements.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.training_config import TrainingConfig
from training.trainer import Trainer
from models.mdacnn_model import MDACNNModel
from models.baseline_models import FunahashiBaselineModel
from preprocessing.data_loader import SABRDataset
from utils.logging_utils import setup_logging
from utils.config_utils import load_config, save_config
from evaluation.metrics import calculate_all_metrics
from visualization.surface_comparison import create_comparison_plots


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SABR volatility surface models')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing preprocessed training data')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save training results')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for this experiment (default: timestamp)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    return parser.parse_args()


def setup_experiment(args):
    """Set up experiment directory and logging."""
    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"sabr_comparison_{timestamp}"
    
    # Create output directory
    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = experiment_dir / "training.log"
    setup_logging(log_file, level=logging.INFO)
    
    return experiment_dir


def setup_device(device_arg):
    """Setup and return the appropriate device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    logging.info(f"Using device: {device}")
    return device


def load_datasets(data_dir, config):
    """Load training, validation, and test datasets."""
    logging.info("Loading datasets...")
    
    # Load datasets
    train_dataset = SABRDataset(
        data_dir=data_dir,
        split='train',
        patch_size=config.patch_size,
        normalize=True
    )
    
    val_dataset = SABRDataset(
        data_dir=data_dir,
        split='val',
        patch_size=config.patch_size,
        normalize=True
    )
    
    test_dataset = SABRDataset(
        data_dir=data_dir,
        split='test',
        patch_size=config.patch_size,
        normalize=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    logging.info(f"Loaded {len(train_dataset)} training samples")
    logging.info(f"Loaded {len(val_dataset)} validation samples")
    logging.info(f"Loaded {len(test_dataset)} test samples")
    
    return train_loader, val_loader, test_loader


def create_models(config, device):
    """Create MDA-CNN and Funahashi baseline models."""
    logging.info("Creating models...")
    
    # MDA-CNN model
    mdacnn_model = MDACNNModel(
        patch_size=config.patch_size,
        point_features_dim=config.point_features_dim,
        cnn_channels=config.cnn_channels,
        mlp_hidden_dims=config.mlp_hidden_dims,
        fusion_dim=config.fusion_dim,
        dropout_rate=config.dropout_rate
    ).to(device)
    
    # Funahashi baseline model
    funahashi_model = FunahashiBaselineModel(
        input_dim=config.point_features_dim,
        hidden_dim=32,  # As specified in Funahashi paper
        num_layers=5,   # As specified in Funahashi paper
        dropout_rate=config.dropout_rate
    ).to(device)
    
    logging.info(f"MDA-CNN parameters: {sum(p.numel() for p in mdacnn_model.parameters()):,}")
    logging.info(f"Funahashi parameters: {sum(p.numel() for p in funahashi_model.parameters()):,}")
    
    return mdacnn_model, funahashi_model


def train_model(model, model_name, train_loader, val_loader, config, device, experiment_dir):
    """Train a single model and return the best model."""
    logging.info(f"Training {model_name} model...")
    
    # Create model-specific output directory
    model_dir = experiment_dir / model_name.lower().replace('-', '_')
    model_dir.mkdir(exist_ok=True)
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        output_dir=model_dir
    )
    
    # Train the model
    best_model, training_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs,
        patience=config.patience,
        save_checkpoints=True
    )
    
    logging.info(f"{model_name} training completed")
    return best_model, training_history


def evaluate_models(mdacnn_model, funahashi_model, test_loader, device, experiment_dir):
    """Evaluate both models on test set and generate comparison metrics."""
    logging.info("Evaluating models on test set...")
    
    models = {
        'MDA-CNN': mdacnn_model,
        'Funahashi': funahashi_model
    }
    
    results = {}
    
    for model_name, model in models.items():
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                if model_name == 'MDA-CNN':
                    patches, point_features, target = batch
                    patches = patches.to(device)
                    point_features = point_features.to(device)
                    pred = model(patches, point_features)
                else:  # Funahashi baseline
                    _, point_features, target = batch
                    point_features = point_features.to(device)
                    pred = model(point_features)
                
                predictions.append(pred.cpu().numpy())
                targets.append(target.numpy())
        
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        # Calculate metrics
        metrics = calculate_all_metrics(predictions, targets)
        results[model_name] = {
            'predictions': predictions,
            'targets': targets,
            'metrics': metrics
        }
        
        logging.info(f"{model_name} Test Metrics:")
        for metric_name, value in metrics.items():
            logging.info(f"  {metric_name}: {value:.6f}")
    
    # Save evaluation results
    eval_results_path = experiment_dir / "evaluation_results.npz"
    np.savez(
        eval_results_path,
        mdacnn_predictions=results['MDA-CNN']['predictions'],
        funahashi_predictions=results['Funahashi']['predictions'],
        targets=results['MDA-CNN']['targets'],
        mdacnn_metrics=results['MDA-CNN']['metrics'],
        funahashi_metrics=results['Funahashi']['metrics']
    )
    
    return results


def generate_comparison_report(results, experiment_dir):
    """Generate comprehensive comparison report and visualizations."""
    logging.info("Generating comparison report...")
    
    # Create comparison plots
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    create_comparison_plots(
        mdacnn_predictions=results['MDA-CNN']['predictions'],
        funahashi_predictions=results['Funahashi']['predictions'],
        targets=results['MDA-CNN']['targets'],
        output_dir=plots_dir
    )
    
    # Generate text report
    report_path = experiment_dir / "comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write("SABR Volatility Surface Model Comparison Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Model Performance Comparison:\n")
        f.write("-" * 30 + "\n")
        
        for model_name in ['MDA-CNN', 'Funahashi']:
            f.write(f"\n{model_name} Model:\n")
            metrics = results[model_name]['metrics']
            for metric_name, value in metrics.items():
                f.write(f"  {metric_name}: {value:.6f}\n")
        
        # Calculate improvement percentages
        f.write("\nImprovement Analysis:\n")
        f.write("-" * 20 + "\n")
        
        mdacnn_metrics = results['MDA-CNN']['metrics']
        funahashi_metrics = results['Funahashi']['metrics']
        
        for metric_name in mdacnn_metrics.keys():
            mdacnn_val = mdacnn_metrics[metric_name]
            funahashi_val = funahashi_metrics[metric_name]
            
            if funahashi_val != 0:
                improvement = ((funahashi_val - mdacnn_val) / funahashi_val) * 100
                f.write(f"{metric_name} improvement: {improvement:.2f}%\n")
    
    logging.info(f"Comparison report saved to {report_path}")


def main():
    """Main training and comparison pipeline."""
    args = parse_arguments()
    
    # Setup experiment
    experiment_dir = setup_experiment(args)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Save configuration to experiment directory
        save_config(config, experiment_dir / "config.yaml")
        
        # Load datasets
        train_loader, val_loader, test_loader = load_datasets(args.data_dir, config)
        
        # Create models
        mdacnn_model, funahashi_model = create_models(config, device)
        
        # Train MDA-CNN model
        mdacnn_trained, mdacnn_history = train_model(
            mdacnn_model, "MDA-CNN", train_loader, val_loader, 
            config, device, experiment_dir
        )
        
        # Train Funahashi baseline model
        funahashi_trained, funahashi_history = train_model(
            funahashi_model, "Funahashi", train_loader, val_loader,
            config, device, experiment_dir
        )
        
        # Evaluate both models
        results = evaluate_models(
            mdacnn_trained, funahashi_trained, test_loader, device, experiment_dir
        )
        
        # Generate comparison report and visualizations
        generate_comparison_report(results, experiment_dir)
        
        logging.info("Training and comparison completed successfully!")
        logging.info(f"Results saved to: {experiment_dir}")
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()