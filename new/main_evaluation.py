#!/usr/bin/env python3
"""
Main evaluation script for SABR volatility surface models.

This script loads trained models and generates comprehensive comparison
analysis including all visualizations and metrics specified in requirements.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.training_config import TrainingConfig
from models.mdacnn_model import MDACNNModel
from models.baseline_models import FunahashiBaselineModel
from preprocessing.data_loader import SABRDataset
from utils.logging_utils import setup_logging
from utils.config_utils import load_config
from evaluation.metrics import calculate_all_metrics
from evaluation.model_comparison import ModelComparison
from visualization.surface_comparison import create_all_comparison_plots


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate SABR volatility surface models')
    parser.add_argument('--experiment-dir', type=str, required=True,
                       help='Directory containing trained models and config')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing test data')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save evaluation results (default: experiment_dir/evaluation)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for evaluation')
    parser.add_argument('--generate-plots', action='store_true', default=True,
                       help='Generate comparison plots')
    parser.add_argument('--detailed-analysis', action='store_true',
                       help='Generate detailed analysis including surface reconstructions')
    return parser.parse_args()


def setup_evaluation(args):
    """Set up evaluation environment."""
    experiment_dir = Path(args.experiment_dir)
    
    if args.output_dir is None:
        output_dir = experiment_dir / "evaluation"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / "evaluation.log"
    setup_logging(log_file, level=logging.INFO)
    
    return experiment_dir, output_dir


def load_trained_models(experiment_dir, config, device):
    """Load the trained MDA-CNN and Funahashi models."""
    logging.info("Loading trained models...")
    
    # Create model instances
    mdacnn_model = MDACNNModel(
        patch_size=config.patch_size,
        point_features_dim=config.point_features_dim,
        cnn_channels=config.cnn_channels,
        mlp_hidden_dims=config.mlp_hidden_dims,
        fusion_dim=config.fusion_dim,
        dropout_rate=config.dropout_rate
    ).to(device)
    
    funahashi_model = FunahashiBaselineModel(
        input_dim=config.point_features_dim,
        hidden_dim=32,
        num_layers=5,
        dropout_rate=config.dropout_rate
    ).to(device)
    
    # Load trained weights
    mdacnn_path = experiment_dir / "mda_cnn" / "best_model.pth"
    funahashi_path = experiment_dir / "funahashi" / "best_model.pth"
    
    if not mdacnn_path.exists():
        raise FileNotFoundError(f"MDA-CNN model not found at {mdacnn_path}")
    if not funahashi_path.exists():
        raise FileNotFoundError(f"Funahashi model not found at {funahashi_path}")
    
    mdacnn_model.load_state_dict(torch.load(mdacnn_path, map_location=device))
    funahashi_model.load_state_dict(torch.load(funahashi_path, map_location=device))
    
    mdacnn_model.eval()
    funahashi_model.eval()
    
    logging.info("Models loaded successfully")
    return mdacnn_model, funahashi_model


def comprehensive_evaluation(mdacnn_model, funahashi_model, test_loader, device, output_dir, detailed=False):
    """Perform comprehensive evaluation of both models."""
    logging.info("Starting comprehensive evaluation...")
    
    # Initialize model comparison
    comparison = ModelComparison(
        models={
            'MDA-CNN': mdacnn_model,
            'Funahashi': funahashi_model
        },
        device=device
    )
    
    # Run evaluation
    results = comparison.evaluate_models(test_loader)
    
    # Save detailed results
    results_path = output_dir / "detailed_results.npz"
    np.savez(
        results_path,
        **{f"{name}_{key}": value for name, model_results in results.items() 
           for key, value in model_results.items()}
    )
    
    # Generate comparison metrics table
    metrics_table = comparison.generate_metrics_table(results)
    
    # Save metrics table
    metrics_path = output_dir / "metrics_comparison.txt"
    with open(metrics_path, 'w') as f:
        f.write("SABR Volatility Surface Model Comparison\n")
        f.write("=" * 50 + "\n\n")
        f.write(metrics_table)
    
    logging.info(f"Metrics comparison saved to {metrics_path}")
    
    if detailed:
        # Generate detailed analysis
        detailed_analysis = comparison.detailed_analysis(results, test_loader)
        
        # Save detailed analysis
        detailed_path = output_dir / "detailed_analysis.txt"
        with open(detailed_path, 'w') as f:
            f.write(detailed_analysis)
        
        logging.info(f"Detailed analysis saved to {detailed_path}")
    
    return results


def generate_all_visualizations(results, output_dir):
    """Generate all comparison visualizations."""
    logging.info("Generating visualizations...")
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Create comprehensive comparison plots
    create_all_comparison_plots(
        mdacnn_predictions=results['MDA-CNN']['predictions'],
        funahashi_predictions=results['Funahashi']['predictions'],
        targets=results['MDA-CNN']['targets'],
        mdacnn_metrics=results['MDA-CNN']['metrics'],
        funahashi_metrics=results['Funahashi']['metrics'],
        output_dir=plots_dir
    )
    
    logging.info(f"All plots saved to {plots_dir}")


def generate_final_report(results, experiment_dir, output_dir):
    """Generate final comprehensive report."""
    logging.info("Generating final report...")
    
    report_path = output_dir / "final_comparison_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# SABR Volatility Surface Model Comparison Report\n\n")
        
        f.write("## Experiment Overview\n")
        f.write(f"- Experiment Directory: {experiment_dir}\n")
        f.write(f"- Evaluation Date: {np.datetime64('now')}\n\n")
        
        f.write("## Model Performance Summary\n\n")
        
        # Performance table
        f.write("| Metric | MDA-CNN | Funahashi | Improvement |\n")
        f.write("|--------|---------|-----------|-------------|\n")
        
        mdacnn_metrics = results['MDA-CNN']['metrics']
        funahashi_metrics = results['Funahashi']['metrics']
        
        for metric_name in mdacnn_metrics.keys():
            mdacnn_val = mdacnn_metrics[metric_name]
            funahashi_val = funahashi_metrics[metric_name]
            
            if funahashi_val != 0:
                improvement = ((funahashi_val - mdacnn_val) / funahashi_val) * 100
                f.write(f"| {metric_name} | {mdacnn_val:.6f} | {funahashi_val:.6f} | {improvement:.2f}% |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Determine best performing model
        mdacnn_mse = mdacnn_metrics.get('mse', float('inf'))
        funahashi_mse = funahashi_metrics.get('mse', float('inf'))
        
        if mdacnn_mse < funahashi_mse:
            improvement = ((funahashi_mse - mdacnn_mse) / funahashi_mse) * 100
            f.write(f"- **MDA-CNN outperforms Funahashi baseline** by {improvement:.2f}% in MSE\n")
        else:
            degradation = ((mdacnn_mse - funahashi_mse) / funahashi_mse) * 100
            f.write(f"- **Funahashi baseline outperforms MDA-CNN** by {degradation:.2f}% in MSE\n")
        
        f.write("- Detailed visualizations available in the plots/ directory\n")
        f.write("- Raw evaluation data saved in detailed_results.npz\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `metrics_comparison.txt`: Detailed metrics comparison\n")
        f.write("- `detailed_results.npz`: Raw evaluation data\n")
        f.write("- `plots/`: All comparison visualizations\n")
        f.write("- `evaluation.log`: Evaluation process log\n")
    
    logging.info(f"Final report saved to {report_path}")


def main():
    """Main evaluation pipeline."""
    args = parse_arguments()
    
    # Setup evaluation
    experiment_dir, output_dir = setup_evaluation(args)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logging.info(f"Using device: {device}")
    
    try:
        # Load configuration
        config_path = experiment_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        config = load_config(config_path)
        
        # Load test dataset
        test_dataset = SABRDataset(
            data_dir=args.data_dir,
            split='test',
            patch_size=config.patch_size,
            normalize=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        logging.info(f"Loaded {len(test_dataset)} test samples")
        
        # Load trained models
        mdacnn_model, funahashi_model = load_trained_models(experiment_dir, config, device)
        
        # Perform comprehensive evaluation
        results = comprehensive_evaluation(
            mdacnn_model, funahashi_model, test_loader, device, 
            output_dir, detailed=args.detailed_analysis
        )
        
        # Generate visualizations
        if args.generate_plots:
            generate_all_visualizations(results, output_dir)
        
        # Generate final report
        generate_final_report(results, experiment_dir, output_dir)
        
        logging.info("Evaluation completed successfully!")
        logging.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logging.error(f"Evaluation failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()