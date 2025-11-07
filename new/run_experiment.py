#!/usr/bin/env python3
"""
Experiment runner script for SABR volatility surface modeling.

This script provides a simple interface to run complete experiments
including training and evaluation with configurable parameters.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run SABR volatility surface experiment')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--data-config', type=str, default='config/data_generation_config.yaml',
                       help='Path to data generation configuration file')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing preprocessed training data')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save experiment results')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for this experiment (default: timestamp)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training and evaluation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--skip-data-generation', action='store_true',
                       help='Skip data generation (assumes data already exists)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only run evaluation')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation and only run training')
    parser.add_argument('--detailed-analysis', action='store_true',
                       help='Generate detailed analysis during evaluation')
    parser.add_argument('--generate-data-only', action='store_true',
                       help='Only generate data and exit')
    return parser.parse_args()


def run_training(args):
    """Run the training script."""
    print("Starting training phase...")
    
    cmd = [
        sys.executable, 'main_training.py',
        '--config', args.config,
        '--data-dir', args.data_dir,
        '--output-dir', args.output_dir,
        '--device', args.device,
        '--seed', str(args.seed)
    ]
    
    if args.experiment_name:
        cmd.extend(['--experiment-name', args.experiment_name])
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print("Training failed!")
        return False, None
    
    # Determine experiment directory
    if args.experiment_name:
        experiment_dir = Path(args.output_dir) / args.experiment_name
    else:
        # Find the most recent experiment directory
        output_path = Path(args.output_dir)
        if output_path.exists():
            experiment_dirs = [d for d in output_path.iterdir() 
                             if d.is_dir() and d.name.startswith('sabr_comparison_')]
            if experiment_dirs:
                experiment_dir = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
            else:
                print("No experiment directory found!")
                return False, None
        else:
            print("Output directory does not exist!")
            return False, None
    
    print(f"Training completed successfully! Results in: {experiment_dir}")
    return True, experiment_dir


def run_evaluation(experiment_dir, args):
    """Run the evaluation script."""
    print("Starting evaluation phase...")
    
    cmd = [
        sys.executable, 'main_evaluation.py',
        '--experiment-dir', str(experiment_dir),
        '--data-dir', args.data_dir,
        '--device', args.device,
        '--generate-plots'
    ]
    
    if args.detailed_analysis:
        cmd.append('--detailed-analysis')
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print("Evaluation failed!")
        return False
    
    print("Evaluation completed successfully!")
    return True


def run_data_generation(args):
    """Run the data generation script."""
    print("Starting data generation phase...")
    
    cmd = [
        sys.executable, 'generate_training_data.py',
        '--config', args.data_config,
        '--output-dir', 'data',
        '--seed', str(args.seed)
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print("Data generation failed!")
        return False
    
    print("Data generation completed successfully!")
    return True


def main():
    """Main experiment runner."""
    args = parse_arguments()
    
    print("SABR Volatility Surface Model Comparison Experiment")
    print("=" * 55)
    print(f"Training config: {args.config}")
    print(f"Data config: {args.data_config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Run data generation phase
    if not args.skip_data_generation:
        # Check if data already exists
        data_path = Path(args.data_dir)
        if data_path.exists() and any(data_path.iterdir()):
            print(f"Data already exists in {args.data_dir}")
            response = input("Regenerate data? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                success = run_data_generation(args)
                if not success:
                    print("Experiment failed during data generation phase!")
                    return 1
        else:
            success = run_data_generation(args)
            if not success:
                print("Experiment failed during data generation phase!")
                return 1
    
    # Exit if only generating data
    if args.generate_data_only:
        print("Data generation completed. Exiting as requested.")
        return 0
    
    experiment_dir = None
    
    # Run training phase
    if not args.skip_training:
        success, experiment_dir = run_training(args)
        if not success:
            print("Experiment failed during training phase!")
            return 1
    else:
        # If skipping training, we need to find the experiment directory
        if args.experiment_name:
            experiment_dir = Path(args.output_dir) / args.experiment_name
        else:
            print("When skipping training, you must specify --experiment-name")
            return 1
        
        if not experiment_dir.exists():
            print(f"Experiment directory not found: {experiment_dir}")
            return 1
    
    # Run evaluation phase
    if not args.skip_evaluation:
        success = run_evaluation(experiment_dir, args)
        if not success:
            print("Experiment failed during evaluation phase!")
            return 1
    
    print("\nExperiment completed successfully!")
    print(f"All results saved to: {experiment_dir}")
    
    # Print summary of generated files
    print("\nGenerated files:")
    print("- Generated training data")
    print("- Training logs and model checkpoints")
    print("- Evaluation metrics and comparison tables")
    print("- Comprehensive visualization plots")
    print("- Final comparison report")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())