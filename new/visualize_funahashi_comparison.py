#!/usr/bin/env python3
"""
Visualize results comparison with Funahashi's Table 3.

This script creates comprehensive visualizations comparing our trained models
with Funahashi's published results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import torch
from pathlib import Path
import json

# Add project root to path
import sys
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.mdacnn_model import MDACNNModel
from models.baseline_models import FunahashiBaselineModel
from preprocessing.data_loader import SABRDataset

# Funahashi's Table 3 data
FUNAHASHI_TABLE_3 = {
    'strikes': [0.4, 0.485, 0.57, 0.655, 0.74, 0.825, 0.91, 0.995, 1.08, 1.165, 1.25, 1.335, 1.42, 1.505, 1.59, 1.675, 1.76, 1.845, 1.93, 2.015, 2.1],
    'Case A': [63.50, 60.42, 57.92, 55.85, 54.09, 52.58, 51.27, 50.12, 49.11, 48.21, 47.41, 46.69, 46.04, 45.47, 44.95, 44.49, 44.06, 43.68, 43.33, 43.02, 42.73],
    'Case B': [56.12, 54.50, 53.24, 52.22, 51.38, 50.69, 50.11, 49.62, 49.20, 48.84, 48.53, 48.27, 48.05, 47.87, 47.71, 47.58, 47.47, 47.38, 47.31, 47.25, 47.20],
    'Case C': [71.05, 66.51, 62.77, 59.62, 56.92, 54.58, 52.52, 50.71, 49.10, 47.67, 46.37, 45.21, 44.16, 43.21, 42.35, 41.58, 40.87, 40.23, 39.64, 39.11, 38.61],
    'Case D': [64.74, 61.24, 58.34, 55.88, 53.75, 51.90, 50.25, 48.78, 47.46, 46.26, 45.17, 44.18, 43.27, 42.45, 41.69, 41.00, 40.36, 39.77, 39.23, 38.73, 38.27]
}

def load_trained_models(experiment_dir):
    """Load the trained models."""
    experiment_path = Path(experiment_dir)
    
    # Load model configurations (infer from data)
    data_dir = "data_funahashi_comparison/processed"
    dataset = SABRDataset(data_dir, split='train')
    sample_patches, sample_features, _ = dataset[0]
    
    patch_size = sample_patches.shape[-1]
    n_features = len(sample_features)
    
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
    
    # Load trained weights
    device = torch.device('cpu')  # Use CPU for inference
    mdacnn_model.load_state_dict(torch.load(experiment_path / "mda-cnn_final.pth", map_location=device))
    funahashi_model.load_state_dict(torch.load(experiment_path / "funahashi_final.pth", map_location=device))
    
    mdacnn_model.eval()
    funahashi_model.eval()
    
    return mdacnn_model, funahashi_model

def load_our_mc_data():
    """Load our MC data for comparison."""
    # Find the most recent run directory
    raw_dir = Path("data_funahashi_comparison/raw")
    run_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    
    if not run_dirs:
        raise FileNotFoundError("No run directories found")
    
    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
    
    # Load MC results
    with open(latest_run / "mc_results.pkl", 'rb') as f:
        mc_results = pickle.load(f)
    
    # Load Hagan results
    with open(latest_run / "hagan_results.pkl", 'rb') as f:
        hagan_results = pickle.load(f)
    
    # Load parameter sets
    with open(latest_run / "parameter_sets.pkl", 'rb') as f:
        parameter_sets = pickle.load(f)
    
    return mc_results, hagan_results, parameter_sets

def get_model_predictions(mdacnn_model, funahashi_model):
    """Get predictions from both trained models."""
    data_dir = "data_funahashi_comparison/processed"
    dataset = SABRDataset(data_dir, split='train')
    
    mdacnn_predictions = []
    funahashi_predictions = []
    hagan_volatilities = []
    
    device = torch.device('cpu')
    
    with torch.no_grad():
        for i in range(len(dataset)):
            patches, features, target = dataset[i]
            
            # Convert to tensors and add batch dimension
            patches = torch.tensor(patches).float().unsqueeze(0).to(device)
            features = torch.tensor(features).float().unsqueeze(0).to(device)
            
            # Get predictions
            mdacnn_pred = mdacnn_model(patches, features).cpu().numpy()[0, 0]
            funahashi_pred = funahashi_model(features).cpu().numpy()[0, 0]
            
            mdacnn_predictions.append(mdacnn_pred)
            funahashi_predictions.append(funahashi_pred)
            
            # The target is the residual, so we need to add it back to get the volatility
            # We'll approximate the Hagan volatility from the data
            hagan_vol = target - mdacnn_pred  # Approximate
            hagan_volatilities.append(hagan_vol)
    
    return np.array(mdacnn_predictions), np.array(funahashi_predictions), np.array(hagan_volatilities)

def create_comparison_table():
    """Create a comparison table similar to Funahashi's Table 3."""
    try:
        # Load our MC data
        mc_results, hagan_results, parameter_sets = load_our_mc_data()
        
        # Create comparison data
        case_names = ["Case A", "Case B", "Case C", "Case D"]
        comparison_data = []
        
        for i, (mc_result, hagan_result, params) in enumerate(zip(mc_results, hagan_results, parameter_sets)):
            case_name = case_names[i]
            
            # Get our MC volatilities
            mc_surface = mc_result.volatility_surface
            if mc_surface.ndim == 2:
                our_mc_vols = mc_surface[0, :] * 100  # Convert to percentage
            else:
                our_mc_vols = mc_surface * 100
            
            # Get our Hagan volatilities
            hagan_surface = hagan_result.volatility_surface
            if hagan_surface.ndim == 2:
                our_hagan_vols = hagan_surface[0, :] * 100
            else:
                our_hagan_vols = hagan_surface * 100
            
            # Get strikes
            strikes = mc_result.strikes
            if strikes.ndim == 2:
                strikes = strikes[0, :]
            
            # Get Funahashi's data
            funahashi_mc_vols = np.array(FUNAHASHI_TABLE_3[case_name])
            funahashi_strikes = np.array(FUNAHASHI_TABLE_3['strikes'])
            
            # Store data for this case
            for j, (strike, our_mc, our_hagan, funa_mc) in enumerate(zip(strikes, our_mc_vols, our_hagan_vols, funahashi_mc_vols)):
                if j < len(funahashi_strikes):
                    comparison_data.append({
                        'Case': case_name,
                        'Strike': strike,
                        'Our_MC': our_mc,
                        'Our_Hagan': our_hagan,
                        'Our_Residual': our_mc - our_hagan,
                        'Funahashi_MC': funa_mc,
                        'Funahashi_Strike': funahashi_strikes[j],
                        'MC_Difference': our_mc - funa_mc
                    })
        
        return pd.DataFrame(comparison_data)
        
    except Exception as e:
        print(f"Error creating comparison table: {e}")
        return None

def plot_volatility_smiles_comparison(df):
    """Plot volatility smiles comparing our results with Funahashi's."""
    if df is None:
        print("No data available for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    cases = ['Case A', 'Case B', 'Case C', 'Case D']
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, case in enumerate(cases):
        ax = axes[i]
        case_data = df[df['Case'] == case]
        
        if len(case_data) > 0:
            # Plot our MC results
            ax.plot(case_data['Strike'], case_data['Our_MC'], 
                   'o-', color=colors[i], label=f'Our MC (T=3y)', linewidth=2, markersize=6)
            
            # Plot Funahashi's MC results
            ax.plot(case_data['Funahashi_Strike'], case_data['Funahashi_MC'], 
                   's--', color='black', label=f'Funahashi MC (T=1y)', linewidth=2, markersize=4)
            
            # Plot our Hagan results
            ax.plot(case_data['Strike'], case_data['Our_Hagan'], 
                   '^:', color='gray', label=f'Our Hagan', linewidth=1, markersize=4)
        
        ax.set_title(f'{case}: α=0.5, β={0.6 if "A" in case or "D" in case else (0.9 if "B" in case else 0.3)}, ν=0.3, ρ={-0.2 if "A" in case or "B" in case or "C" in case else -0.5}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Strike', fontsize=10)
        ax.set_ylabel('Implied Volatility (%)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Set reasonable y-axis limits
        if len(case_data) > 0:
            y_min = min(case_data['Our_MC'].min(), case_data['Funahashi_MC'].min()) - 2
            y_max = max(case_data['Our_MC'].max(), case_data['Funahashi_MC'].max()) + 2
            ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.suptitle('Volatility Smiles: Our Results vs Funahashi\'s Table 3', 
                fontsize=16, fontweight='bold', y=1.02)
    
    # Save the plot
    output_dir = Path("results/visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "volatility_smiles_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_predictions_comparison(experiment_dir):
    """Plot comparison of model predictions."""
    try:
        # Load trained models
        mdacnn_model, funahashi_model = load_trained_models(experiment_dir)
        
        # Get model predictions
        mdacnn_preds, funahashi_preds, hagan_vols = get_model_predictions(mdacnn_model, funahashi_model)
        
        # Load actual targets
        data_dir = "data_funahashi_comparison/processed"
        dataset = SABRDataset(data_dir, split='train')
        actual_residuals = [dataset[i][2] for i in range(len(dataset))]
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Residual predictions comparison
        axes[0, 0].scatter(actual_residuals, mdacnn_preds, alpha=0.6, label='MDA-CNN', color='blue')
        axes[0, 0].scatter(actual_residuals, funahashi_preds, alpha=0.6, label='Funahashi', color='red')
        axes[0, 0].plot([-0.1, 0.1], [-0.1, 0.1], 'k--', alpha=0.5, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Residuals')
        axes[0, 0].set_ylabel('Predicted Residuals')
        axes[0, 0].set_title('Residual Predictions: Actual vs Predicted')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Prediction errors
        mdacnn_errors = np.array(mdacnn_preds) - np.array(actual_residuals)
        funahashi_errors = np.array(funahashi_preds) - np.array(actual_residuals)
        
        axes[0, 1].hist(mdacnn_errors, bins=20, alpha=0.6, label='MDA-CNN Errors', color='blue')
        axes[0, 1].hist(funahashi_errors, bins=20, alpha=0.6, label='Funahashi Errors', color='red')
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Prediction Errors')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Model comparison metrics
        mdacnn_mse = np.mean(mdacnn_errors**2)
        funahashi_mse = np.mean(funahashi_errors**2)
        mdacnn_mae = np.mean(np.abs(mdacnn_errors))
        funahashi_mae = np.mean(np.abs(funahashi_errors))
        
        metrics = ['MSE', 'MAE']
        mdacnn_values = [mdacnn_mse, mdacnn_mae]
        funahashi_values = [funahashi_mse, funahashi_mae]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, mdacnn_values, width, label='MDA-CNN', color='blue', alpha=0.7)
        axes[1, 0].bar(x + width/2, funahashi_values, width, label='Funahashi', color='red', alpha=0.7)
        axes[1, 0].set_xlabel('Metrics')
        axes[1, 0].set_ylabel('Error Value')
        axes[1, 0].set_title('Model Performance Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(metrics)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Residuals vs predictions scatter
        axes[1, 1].scatter(range(len(actual_residuals)), actual_residuals, alpha=0.6, label='Actual', color='black', s=20)
        axes[1, 1].scatter(range(len(mdacnn_preds)), mdacnn_preds, alpha=0.6, label='MDA-CNN', color='blue', s=15)
        axes[1, 1].scatter(range(len(funahashi_preds)), funahashi_preds, alpha=0.6, label='Funahashi', color='red', s=15)
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Residual Value')
        axes[1, 1].set_title('Residual Predictions by Sample')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Model Predictions Analysis', fontsize=16, fontweight='bold', y=1.02)
        
        # Save the plot
        output_dir = Path("results/visualization")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "model_predictions_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        print(f"MDA-CNN:")
        print(f"  MSE: {mdacnn_mse:.8f}")
        print(f"  MAE: {mdacnn_mae:.8f}")
        print(f"  RMSE: {np.sqrt(mdacnn_mse):.8f}")
        print(f"\nFunahashi Baseline:")
        print(f"  MSE: {funahashi_mse:.8f}")
        print(f"  MAE: {funahashi_mae:.8f}")
        print(f"  RMSE: {np.sqrt(funahashi_mse):.8f}")
        print(f"\nImprovement (MDA-CNN vs Funahashi):")
        print(f"  MSE: {((funahashi_mse - mdacnn_mse) / funahashi_mse * 100):+.2f}%")
        print(f"  MAE: {((funahashi_mae - mdacnn_mae) / funahashi_mae * 100):+.2f}%")
        
    except Exception as e:
        print(f"Error plotting model predictions: {e}")
        import traceback
        traceback.print_exc()

def create_funahashi_style_table(df):
    """Create a table in Funahashi's style."""
    if df is None:
        print("No data available for table creation")
        return
    
    print("\n" + "="*100)
    print("COMPARISON WITH FUNAHASHI'S TABLE 3")
    print("="*100)
    print("Our data: T = 3.0 years | Funahashi's data: T = 1.0 year")
    print("Note: Volatility differences expected due to different maturities")
    print()
    
    # Create a formatted table
    cases = ['Case A', 'Case B', 'Case C', 'Case D']
    
    for case in cases:
        case_data = df[df['Case'] == case].sort_values('Strike')
        if len(case_data) == 0:
            continue
            
        print(f"\n{case}:")
        print("-" * 80)
        print(f"{'Strike':<8} {'Our MC':<10} {'Our Hagan':<12} {'Our Residual':<14} {'Funahashi MC':<14} {'Difference':<10}")
        print("-" * 80)
        
        for _, row in case_data.iterrows():
            print(f"{row['Strike']:<8.3f} {row['Our_MC']:<10.2f} {row['Our_Hagan']:<12.2f} "
                  f"{row['Our_Residual']:<14.4f} {row['Funahashi_MC']:<14.2f} {row['MC_Difference']:<10.2f}")

def main():
    """Main visualization function."""
    print("FUNAHASHI COMPARISON VISUALIZATION")
    print("=" * 50)
    
    # Find the most recent experiment
    results_dir = Path("results")
    experiment_dirs = [d for d in results_dir.iterdir() if d.is_dir() and "funahashi_comparison" in d.name]
    
    if not experiment_dirs:
        print("No experiment results found!")
        return
    
    latest_experiment = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Using experiment: {latest_experiment.name}")
    
    try:
        # Create comparison table
        print("\n1. Creating comparison table...")
        df = create_comparison_table()
        
        if df is not None:
            # Print Funahashi-style table
            create_funahashi_style_table(df)
            
            # Plot volatility smiles comparison
            print("\n2. Creating volatility smiles comparison...")
            plot_volatility_smiles_comparison(df)
        
        # Plot model predictions comparison
        print("\n3. Creating model predictions analysis...")
        plot_model_predictions_comparison(latest_experiment)
        
        print("\n" + "="*50)
        print("🎉 VISUALIZATION COMPLETED!")
        print("="*50)
        print("Generated plots:")
        print("- results/visualization/volatility_smiles_comparison.png")
        print("- results/visualization/model_predictions_comparison.png")
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()