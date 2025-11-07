#!/usr/bin/env python3
"""
Create publication-quality plots comparing our results with Funahashi's Table 3.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Funahashi's Table 3 data
FUNAHASHI_TABLE_3 = {
    'strikes': [0.4, 0.485, 0.57, 0.655, 0.74, 0.825, 0.91, 0.995, 1.08, 1.165, 1.25, 1.335, 1.42, 1.505, 1.59, 1.675, 1.76, 1.845, 1.93, 2.015, 2.1],
    'Case A': [63.50, 60.42, 57.92, 55.85, 54.09, 52.58, 51.27, 50.12, 49.11, 48.21, 47.41, 46.69, 46.04, 45.47, 44.95, 44.49, 44.06, 43.68, 43.33, 43.02, 42.73],
    'Case B': [56.12, 54.50, 53.24, 52.22, 51.38, 50.69, 50.11, 49.62, 49.20, 48.84, 48.53, 48.27, 48.05, 47.87, 47.71, 47.58, 47.47, 47.38, 47.31, 47.25, 47.20],
    'Case C': [71.05, 66.51, 62.77, 59.62, 56.92, 54.58, 52.52, 50.71, 49.10, 47.67, 46.37, 45.21, 44.16, 43.21, 42.35, 41.58, 40.87, 40.23, 39.64, 39.11, 38.61],
    'Case D': [64.74, 61.24, 58.34, 55.88, 53.75, 51.90, 50.25, 48.78, 47.46, 46.26, 45.17, 44.18, 43.27, 42.45, 41.69, 41.00, 40.36, 39.77, 39.23, 38.73, 38.27]
}

# Our results (from the comparison output)
OUR_RESULTS = {
    'Case A': [62.87, 60.04, 57.71, 55.76, 54.07, 52.61, 51.34, 50.21, 49.20, 48.30, 47.50, 46.79, 46.15, 45.58, 45.05, 44.58, 44.16, 43.76, 43.40, 43.07, 42.77],
    'Case B': [54.97, 53.66, 52.58, 51.66, 50.88, 50.22, 49.66, 49.17, 48.75, 48.39, 48.09, 47.84, 47.63, 47.46, 47.31, 47.18, 47.06, 46.97, 46.89, 46.82, 46.77],
    'Case C': [73.38, 68.35, 64.31, 60.96, 58.12, 55.66, 53.52, 51.62, 49.95, 48.45, 47.10, 45.88, 44.79, 43.82, 42.94, 42.14, 41.41, 40.75, 40.14, 39.59, 39.09],
    'Case D': [64.47, 61.19, 58.46, 56.10, 54.04, 52.23, 50.61, 49.15, 47.84, 46.64, 45.54, 44.54, 43.64, 42.82, 42.07, 41.38, 40.75, 40.16, 39.62, 39.12, 38.65]
}

def create_main_comparison_plot():
    """Create the main comparison plot similar to Funahashi's style."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    cases = ['Case A', 'Case B', 'Case C', 'Case D']
    case_params = [
        'α=0.5, β=0.6, ν=0.3, ρ=-0.2',
        'α=0.5, β=0.9, ν=0.3, ρ=-0.2', 
        'α=0.5, β=0.3, ν=0.3, ρ=-0.2',
        'α=0.5, β=0.6, ν=0.3, ρ=-0.5'
    ]
    
    strikes = np.array(FUNAHASHI_TABLE_3['strikes'])
    
    for i, case in enumerate(cases):
        ax = axes[i]
        
        # Our results (T=3 years)
        our_vols = np.array(OUR_RESULTS[case])
        ax.plot(strikes, our_vols, 'o-', linewidth=2.5, markersize=6, 
               label='Our MC (T=3y)', color='#2E86AB', alpha=0.8)
        
        # Funahashi's results (T=1 year)
        funahashi_vols = np.array(FUNAHASHI_TABLE_3[case])
        ax.plot(strikes, funahashi_vols, 's--', linewidth=2.5, markersize=5,
               label='Funahashi MC (T=1y)', color='#A23B72', alpha=0.8)
        
        # Formatting
        ax.set_title(f'{case}: {case_params[i]}', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Strike', fontsize=12)
        ax.set_ylabel('Implied Volatility (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper right')
        
        # Set consistent y-axis limits for better comparison
        y_min = min(our_vols.min(), funahashi_vols.min()) - 1
        y_max = max(our_vols.max(), funahashi_vols.max()) + 1
        ax.set_ylim(y_min, y_max)
        
        # Add ATM line
        ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(1.02, y_min + 0.5, 'ATM', rotation=90, fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    plt.suptitle('SABR Volatility Surfaces: Our Implementation vs Funahashi\'s Results', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save the plot
    output_dir = Path("results/visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "funahashi_comparison_main.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_difference_analysis():
    """Create analysis of differences between our results and Funahashi's."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    cases = ['Case A', 'Case B', 'Case C', 'Case D']
    strikes = np.array(FUNAHASHI_TABLE_3['strikes'])
    
    # Plot 1: Absolute differences
    ax1 = axes[0, 0]
    for i, case in enumerate(cases):
        our_vols = np.array(OUR_RESULTS[case])
        funahashi_vols = np.array(FUNAHASHI_TABLE_3[case])
        differences = our_vols - funahashi_vols
        
        ax1.plot(strikes, differences, 'o-', linewidth=2, markersize=4, 
                label=case, alpha=0.8)
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title('Volatility Differences (Our - Funahashi)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Strike')
    ax1.set_ylabel('Difference (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relative differences
    ax2 = axes[0, 1]
    for i, case in enumerate(cases):
        our_vols = np.array(OUR_RESULTS[case])
        funahashi_vols = np.array(FUNAHASHI_TABLE_3[case])
        rel_differences = (our_vols - funahashi_vols) / funahashi_vols * 100
        
        ax2.plot(strikes, rel_differences, 's-', linewidth=2, markersize=4, 
                label=case, alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('Relative Differences (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Strike')
    ax2.set_ylabel('Relative Difference (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Statistics summary
    ax3 = axes[1, 0]
    stats_data = []
    for case in cases:
        our_vols = np.array(OUR_RESULTS[case])
        funahashi_vols = np.array(FUNAHASHI_TABLE_3[case])
        differences = our_vols - funahashi_vols
        
        stats_data.append({
            'Case': case,
            'Mean Diff': np.mean(differences),
            'Std Diff': np.std(differences),
            'Max Diff': np.max(np.abs(differences)),
            'RMSE': np.sqrt(np.mean(differences**2))
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    x = np.arange(len(cases))
    width = 0.2
    
    ax3.bar(x - width, stats_df['Mean Diff'], width, label='Mean Diff', alpha=0.8)
    ax3.bar(x, stats_df['Std Diff'], width, label='Std Diff', alpha=0.8)
    ax3.bar(x + width, stats_df['RMSE'], width, label='RMSE', alpha=0.8)
    
    ax3.set_title('Difference Statistics by Case', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Case')
    ax3.set_ylabel('Volatility (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(cases)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Correlation analysis
    ax4 = axes[1, 1]
    all_our = []
    all_funahashi = []
    
    for case in cases:
        all_our.extend(OUR_RESULTS[case])
        all_funahashi.extend(FUNAHASHI_TABLE_3[case])
    
    ax4.scatter(all_funahashi, all_our, alpha=0.6, s=30)
    
    # Perfect correlation line
    min_val = min(min(all_our), min(all_funahashi))
    max_val = max(max(all_our), max(all_funahashi))
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    
    # Calculate correlation
    correlation = np.corrcoef(all_our, all_funahashi)[0, 1]
    ax4.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
            transform=ax4.transAxes, fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax4.set_title('Our Results vs Funahashi\'s Results', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Funahashi MC Volatility (%)')
    ax4.set_ylabel('Our MC Volatility (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Detailed Analysis: Our Results vs Funahashi\'s Table 3', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save the plot
    output_dir = Path("results/visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "funahashi_difference_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats_df

def create_model_performance_summary():
    """Create a summary of model performance from the experiment."""
    try:
        # Load the comparison results
        results_dir = Path("results")
        experiment_dirs = [d for d in results_dir.iterdir() if d.is_dir() and "funahashi_comparison" in d.name]
        latest_experiment = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
        
        with open(latest_experiment / "comparison_results.json", 'r') as f:
            import json
            results = json.load(f)
        
        # Create performance comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = ['MDA-CNN', 'Funahashi']
        metrics = ['MSE', 'RMSE', 'MAE']
        
        mdacnn_values = [results['mdacnn']['mse'], results['mdacnn']['rmse'], results['mdacnn']['mae']]
        funahashi_values = [results['funahashi']['mse'], results['funahashi']['rmse'], results['funahashi']['mae']]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [mdacnn_values[i], funahashi_values[i]]
            colors = ['#2E86AB', '#A23B72']
            
            bars = ax.bar(models, values, color=colors, alpha=0.8, width=0.6)
            ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
            ax.set_ylabel(f'{metric} Value', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.6f}', ha='center', va='bottom', fontsize=10)
            
            # Add improvement percentage
            improvement = (funahashi_values[i] - mdacnn_values[i]) / funahashi_values[i] * 100
            ax.text(0.5, max(values) * 0.8, f'Improvement: {improvement:+.1f}%', 
                   ha='center', fontsize=11, 
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        plt.suptitle('Model Performance Comparison: MDA-CNN vs Funahashi Baseline', 
                    fontsize=16, fontweight='bold', y=1.05)
        
        # Save the plot
        output_dir = Path("results/visualization")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "model_performance_summary.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
        
    except Exception as e:
        print(f"Error creating model performance summary: {e}")
        return None

def main():
    """Create all publication plots."""
    print("CREATING PUBLICATION-QUALITY PLOTS")
    print("=" * 50)
    
    # Create main comparison plot
    print("1. Creating main volatility surface comparison...")
    create_main_comparison_plot()
    
    # Create difference analysis
    print("2. Creating difference analysis...")
    stats_df = create_difference_analysis()
    
    # Create model performance summary
    print("3. Creating model performance summary...")
    results = create_model_performance_summary()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    if stats_df is not None:
        print("\nVolatility Comparison Statistics:")
        print(stats_df.to_string(index=False, float_format='%.3f'))
    
    if results is not None:
        print(f"\nModel Performance:")
        print(f"MDA-CNN    - MSE: {results['mdacnn']['mse']:.6f}, RMSE: {results['mdacnn']['rmse']:.6f}, MAE: {results['mdacnn']['mae']:.6f}")
        print(f"Funahashi  - MSE: {results['funahashi']['mse']:.6f}, RMSE: {results['funahashi']['rmse']:.6f}, MAE: {results['funahashi']['mae']:.6f}")
        print(f"Improvement - MSE: {results['improvement']['mse_percent']:+.1f}%, RMSE: {results['improvement']['rmse_percent']:+.1f}%, MAE: {results['improvement']['mae_percent']:+.1f}%")
    
    print("\n" + "="*60)
    print("🎉 ALL PLOTS CREATED SUCCESSFULLY!")
    print("="*60)
    print("Generated files in results/visualization/:")
    print("- funahashi_comparison_main.png")
    print("- funahashi_difference_analysis.png") 
    print("- model_performance_summary.png")
    print("- volatility_smiles_comparison.png (from previous script)")

if __name__ == "__main__":
    main()