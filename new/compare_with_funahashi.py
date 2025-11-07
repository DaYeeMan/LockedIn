#!/usr/bin/env python3
"""
Compare our generated MC values with Funahashi's Table 3.

This script loads our generated data and compares the MC implied volatilities
with Funahashi's published results.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Funahashi's Table 3 data (MC implied volatilities)
FUNAHASHI_TABLE_3 = {
    'strikes': [0.4, 0.485, 0.57, 0.655, 0.74, 0.825, 0.91, 0.995, 1.08, 1.165, 1.25, 1.335, 1.42, 1.505, 1.59, 1.675, 1.76, 1.845, 1.93, 2.015, 2.1],
    'Case A': [63.50, 60.42, 57.92, 55.85, 54.09, 52.58, 51.27, 50.12, 49.11, 48.21, 47.41, 46.69, 46.04, 45.47, 44.95, 44.49, 44.06, 43.68, 43.33, 43.02, 42.73],
    'Case B': [56.12, 54.50, 53.24, 52.22, 51.38, 50.69, 50.11, 49.62, 49.20, 48.84, 48.53, 48.27, 48.05, 47.87, 47.71, 47.58, 47.47, 47.38, 47.31, 47.25, 47.20],
    'Case C': [71.05, 66.51, 62.77, 59.62, 56.92, 54.58, 52.52, 50.71, 49.10, 47.67, 46.37, 45.21, 44.16, 43.21, 42.35, 41.58, 40.87, 40.23, 39.64, 39.11, 38.61],
    'Case D': [64.74, 61.24, 58.34, 55.88, 53.75, 51.90, 50.25, 48.78, 47.46, 46.26, 45.17, 44.18, 43.27, 42.45, 41.69, 41.00, 40.36, 39.77, 39.23, 38.73, 38.27]
}

def load_our_results():
    """Load our generated MC results."""
    # Find the most recent run directory
    raw_dir = Path("data_funahashi_comparison/raw")
    run_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    
    if not run_dirs:
        raise FileNotFoundError("No run directories found")
    
    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
    
    # Load MC results
    with open(latest_run / "mc_results.pkl", 'rb') as f:
        mc_results = pickle.load(f)
    
    # Load parameter sets
    with open(latest_run / "parameter_sets.pkl", 'rb') as f:
        parameter_sets = pickle.load(f)
    
    return mc_results, parameter_sets

def extract_mc_volatilities(mc_results, parameter_sets):
    """Extract MC implied volatilities for comparison."""
    case_names = ["Case A", "Case B", "Case C", "Case D"]
    our_results = {}
    
    for i, (mc_result, params) in enumerate(zip(mc_results, parameter_sets)):
        case_name = case_names[i]
        
        # Get volatility surface (should be 1 maturity × 21 strikes)
        vol_surface = mc_result.volatility_surface
        
        if vol_surface.ndim == 2:
            # Take first (and only) maturity
            volatilities = vol_surface[0, :] * 100  # Convert to percentage
        else:
            volatilities = vol_surface * 100
        
        # Get strikes
        strikes = mc_result.strikes
        if strikes.ndim == 2:
            strikes = strikes[0, :]
        
        our_results[case_name] = {
            'strikes': strikes,
            'volatilities': volatilities,
            'params': params
        }
    
    return our_results

def compare_results():
    """Compare our results with Funahashi's Table 3."""
    print("COMPARISON WITH FUNAHASHI'S TABLE 3")
    print("=" * 60)
    
    try:
        # Load our results
        mc_results, parameter_sets = load_our_results()
        our_results = extract_mc_volatilities(mc_results, parameter_sets)
        
        print(f"Our data: T = 3.0 years")
        print(f"Funahashi's data: T = 1.0 year")
        print(f"Note: Different maturities will give different volatility levels")
        print()
        
        # Compare each case
        for case_name in ["Case A", "Case B", "Case C", "Case D"]:
            print(f"\n{case_name}:")
            print("-" * 40)
            
            # Get our data
            our_data = our_results[case_name]
            our_strikes = our_data['strikes']
            our_vols = our_data['volatilities']
            
            # Get Funahashi's data
            funahashi_strikes = np.array(FUNAHASHI_TABLE_3['strikes'])
            funahashi_vols = np.array(FUNAHASHI_TABLE_3[case_name])
            
            # Print comparison table
            print(f"{'Strike':<8} {'Our MC (T=3)':<12} {'Funahashi (T=1)':<15} {'Difference':<10}")
            print("-" * 50)
            
            # Find matching strikes
            for i, strike in enumerate(funahashi_strikes):
                # Find closest strike in our data
                if i < len(our_strikes) and i < len(our_vols):
                    our_vol = our_vols[i]
                    funahashi_vol = funahashi_vols[i]
                    diff = our_vol - funahashi_vol
                    
                    print(f"{strike:<8.3f} {our_vol:<12.2f} {funahashi_vol:<15.2f} {diff:<10.2f}")
            
            # Calculate statistics
            if len(our_vols) >= len(funahashi_vols):
                our_subset = our_vols[:len(funahashi_vols)]
                mean_diff = np.mean(our_subset - funahashi_vols)
                std_diff = np.std(our_subset - funahashi_vols)
                
                print(f"\nStatistics for {case_name}:")
                print(f"  Mean difference: {mean_diff:.2f}%")
                print(f"  Std difference:  {std_diff:.2f}%")
                print(f"  Our ATM vol:     {our_subset[len(our_subset)//2]:.2f}%")
                print(f"  Funahashi ATM:   {funahashi_vols[len(funahashi_vols)//2]:.2f}%")
        
        print("\n" + "=" * 60)
        print("ANALYSIS:")
        print("=" * 60)
        print("1. Our data uses T=3 years vs Funahashi's T=1 year")
        print("2. Longer maturity typically gives different volatility levels")
        print("3. The SABR model parameters are identical")
        print("4. Differences are expected due to maturity difference")
        print("5. The volatility smile shapes should be similar")
        
        return True
        
    except Exception as e:
        print(f"Error comparing results: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main comparison function."""
    success = compare_results()
    
    if success:
        print("\n✅ Comparison completed successfully!")
        print("\nTo generate data with T=1 year for exact comparison:")
        print("1. Update maturity_range to [1.0, 1.01] in config")
        print("2. Regenerate data")
    else:
        print("\n❌ Comparison failed!")

if __name__ == "__main__":
    main()