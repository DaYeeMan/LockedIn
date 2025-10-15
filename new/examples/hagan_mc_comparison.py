"""
Quick comparison between Hagan analytical and Monte Carlo SABR surfaces.

This script demonstrates that both generators work with the same parameter sets
and produce reasonable results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data_generation.hagan_surface_generator import HaganSurfaceGenerator
from data_generation.sabr_mc_generator import SABRMCGenerator, MCConfig
from data_generation.sabr_params import SABRParams, GridConfig


def main():
    """Compare Hagan and MC surfaces."""
    print("Hagan vs Monte Carlo SABR Surface Comparison")
    print("=" * 45)
    
    # Create test parameters
    sabr_params = SABRParams(F0=1.0, alpha=0.2, beta=0.5, nu=0.3, rho=-0.2)
    
    # Simple grid for quick comparison
    grid_config = GridConfig(
        maturity_range=(1.0, 1.1),  # Small range for single maturity
        n_strikes=5,
        n_maturities=1
    )
    
    print(f"SABR Parameters: α={sabr_params.alpha}, β={sabr_params.beta}, ν={sabr_params.nu}, ρ={sabr_params.rho}")
    print()
    
    # Generate Hagan surface
    print("Generating Hagan analytical surface...")
    hagan_gen = HaganSurfaceGenerator()
    hagan_result = hagan_gen.generate_volatility_surface(sabr_params, grid_config)
    
    print(f"Hagan surface generated in {hagan_result.computation_time:.4f} seconds")
    
    # Generate MC surface (small number of paths for speed)
    print("Generating Monte Carlo surface...")
    mc_config = MCConfig(n_paths=10000, n_steps=100, random_seed=42)
    mc_gen = SABRMCGenerator(mc_config)
    mc_result = mc_gen.generate_volatility_surface(sabr_params, grid_config)
    
    print(f"MC surface generated in {mc_result.computation_time:.4f} seconds")
    print()
    
    # Compare results
    print("Volatility Comparison:")
    print("Strike    Hagan     MC       Diff     Rel.Err")
    print("-" * 45)
    
    hagan_strikes = hagan_result.strikes[0]
    hagan_vols = hagan_result.volatility_surface[0]
    mc_strikes = mc_result.strikes[0]
    mc_vols = mc_result.volatility_surface[0]
    
    # Find common strikes (should be the same)
    for i in range(min(len(hagan_strikes), len(mc_strikes))):
        if not np.isnan(hagan_strikes[i]) and not np.isnan(mc_strikes[i]):
            h_vol = hagan_vols[i]
            m_vol = mc_vols[i]
            
            if not np.isnan(h_vol) and not np.isnan(m_vol):
                diff = h_vol - m_vol
                rel_err = abs(diff) / h_vol * 100
                
                print(f"{hagan_strikes[i]:.3f}    {h_vol:.4f}   {m_vol:.4f}   {diff:+.4f}   {rel_err:.1f}%")
    
    print()
    print("Comparison completed successfully!")
    print("Note: Some differences are expected due to MC sampling error.")


if __name__ == "__main__":
    main()