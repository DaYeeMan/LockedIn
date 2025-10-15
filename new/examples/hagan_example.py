"""
Example usage of Hagan analytical SABR volatility surface generator.

This script demonstrates how to use the HaganSurfaceGenerator to create
analytical volatility surfaces and validate them against known properties.
"""

import sys
import os
# Add parent directory to path to import from data_generation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data_generation.hagan_surface_generator import HaganSurfaceGenerator, HaganConfig
from data_generation.sabr_params import SABRParams, GridConfig


def main():
    """Main example function."""
    print("Hagan Analytical SABR Surface Generator Example")
    print("=" * 50)
    
    # Create SABR parameters
    sabr_params = SABRParams(
        F0=1.0,      # Forward price
        alpha=0.2,   # Initial volatility
        beta=0.5,    # Elasticity parameter
        nu=0.4,      # Vol-of-vol
        rho=-0.3     # Correlation
    )
    
    print(f"SABR Parameters:")
    print(f"  F0 (Forward): {sabr_params.F0}")
    print(f"  α (Alpha): {sabr_params.alpha}")
    print(f"  β (Beta): {sabr_params.beta}")
    print(f"  ν (Nu): {sabr_params.nu}")
    print(f"  ρ (Rho): {sabr_params.rho}")
    print()
    
    # Create grid configuration
    grid_config = GridConfig(
        maturity_range=(0.5, 3.0),
        n_strikes=21,
        n_maturities=5,
        include_extended_wings=False
    )
    
    print(f"Grid Configuration:")
    print(f"  Maturity range: {grid_config.maturity_range}")
    print(f"  Number of strikes: {grid_config.n_strikes}")
    print(f"  Number of maturities: {grid_config.n_maturities}")
    print()
    
    # Create Hagan generator
    hagan_config = HaganConfig(
        use_pde_correction=True,
        numerical_tolerance=1e-12
    )
    generator = HaganSurfaceGenerator(hagan_config)
    
    # Generate volatility surface
    print("Generating Hagan analytical surface...")
    result = generator.generate_volatility_surface(sabr_params, grid_config)
    
    print(f"Surface generation completed in {result.computation_time:.4f} seconds")
    print(f"Grid shape: {result.volatility_surface.shape}")
    print(f"Number of warnings: {len(result.numerical_warnings)}")
    
    if result.numerical_warnings:
        print("Warnings:")
        for warning in result.numerical_warnings:
            print(f"  - {warning}")
    print()
    
    # Validate the surface
    print("Validating surface...")
    validation = generator.validate_surface(result, sabr_params)
    
    print(f"Validation passed: {validation['passed']}")
    print(f"Number of checks: {len(validation['checks'])}")
    
    for check in validation['checks']:
        status = "✓" if check['passed'] else "✗"
        print(f"  {status} {check['name']}")
    
    if validation['warnings']:
        print("Validation warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    print()
    
    # Display some surface statistics
    valid_vols = result.volatility_surface[~np.isnan(result.volatility_surface)]
    
    print("Surface Statistics:")
    print(f"  Total grid points: {result.volatility_surface.size}")
    print(f"  Valid volatilities: {len(valid_vols)}")
    print(f"  Min volatility: {np.min(valid_vols):.4f}")
    print(f"  Max volatility: {np.max(valid_vols):.4f}")
    print(f"  Mean volatility: {np.mean(valid_vols):.4f}")
    print(f"  Std volatility: {np.std(valid_vols):.4f}")
    print()
    
    # Show ATM volatilities across maturities
    print("ATM Volatilities by Maturity:")
    for i, maturity in enumerate(result.maturities):
        strikes = result.strikes[i]
        vols = result.volatility_surface[i]
        
        valid_mask = ~np.isnan(strikes) & ~np.isnan(vols)
        if np.any(valid_mask):
            valid_strikes = strikes[valid_mask]
            valid_vols = vols[valid_mask]
            
            # Find ATM (closest to forward)
            atm_idx = np.argmin(np.abs(valid_strikes - sabr_params.F0))
            atm_strike = valid_strikes[atm_idx]
            atm_vol = valid_vols[atm_idx]
            
            print(f"  T={maturity:.2f}: K={atm_strike:.3f}, σ={atm_vol:.4f}")
    print()
    
    # Demonstrate single volatility calculation
    print("Single Volatility Calculation Example:")
    test_strikes = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    test_maturity = 1.0
    
    test_vols = generator.hagan_volatility(sabr_params, test_strikes, test_maturity)
    
    print(f"Maturity: {test_maturity}")
    for strike, vol in zip(test_strikes, test_vols):
        moneyness = strike / sabr_params.F0
        print(f"  K={strike:.1f} (m={moneyness:.2f}): σ={vol:.4f}")
    print()
    
    # Test different configurations
    print("Testing Different Configurations:")
    
    # Without PDE correction
    config_no_pde = HaganConfig(use_pde_correction=False)
    gen_no_pde = HaganSurfaceGenerator(config_no_pde)
    
    atm_vol_with_pde = generator.hagan_volatility(sabr_params, np.array([1.0]), 1.0)[0]
    atm_vol_no_pde = gen_no_pde.hagan_volatility(sabr_params, np.array([1.0]), 1.0)[0]
    
    print(f"ATM vol with PDE correction: {atm_vol_with_pde:.6f}")
    print(f"ATM vol without PDE correction: {atm_vol_no_pde:.6f}")
    print(f"PDE correction effect: {((atm_vol_with_pde/atm_vol_no_pde - 1) * 100):.3f}%")
    print()
    
    # Test extreme parameters
    print("Testing Edge Cases:")
    
    # High correlation
    extreme_params = SABRParams(F0=1.0, alpha=0.3, beta=0.7, nu=0.6, rho=0.7)
    extreme_vol = generator.hagan_volatility(extreme_params, np.array([1.0]), 1.0)[0]
    print(f"High correlation (ρ=0.7) ATM vol: {extreme_vol:.4f}")
    
    # Low vol-of-vol
    low_nu_params = SABRParams(F0=1.0, alpha=0.2, beta=0.5, nu=0.05, rho=0.0)
    low_nu_vol = generator.hagan_volatility(low_nu_params, np.array([1.0]), 1.0)[0]
    print(f"Low vol-of-vol (ν=0.05) ATM vol: {low_nu_vol:.4f}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()