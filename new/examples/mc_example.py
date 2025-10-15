"""
Example demonstrating SABR Monte Carlo simulation engine.

This script shows how to use the Monte Carlo generator to create
volatility surfaces and validate the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generation.sabr_params import SABRParams, GridConfig
from data_generation.sabr_mc_generator import SABRMCGenerator, MCConfig, validate_mc_accuracy


def main():
    """Run Monte Carlo example."""
    print("SABR Monte Carlo Simulation Example")
    print("=" * 40)
    
    # Define SABR parameters (typical market conditions)
    sabr_params = SABRParams(
        F0=1.0,      # Forward price
        alpha=0.3,   # Initial volatility
        beta=0.5,    # Elasticity parameter
        nu=0.4,      # Vol-of-vol
        rho=-0.2     # Correlation
    )
    
    print(f"SABR Parameters:")
    print(f"  F0 = {sabr_params.F0}")
    print(f"  α = {sabr_params.alpha}")
    print(f"  β = {sabr_params.beta}")
    print(f"  ν = {sabr_params.nu}")
    print(f"  ρ = {sabr_params.rho}")
    
    # Define grid configuration
    grid_config = GridConfig(
        maturity_range=(1.0, 5.0),
        n_strikes=15,
        n_maturities=5,
        include_extended_wings=False
    )
    
    print(f"\nGrid Configuration:")
    print(f"  Maturity range: {grid_config.maturity_range}")
    print(f"  Number of strikes: {grid_config.n_strikes}")
    print(f"  Number of maturities: {grid_config.n_maturities}")
    
    # Configure Monte Carlo simulation
    mc_config = MCConfig(
        n_paths=50000,
        n_steps=200,
        random_seed=42,
        use_antithetic=True,
        convergence_check=True
    )
    
    print(f"\nMonte Carlo Configuration:")
    print(f"  Number of paths: {mc_config.n_paths}")
    print(f"  Number of time steps: {mc_config.n_steps}")
    print(f"  Use antithetic variates: {mc_config.use_antithetic}")
    
    # Generate volatility surface
    print(f"\nGenerating volatility surface...")
    generator = SABRMCGenerator(mc_config)
    
    result = generator.generate_volatility_surface(sabr_params, grid_config)
    
    print(f"Surface generation completed in {result.computation_time:.2f} seconds")
    
    # Display results
    print(f"\nResults Summary:")
    print(f"  Surface shape: {result.volatility_surface.shape}")
    print(f"  Valid volatilities: {np.sum(~np.isnan(result.volatility_surface))}")
    
    # Show convergence information
    print(f"\nConvergence Information:")
    overall_info = result.convergence_info['overall']
    print(f"  Total strikes computed: {overall_info['total_strikes']}")
    print(f"  Valid volatilities: {overall_info['total_valid_volatilities']}")
    print(f"  Convergence achieved: {overall_info['convergence_achieved']}")
    
    # Validate accuracy
    print(f"\nValidating Monte Carlo accuracy...")
    validation = validate_mc_accuracy(sabr_params, result, tolerance=0.1)
    
    print(f"Validation passed: {validation['passed']}")
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    print(f"\nValidation checks:")
    for check in validation['checks']:
        status = "✓" if check['passed'] else "✗"
        print(f"  {status} {check['name']}")
    
    # Display volatility smile for first maturity
    print(f"\nVolatility Smile (T = {result.maturities[0]:.1f} years):")
    strikes_0 = result.strikes[0]
    vols_0 = result.volatility_surface[0]
    
    valid_mask = ~np.isnan(strikes_0) & ~np.isnan(vols_0)
    valid_strikes = strikes_0[valid_mask]
    valid_vols = vols_0[valid_mask]
    
    if len(valid_strikes) > 0:
        print("  Strike    Volatility   Moneyness")
        print("  ------    ----------   ---------")
        for k, vol in zip(valid_strikes, valid_vols):
            moneyness = k / sabr_params.F0
            print(f"  {k:6.3f}    {vol:8.4f}     {moneyness:6.3f}")
    
    # Create simple visualization if matplotlib is available
    try:
        create_visualization(result, sabr_params)
        print(f"\nVisualization saved as 'mc_example_surface.png'")
    except ImportError:
        print(f"\nMatplotlib not available for visualization")
    except Exception as e:
        print(f"\nVisualization failed: {e}")


def create_visualization(result, sabr_params):
    """Create simple visualization of the volatility surface."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot volatility smile for first maturity
    strikes_0 = result.strikes[0]
    vols_0 = result.volatility_surface[0]
    
    valid_mask = ~np.isnan(strikes_0) & ~np.isnan(vols_0)
    valid_strikes = strikes_0[valid_mask]
    valid_vols = vols_0[valid_mask]
    
    if len(valid_strikes) > 0:
        moneyness = valid_strikes / sabr_params.F0
        ax1.plot(moneyness, valid_vols, 'bo-', linewidth=2, markersize=6)
        ax1.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='ATM')
        ax1.set_xlabel('Moneyness (K/F)')
        ax1.set_ylabel('Implied Volatility')
        ax1.set_title(f'Volatility Smile (T = {result.maturities[0]:.1f}y)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Plot volatility surface (if we have multiple maturities)
    if len(result.maturities) > 1:
        # Create meshgrid for surface plot
        valid_data = []
        for i, maturity in enumerate(result.maturities):
            strikes_i = result.strikes[i]
            vols_i = result.volatility_surface[i]
            
            valid_mask = ~np.isnan(strikes_i) & ~np.isnan(vols_i)
            if np.sum(valid_mask) > 0:
                valid_strikes = strikes_i[valid_mask]
                valid_vols = vols_i[valid_mask]
                
                for k, vol in zip(valid_strikes, valid_vols):
                    moneyness = k / sabr_params.F0
                    valid_data.append([maturity, moneyness, vol])
        
        if valid_data:
            data = np.array(valid_data)
            scatter = ax2.scatter(data[:, 0], data[:, 1], c=data[:, 2], 
                                cmap='viridis', s=50, alpha=0.7)
            ax2.set_xlabel('Maturity (years)')
            ax2.set_ylabel('Moneyness (K/F)')
            ax2.set_title('Volatility Surface')
            plt.colorbar(scatter, ax=ax2, label='Implied Volatility')
    
    plt.tight_layout()
    plt.savefig('mc_example_surface.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()