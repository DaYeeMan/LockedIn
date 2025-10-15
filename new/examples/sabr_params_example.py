"""
Example usage of SABR parameter and grid configuration classes.

This script demonstrates how to use the SABRParams, GridConfig, and 
ParameterSampler classes for SABR volatility surface modeling.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_generation'))

import numpy as np
import matplotlib.pyplot as plt
from sabr_params import SABRParams, GridConfig, ParameterSampler


def demonstrate_sabr_params():
    """Demonstrate SABRParams usage."""
    print("=== SABR Parameters Demo ===")
    
    # Create valid SABR parameters
    params = SABRParams(F0=100.0, alpha=0.3, beta=0.5, nu=0.4, rho=0.2)
    print(f"Created SABR params: {params}")
    print(f"Parameters dict: {params.to_dict()}")
    
    # Demonstrate validation
    try:
        invalid_params = SABRParams(F0=100.0, alpha=0.7, beta=0.5, nu=0.4, rho=0.2)
    except ValueError as e:
        print(f"Validation caught invalid alpha: {e}")
    
    print()


def demonstrate_grid_config():
    """Demonstrate GridConfig usage."""
    print("=== Grid Configuration Demo ===")
    
    # Create grid configuration
    config = GridConfig(n_strikes=21, n_maturities=5)
    params = SABRParams(F0=100.0, alpha=0.3, beta=0.5, nu=0.4, rho=0.2)
    
    # Generate strike range for different maturities
    maturities = [1.0, 2.0, 5.0]
    print("Funahashi strike ranges:")
    for T in maturities:
        K1, K2 = config.calculate_strike_range(params, T)
        print(f"  T={T:.1f}y: K1={K1:.2f}, K2={K2:.2f}")
    
    # Generate strike grid
    strikes = config.generate_strike_grid(params, maturity=2.0)
    print(f"\nStrike grid (T=2y): {len(strikes)} strikes")
    print(f"  Range: {strikes[0]:.2f} to {strikes[-1]:.2f}")
    
    # Compare with extended wings
    config_extended = GridConfig(n_strikes=21, include_extended_wings=True)
    strikes_extended = config_extended.generate_strike_grid(params, maturity=2.0)
    print(f"\nExtended wing strikes: {len(strikes_extended)} strikes")
    print(f"  Range: {strikes_extended[0]:.2f} to {strikes_extended[-1]:.2f}")
    print(f"  Additional wing strikes: 0.3f={0.3*params.F0:.1f}, 2.5f={2.5*params.F0:.1f}")
    
    # Generate full grid
    K_grid, T_grid = config.generate_full_grid(params)
    print(f"\nFull grid shape: {K_grid.shape} (strikes x maturities)")
    
    print()


def demonstrate_parameter_sampling():
    """Demonstrate parameter sampling strategies."""
    print("=== Parameter Sampling Demo ===")
    
    sampler = ParameterSampler(random_seed=42)
    
    # Uniform sampling
    uniform_params = sampler.uniform_sampling(n_samples=50)
    print(f"Generated {len(uniform_params)} uniform samples")
    
    # Latin Hypercube Sampling
    lhs_params = sampler.latin_hypercube_sampling(n_samples=50)
    print(f"Generated {len(lhs_params)} LHS samples")
    
    # Stratified sampling
    stratified_params = sampler.stratified_sampling(n_samples=60)
    print(f"Generated {len(stratified_params)} stratified samples")
    
    # Analyze parameter distributions
    uniform_alphas = [p.alpha for p in uniform_params]
    lhs_alphas = [p.alpha for p in lhs_params]
    stratified_alphas = [p.alpha for p in stratified_params]
    
    print(f"\nAlpha distribution statistics:")
    print(f"  Uniform - mean: {np.mean(uniform_alphas):.3f}, std: {np.std(uniform_alphas):.3f}")
    print(f"  LHS - mean: {np.mean(lhs_alphas):.3f}, std: {np.std(lhs_alphas):.3f}")
    print(f"  Stratified - mean: {np.mean(stratified_alphas):.3f}, std: {np.std(stratified_alphas):.3f}")
    
    print()


def plot_parameter_distributions():
    """Plot parameter distributions for different sampling strategies."""
    print("=== Creating Parameter Distribution Plots ===")
    
    sampler = ParameterSampler(random_seed=42)
    
    # Generate samples
    uniform_params = sampler.uniform_sampling(n_samples=200)
    lhs_params = sampler.latin_hypercube_sampling(n_samples=200)
    stratified_params = sampler.stratified_sampling(n_samples=200)
    
    # Extract alpha and nu for plotting
    uniform_alpha = [p.alpha for p in uniform_params]
    uniform_nu = [p.nu for p in uniform_params]
    
    lhs_alpha = [p.alpha for p in lhs_params]
    lhs_nu = [p.nu for p in lhs_params]
    
    stratified_alpha = [p.alpha for p in stratified_params]
    stratified_nu = [p.nu for p in stratified_params]
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Uniform sampling
    axes[0].scatter(uniform_alpha, uniform_nu, alpha=0.6, s=20)
    axes[0].set_title('Uniform Sampling')
    axes[0].set_xlabel('Alpha')
    axes[0].set_ylabel('Nu')
    axes[0].grid(True, alpha=0.3)
    
    # LHS sampling
    axes[1].scatter(lhs_alpha, lhs_nu, alpha=0.6, s=20, color='orange')
    axes[1].set_title('Latin Hypercube Sampling')
    axes[1].set_xlabel('Alpha')
    axes[1].set_ylabel('Nu')
    axes[1].grid(True, alpha=0.3)
    
    # Stratified sampling
    axes[2].scatter(stratified_alpha, stratified_nu, alpha=0.6, s=20, color='green')
    axes[2].set_title('Stratified Sampling')
    axes[2].set_xlabel('Alpha')
    axes[2].set_ylabel('Nu')
    axes[2].grid(True, alpha=0.3)
    
    # Add parameter range boundaries
    for ax in axes:
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Regime boundaries')
        ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=0.4, color='red', linestyle='--', alpha=0.5)
        ax.set_xlim(SABRParams.ALPHA_RANGE)
        ax.set_ylim(SABRParams.NU_RANGE)
    
    axes[0].legend()
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'parameter_sampling_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"Saved parameter distribution plot to: {output_dir}/parameter_sampling_comparison.png")
    
    plt.show()


def demonstrate_strike_range_behavior():
    """Demonstrate how strike ranges vary with parameters."""
    print("=== Strike Range Behavior Demo ===")
    
    config = GridConfig()
    base_params = SABRParams(F0=100.0, alpha=0.3, beta=0.5, nu=0.4, rho=0.2)
    
    # Test different alpha values
    alphas = [0.1, 0.3, 0.5]
    maturity = 2.0
    
    print(f"Strike ranges for T={maturity}y, varying alpha:")
    for alpha in alphas:
        params = SABRParams(F0=base_params.F0, alpha=alpha, beta=base_params.beta, 
                           nu=base_params.nu, rho=base_params.rho)
        K1, K2 = config.calculate_strike_range(params, maturity)
        width = K2 - K1
        print(f"  α={alpha:.1f}: K1={K1:.1f}, K2={K2:.1f}, width={width:.1f}")
    
    # Test different maturities
    maturities = [0.5, 1.0, 2.0, 5.0]
    print(f"\nStrike ranges for α={base_params.alpha}, varying maturity:")
    for T in maturities:
        K1, K2 = config.calculate_strike_range(base_params, T)
        width = K2 - K1
        print(f"  T={T:.1f}y: K1={K1:.1f}, K2={K2:.1f}, width={width:.1f}")
    
    print()


if __name__ == "__main__":
    print("SABR Parameters and Grid Configuration Example")
    print("=" * 50)
    
    demonstrate_sabr_params()
    demonstrate_grid_config()
    demonstrate_parameter_sampling()
    demonstrate_strike_range_behavior()
    
    # Create plots if matplotlib is available
    try:
        plot_parameter_distributions()
    except ImportError:
        print("Matplotlib not available - skipping plots")
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    print("Demo completed successfully!")