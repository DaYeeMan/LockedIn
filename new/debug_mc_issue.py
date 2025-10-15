"""
Debug script to investigate MC simulation issue for second maturity.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from data_generation.sabr_params import SABRParams, GridConfig
from data_generation.sabr_mc_generator import SABRMCGenerator, MCConfig

def debug_mc_issue():
    """Debug the MC simulation issue."""
    print("Debugging MC simulation issue...")
    
    # Use the same parameters that failed
    sabr_params = SABRParams(
        F0=1.0,
        alpha=0.2559970653660493,
        beta=0.8704285838459498,
        nu=0.6721948505396944,
        rho=0.1479877262955549
    )
    
    # Create MC generator
    mc_config = MCConfig(n_paths=1000, n_steps=50, random_seed=42)
    generator = SABRMCGenerator(mc_config)
    
    # Create grid config
    grid_config = GridConfig(n_strikes=5, n_maturities=2)
    maturities = grid_config.generate_maturity_grid()
    
    print(f"Testing maturities: {maturities}")
    print(f"SABR params: F0={sabr_params.F0}, alpha={sabr_params.alpha}, beta={sabr_params.beta}, nu={sabr_params.nu}, rho={sabr_params.rho}")
    
    for i, maturity in enumerate(maturities):
        print(f"\n--- Testing maturity {i+1}: {maturity} years ---")
        
        # Generate strikes
        strikes = grid_config.generate_strike_grid(sabr_params, maturity)
        print(f"Strikes: {strikes}")
        
        # Simulate paths
        try:
            forward_paths, alpha_paths = generator.simulate_paths(sabr_params, maturity)
            print(f"Forward paths shape: {forward_paths.shape}")
            print(f"Final forward prices - min: {np.min(forward_paths[:, -1]):.4f}, max: {np.max(forward_paths[:, -1]):.4f}, mean: {np.mean(forward_paths[:, -1]):.4f}")
            
            # Check for any issues with paths
            nan_count = np.sum(np.isnan(forward_paths))
            inf_count = np.sum(np.isinf(forward_paths))
            negative_count = np.sum(forward_paths <= 0)
            
            print(f"Path issues - NaN: {nan_count}, Inf: {inf_count}, Negative/Zero: {negative_count}")
            
            # Test option pricing for each strike
            for j, strike in enumerate(strikes):
                option_price = generator.calculate_option_price(forward_paths, strike)
                print(f"  Strike {strike:.4f}: Option price = {option_price:.6f}")
                
                if option_price > 0:
                    impl_vol = generator.implied_volatility_from_price(
                        option_price, sabr_params.F0, strike, maturity
                    )
                    print(f"    Implied vol = {impl_vol:.6f}")
                else:
                    print(f"    Zero option price - cannot calculate implied vol")
                    
        except Exception as e:
            print(f"Error simulating paths for maturity {maturity}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_mc_issue()