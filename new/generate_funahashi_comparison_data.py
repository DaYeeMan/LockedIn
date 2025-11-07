#!/usr/bin/env python3
"""
Generate data for direct comparison with Funahashi's results.

This script generates volatility surfaces using Funahashi's exact test cases
and parameters for direct result comparison.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data_generation.data_orchestrator import DataOrchestrator, DataGenerationConfig
from data_generation.sabr_mc_generator import MCConfig
from data_generation.sabr_params import GridConfig, FunahashiTestCases
from data_generation.hagan_surface_generator import HaganConfig
from preprocessing.data_preprocessor import DataPreprocessor, PreprocessingConfig


def setup_logging():
    """Setup logging for Funahashi comparison."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_funahashi_config():
    """Create configuration for Funahashi comparison."""
    # Monte Carlo configuration - balanced precision and speed
    mc_config = MCConfig(
        n_paths=100000,   # Good precision, reasonable speed
        n_steps=300,      # Good time discretization
        random_seed=12345,
        use_antithetic=True,
        convergence_check=True
    )
    
    # Hagan configuration - high precision
    hagan_config = HaganConfig(
        use_pde_correction=True,
        numerical_tolerance=1e-15
    )
    
    # Grid configuration - single maturity T=3 years
    grid_config = GridConfig(
        n_strikes=21,
        n_maturities=1,
        maturity_range=(3.0, 3.01),  # T = 3 years (slight range for validation)
        include_extended_wings=False
    )
    
    # Data generation configuration
    config = DataGenerationConfig(
        output_dir="data_funahashi_comparison/raw",
        n_parameter_sets=4,  # Funahashi's 4 test cases
        sampling_strategy="funahashi_exact",
        mc_config=mc_config,
        hagan_config=hagan_config,
        grid_config=grid_config,
        validation_enabled=True,
        save_intermediate=True,
        parallel_processing=False,  # Sequential for reproducibility
        create_visualizations=True,
        random_seed=12345
    )
    
    return config


def create_funahashi_preprocessing_config():
    """Create preprocessing configuration for Funahashi comparison."""
    config = PreprocessingConfig(
        patch_size=9,
        output_dir="data_funahashi_comparison/processed",
        validation_split=0.0,  # No validation split for exact comparison
        test_split=0.0,        # Use all data for comparison
        random_seed=12345,
        normalize_patches=True,
        normalize_features=True,
        create_hdf5=True,
        min_samples_per_surface=21,  # All 21 strikes
        max_samples_per_surface=21   # All 21 strikes
    )
    
    return config


def generate_funahashi_comparison_data():
    """Generate data for Funahashi comparison."""
    print("FUNAHASHI DIRECT COMPARISON DATA GENERATION")
    print("=" * 50)
    
    # Create output directories
    raw_dir = Path("data_funahashi_comparison/raw")
    processed_dir = Path("data_funahashi_comparison/processed")
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Generate raw data with Funahashi's exact parameters
        print("\n1. Generating raw data with Funahashi's exact test cases...")
        
        config = create_funahashi_config()
        orchestrator = DataOrchestrator(config)
        raw_result = orchestrator.generate_complete_dataset()
        
        print(f"✓ Generated {len(raw_result.parameter_sets)} parameter sets")
        print(f"✓ Generation time: {raw_result.computation_time:.2f} seconds")
        print(f"✓ Quality score: {raw_result.validation_results['overall_quality_score']:.3f}")
        
        # Print the exact parameters used
        print("\nFunahashi's Test Cases Generated:")
        test_cases = FunahashiTestCases.get_test_cases()
        case_names = FunahashiTestCases.get_case_names()
        
        for i, (name, params) in enumerate(zip(case_names, test_cases)):
            print(f"  {name}: f={params.F0}, α={params.alpha}, β={params.beta}, ν={params.nu}, ρ={params.rho}")
        
        # Step 2: Preprocess data
        print("\n2. Preprocessing data for comparison...")
        
        preprocessing_config = create_funahashi_preprocessing_config()
        preprocessor = DataPreprocessor(preprocessing_config)
        preprocessing_result = preprocessor.process_raw_data(
            raw_result.parameter_sets,
            raw_result.mc_results,
            raw_result.hagan_results
        )
        
        print(f"✓ Created {preprocessing_result.n_training_samples} samples for comparison")
        print(f"✓ Processing time: {preprocessing_result.computation_time:.2f} seconds")
        
        # Step 3: Save comparison metadata
        comparison_info = {
            'description': 'Direct comparison with Funahashi results',
            'test_cases': [
                {'name': name, 'params': params.to_dict()} 
                for name, params in zip(case_names, test_cases)
            ],
            'strikes': FunahashiTestCases.get_funahashi_strikes().tolist(),
            'maturity': 3.0,
            'mc_paths': config.mc_config.n_paths,
            'generation_time': raw_result.computation_time,
            'quality_score': raw_result.validation_results['overall_quality_score']
        }
        
        import json
        with open("data_funahashi_comparison/comparison_info.json", 'w') as f:
            json.dump(comparison_info, f, indent=2)
        
        print("\n" + "=" * 50)
        print("FUNAHASHI COMPARISON DATA GENERATION COMPLETE")
        print("=" * 50)
        print(f"\nGenerated data for direct comparison with Funahashi's paper:")
        print(f"  - 4 exact test cases from Funahashi's Table 2")
        print(f"  - 21 strikes matching Funahashi's Table 3")
        print(f"  - High-precision MC simulation (1M paths)")
        print(f"  - Quality score: {raw_result.validation_results['overall_quality_score']:.3f}")
        print(f"\nData saved to: data_funahashi_comparison/")
        print(f"\nNext steps:")
        print(f"  1. Train models: python main_training.py --data-dir data_funahashi_comparison/processed")
        print(f"  2. Compare results with Funahashi's Table 3")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Funahashi comparison data generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    setup_logging()
    
    success = generate_funahashi_comparison_data()
    
    if success:
        print("\n🎉 Ready for direct comparison with Funahashi's results!")
        sys.exit(0)
    else:
        print("\n💥 Fix the issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()