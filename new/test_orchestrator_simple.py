"""
Simple test script for data orchestrator functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_generation.data_orchestrator import DataOrchestrator, DataGenerationConfig
from data_generation.sabr_mc_generator import MCConfig
from data_generation.sabr_params import GridConfig

def test_simple_orchestrator():
    """Test basic orchestrator functionality."""
    print("Testing data orchestrator...")
    
    # Create minimal configuration
    config = DataGenerationConfig(
        output_dir="new/data/test_simple",
        n_parameter_sets=2,
        sampling_strategy="uniform",
        parallel_processing=False,
        validation_enabled=True,
        create_visualizations=False,
        save_intermediate=False
    )
    
    # Use better MC configuration for more accurate results
    config.mc_config = MCConfig(
        n_paths=10000,  # More paths for better convergence
        n_steps=100,    # More time steps for accuracy
        random_seed=42
    )
    
    # Use simple grid
    config.grid_config = GridConfig(
        n_strikes=5,
        n_maturities=2
    )
    
    # Create orchestrator and run
    orchestrator = DataOrchestrator(config)
    result = orchestrator.generate_complete_dataset()
    
    print(f"✓ Generated {len(result.parameter_sets)} parameter sets")
    print(f"✓ Generated {len(result.mc_results)} MC results")
    print(f"✓ Generated {len(result.hagan_results)} Hagan results")
    print(f"✓ Computation time: {result.computation_time:.2f} seconds")
    print(f"✓ Validation passed: {result.validation_results['passed_sets']}/{result.validation_results['total_sets']} sets")
    print(f"✓ Overall quality score: {result.validation_results['overall_quality_score']:.3f}")
    
    # Check file creation
    files_created = 0
    for file_path in result.file_paths.values():
        if isinstance(file_path, str) and not file_path.endswith('_dir'):
            if os.path.exists(file_path):
                files_created += 1
    
    print(f"✓ Created {files_created} data files")
    
    return True

if __name__ == "__main__":
    try:
        success = test_simple_orchestrator()
        print("\n✅ Data orchestrator test completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)