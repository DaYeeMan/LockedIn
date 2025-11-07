#!/usr/bin/env python3
"""
Test script for data generation process.

This script runs a minimal data generation test to verify the pipeline works
before running the full data generation.
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
from data_generation.sabr_params import GridConfig
from data_generation.hagan_surface_generator import HaganConfig
from preprocessing.data_preprocessor import DataPreprocessor, PreprocessingConfig


def setup_logging():
    """Setup simple logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def test_data_generation():
    """Test the complete data generation pipeline with minimal data."""
    print("Testing SABR data generation pipeline...")
    
    # Create test output directory
    test_dir = Path("test_data_output")
    raw_dir = test_dir / "raw"
    processed_dir = test_dir / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Test raw data generation
        print("\n1. Testing raw data generation...")
        
        # Minimal configuration for testing
        mc_config = MCConfig(
            n_paths=1000,  # Very small for testing
            n_steps=50,
            random_seed=42,
            use_antithetic=False,
            convergence_check=False
        )
        
        hagan_config = HaganConfig()
        
        grid_config = GridConfig(
            n_strikes=5,    # Small grid for testing
            n_maturities=2
        )
        
        data_gen_config = DataGenerationConfig(
            output_dir=str(raw_dir),
            n_parameter_sets=3,  # Very small for testing
            sampling_strategy="uniform",
            mc_config=mc_config,
            hagan_config=hagan_config,
            grid_config=grid_config,
            validation_enabled=True,
            save_intermediate=False,
            parallel_processing=False,  # Disable for testing
            create_visualizations=False,
            random_seed=42
        )
        
        # Generate raw data
        orchestrator = DataOrchestrator(data_gen_config)
        raw_result = orchestrator.generate_complete_dataset()
        
        print(f"✓ Generated {len(raw_result.parameter_sets)} parameter sets")
        print(f"✓ Generation time: {raw_result.computation_time:.2f} seconds")
        print(f"✓ Quality score: {raw_result.validation_results['overall_quality_score']:.3f}")
        
        # Step 2: Test preprocessing
        print("\n2. Testing data preprocessing...")
        
        preprocessing_config = PreprocessingConfig(
            patch_size=5,  # Small patches for testing
            output_dir=str(processed_dir),
            validation_split=0.2,
            test_split=0.2,
            random_seed=42,
            normalize_patches=True,
            normalize_features=True,
            create_hdf5=True,
            min_samples_per_surface=2,
            max_samples_per_surface=5
        )
        
        preprocessor = DataPreprocessor(preprocessing_config)
        preprocessing_result = preprocessor.process_raw_data(
            raw_result.parameter_sets,
            raw_result.mc_results,
            raw_result.hagan_results
        )
        
        print(f"✓ Created {preprocessing_result.n_training_samples} training samples")
        print(f"✓ Created {preprocessing_result.n_validation_samples} validation samples")
        print(f"✓ Created {preprocessing_result.n_test_samples} test samples")
        print(f"✓ Processing time: {preprocessing_result.computation_time:.2f} seconds")
        
        # Step 3: Test data loading
        print("\n3. Testing data loading...")
        
        from preprocessing.data_loader import SABRDataset
        
        # Test loading the generated data
        train_dataset = SABRDataset(str(processed_dir), split='train')
        val_dataset = SABRDataset(str(processed_dir), split='val')
        test_dataset = SABRDataset(str(processed_dir), split='test')
        
        print(f"✓ Loaded {len(train_dataset)} training samples")
        print(f"✓ Loaded {len(val_dataset)} validation samples")
        print(f"✓ Loaded {len(test_dataset)} test samples")
        
        # Test getting a sample
        if len(train_dataset) > 0:
            patch, features, target = train_dataset[0]
            print(f"✓ Sample shapes: patch {patch.shape}, features {features.shape}, target {target}")
        
        print("\n✅ All tests passed! Data generation pipeline is working correctly.")
        print(f"\nTest output saved to: {test_dir}")
        print("\nYou can now run the full data generation:")
        print("  python generate_training_data.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test directory (optional)
        # import shutil
        # if test_dir.exists():
        #     shutil.rmtree(test_dir)
        pass


def main():
    """Main test function."""
    setup_logging()
    
    print("SABR Volatility Surface Data Generation Test")
    print("=" * 50)
    
    success = test_data_generation()
    
    if success:
        print("\n🎉 Ready to generate full dataset!")
        sys.exit(0)
    else:
        print("\n💥 Fix the issues above before running full generation.")
        sys.exit(1)


if __name__ == "__main__":
    main()