#!/usr/bin/env python3
"""
Data generation script for SABR volatility surface modeling.

This script generates the complete dataset needed for training MDA-CNN and 
Funahashi baseline models. It creates both high-fidelity Monte Carlo and 
low-fidelity Hagan analytical surfaces, then preprocesses them for training.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data_generation.data_orchestrator import DataOrchestrator, DataGenerationConfig
from data_generation.sabr_mc_generator import MCConfig
from data_generation.sabr_params import GridConfig
from data_generation.hagan_surface_generator import HaganConfig
from preprocessing.data_preprocessor import DataPreprocessor, PreprocessingConfig

def setup_logging(log_file, level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate SABR volatility surface training data')
    
    # Configuration file
    parser.add_argument('--config', type=str, default='config/data_generation_config.yaml',
                       help='Path to configuration file')
    
    # Override parameters
    parser.add_argument('--n-parameter-sets', type=int, default=None,
                       help='Number of SABR parameter sets to generate')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Base output directory')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel processing')
    parser.add_argument('--create-visualizations', action='store_true',
                       help='Create visualization plots during generation')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found, using defaults")
        return {}
    except Exception as e:
        logging.error(f"Error loading config file {config_path}: {e}")
        return {}


def setup_experiment(args):
    """Set up experiment directory and logging."""
    # Create output directories
    raw_dir = Path(args.output_dir) / "raw"
    processed_dir = Path(args.output_dir) / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = raw_dir / "data_generation.log"
    setup_logging(log_file, level=logging.INFO)
    
    return raw_dir, processed_dir


def create_data_generation_config(config_dict: dict, args, raw_dir):
    """Create data generation configuration."""
    # Apply command line overrides
    if args.n_parameter_sets is not None:
        config_dict['n_parameter_sets'] = args.n_parameter_sets
    if args.seed is not None:
        config_dict['random_seed'] = args.seed
    if args.parallel:
        config_dict['parallel_processing'] = True
    if args.create_visualizations:
        config_dict['create_visualizations'] = True
    
    # Monte Carlo configuration
    mc_config = MCConfig(
        n_paths=config_dict.get('mc_paths', 100000),
        n_steps=config_dict.get('mc_steps', 300),
        random_seed=config_dict.get('random_seed', 42),
        use_antithetic=config_dict.get('use_antithetic', True),
        convergence_check=config_dict.get('convergence_check', True)
    )
    
    # Hagan configuration
    hagan_config = HaganConfig(
        use_pde_correction=True,
        numerical_tolerance=config_dict.get('numerical_tolerance', 1e-12)
    )
    
    # Grid configuration (following Funahashi's approach)
    grid_config = GridConfig(
        n_strikes=config_dict.get('n_strikes', 21),
        n_maturities=config_dict.get('n_maturities', 5),
        maturity_range=tuple(config_dict.get('maturity_range', [1.0, 10.0])),
        include_extended_wings=config_dict.get('add_wing_strikes', True)
    )
    
    # Main data generation configuration
    config = DataGenerationConfig(
        output_dir=str(raw_dir),
        n_parameter_sets=config_dict.get('n_parameter_sets', 1000),
        sampling_strategy=config_dict.get('sampling_strategy', 'lhs'),
        mc_config=mc_config,
        hagan_config=hagan_config,
        grid_config=grid_config,
        validation_enabled=config_dict.get('validation_enabled', True),
        save_intermediate=config_dict.get('save_intermediate', True),
        parallel_processing=config_dict.get('parallel_processing', True),
        n_workers=config_dict.get('n_workers'),
        random_seed=config_dict.get('random_seed', 42),
        create_visualizations=config_dict.get('create_visualizations', False)
    )
    
    return config


def create_preprocessing_config(config_dict: dict, args, processed_dir):
    """Create preprocessing configuration."""
    config = PreprocessingConfig(
        patch_size=config_dict.get('patch_size', 9),
        output_dir=str(processed_dir),
        validation_split=config_dict.get('validation_split', 0.15),
        test_split=config_dict.get('test_split', 0.15),
        random_seed=config_dict.get('random_seed', 42),
        normalize_patches=config_dict.get('normalize_patches', True),
        normalize_features=config_dict.get('normalize_features', True),
        create_hdf5=config_dict.get('create_hdf5', True),
        batch_size=config_dict.get('batch_size', 64),
        min_samples_per_surface=config_dict.get('min_samples_per_surface', 10),
        max_samples_per_surface=config_dict.get('max_samples_per_surface', 50)
    )
    
    return config


def generate_raw_data(config):
    """Generate raw volatility surface data."""
    logging.info("Starting raw data generation...")
    logging.info(f"Configuration: {config.n_parameter_sets} parameter sets, "
                f"{config.mc_config.n_paths} MC paths, {config.sampling_strategy} sampling")
    
    # Create orchestrator and generate data
    orchestrator = DataOrchestrator(config)
    result = orchestrator.generate_complete_dataset()
    
    # Log results
    logging.info(f"Raw data generation completed in {result.computation_time:.2f} seconds")
    logging.info(f"Generated {len(result.parameter_sets)} parameter sets")
    logging.info(f"Validation: {result.validation_results['passed_sets']}/{result.validation_results['total_sets']} sets passed")
    logging.info(f"Overall quality score: {result.validation_results['overall_quality_score']:.3f}")
    
    return result


def preprocess_data(raw_result, preprocessing_config):
    """Preprocess raw data for training."""
    logging.info("Starting data preprocessing...")
    
    # Create preprocessor
    preprocessor = DataPreprocessor(preprocessing_config)
    
    # Process the raw data
    preprocessing_result = preprocessor.process_raw_data(
        parameter_sets=raw_result.parameter_sets,
        mc_results=raw_result.mc_results,
        hagan_results=raw_result.hagan_results
    )
    
    logging.info(f"Preprocessing completed in {preprocessing_result.computation_time:.2f} seconds")
    logging.info(f"Created {preprocessing_result.n_training_samples} training samples")
    logging.info(f"Created {preprocessing_result.n_validation_samples} validation samples")
    logging.info(f"Created {preprocessing_result.n_test_samples} test samples")
    
    return preprocessing_result


def print_summary(raw_result, preprocessing_result, config_dict, output_dir):
    """Print generation summary."""
    print("\n" + "="*70)
    print("SABR VOLATILITY SURFACE DATA GENERATION COMPLETE")
    print("="*70)
    
    print(f"\nData Generation Summary:")
    print(f"  Parameter sets: {len(raw_result.parameter_sets)}")
    print(f"  Sampling strategy: {config_dict.get('sampling_strategy', 'lhs')}")
    print(f"  MC paths per surface: {config_dict.get('mc_paths', 100000):,}")
    print(f"  Grid size: {config_dict.get('n_strikes', 21)} strikes × {config_dict.get('n_maturities', 5)} maturities")
    print(f"  Generation time: {raw_result.computation_time:.1f} seconds")
    
    print(f"\nData Quality:")
    print(f"  Validation passed: {raw_result.validation_results['passed_sets']}/{raw_result.validation_results['total_sets']} sets")
    print(f"  Quality score: {raw_result.validation_results['overall_quality_score']:.3f}")
    
    print(f"\nPreprocessing Summary:")
    print(f"  Training samples: {preprocessing_result.n_training_samples:,}")
    print(f"  Validation samples: {preprocessing_result.n_validation_samples:,}")
    print(f"  Test samples: {preprocessing_result.n_test_samples:,}")
    print(f"  Patch size: {config_dict.get('patch_size', 9)}×{config_dict.get('patch_size', 9)}")
    print(f"  Processing time: {preprocessing_result.computation_time:.1f} seconds")
    
    print(f"\nOutput Files:")
    print(f"  Raw data: {output_dir}/raw/")
    print(f"  Processed data: {output_dir}/processed/")
    
    print(f"\nNext Steps:")
    print(f"  1. Run training: python main_training.py --data-dir {output_dir}/processed")
    print(f"  2. Or run complete experiment: python run_experiment.py")
    
    print("\n✅ Data generation completed successfully!")


def main():
    """Main data generation pipeline."""
    args = parse_arguments()
    
    # Load configuration
    config_dict = load_config(args.config)
    
    # Override output directory if specified
    output_dir = args.output_dir or config_dict.get('output_dir', 'data')
    
    # Setup experiment
    raw_dir = Path(output_dir) / "raw"
    processed_dir = Path(output_dir) / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = raw_dir / "data_generation.log"
    setup_logging(log_file, level=logging.INFO)
    
    try:
        # Create configurations
        data_gen_config = create_data_generation_config(config_dict, args, raw_dir)
        preprocessing_config = create_preprocessing_config(config_dict, args, processed_dir)
        
        # Generate raw data
        raw_result = generate_raw_data(data_gen_config)
        
        # Preprocess data for training
        preprocessing_result = preprocess_data(raw_result, preprocessing_config)
        
        # Print summary
        print_summary(raw_result, preprocessing_result, config_dict, output_dir)
        
    except Exception as e:
        logging.error(f"Data generation failed: {str(e)}")
        print(f"\n❌ Data generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()