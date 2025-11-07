#!/usr/bin/env python3
"""
Pipeline status checker for SABR volatility surface modeling project.

This script checks if all components are properly set up and can run the complete pipeline.
"""

import os
import sys
from pathlib import Path
import importlib.util


def check_file_exists(file_path: str, description: str) -> bool:
    """Check if a file exists."""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (MISSING)")
        return False


def check_import(module_name: str, description: str) -> bool:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_name} (IMPORT ERROR: {e})")
        return False


def check_directory_structure():
    """Check if all required directories exist."""
    print("\n📁 DIRECTORY STRUCTURE")
    print("-" * 40)
    
    required_dirs = [
        ("config", "Configuration directory"),
        ("data_generation", "Data generation modules"),
        ("preprocessing", "Data preprocessing modules"),
        ("models", "Model implementations"),
        ("training", "Training infrastructure"),
        ("evaluation", "Evaluation modules"),
        ("visualization", "Visualization modules"),
        ("utils", "Utility modules"),
    ]
    
    all_exist = True
    for dir_path, description in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {description}: {dir_path}")
        else:
            print(f"❌ {description}: {dir_path} (MISSING)")
            all_exist = False
    
    return all_exist


def check_config_files():
    """Check if all configuration files exist."""
    print("\n⚙️  CONFIGURATION FILES")
    print("-" * 40)
    
    config_files = [
        ("config/data_generation_config.yaml", "Data generation config"),
        ("config/training_config.yaml", "Training config"),
        ("config/default_config.yaml", "Default config"),
    ]
    
    all_exist = True
    for file_path, description in config_files:
        if not check_file_exists(file_path, description):
            all_exist = False
    
    return all_exist


def check_main_scripts():
    """Check if all main execution scripts exist."""
    print("\n🚀 MAIN EXECUTION SCRIPTS")
    print("-" * 40)
    
    main_scripts = [
        ("generate_training_data.py", "Data generation script"),
        ("main_training.py", "Training script"),
        ("main_evaluation.py", "Evaluation script"),
        ("run_experiment.py", "Complete experiment runner"),
        ("test_data_generation.py", "Data generation test"),
    ]
    
    all_exist = True
    for file_path, description in main_scripts:
        if not check_file_exists(file_path, description):
            all_exist = False
    
    return all_exist


def check_core_modules():
    """Check if core modules can be imported."""
    print("\n🔧 CORE MODULES")
    print("-" * 40)
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    core_modules = [
        ("data_generation.data_orchestrator", "Data orchestrator"),
        ("data_generation.sabr_mc_generator", "Monte Carlo generator"),
        ("data_generation.hagan_surface_generator", "Hagan surface generator"),
        ("preprocessing.data_preprocessor", "Data preprocessor"),
        ("preprocessing.data_loader", "Data loader"),
        ("models.mdacnn_model", "MDA-CNN model"),
        ("models.baseline_models", "Baseline models"),
        ("config.training_config", "Training configuration"),
        ("utils.config_utils", "Config utilities"),
        ("utils.logging_utils", "Logging utilities"),
    ]
    
    all_importable = True
    for module_name, description in core_modules:
        if not check_import(module_name, description):
            all_importable = False
    
    return all_importable


def check_dependencies():
    """Check if required dependencies are available."""
    print("\n📦 DEPENDENCIES")
    print("-" * 40)
    
    dependencies = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("torch", "PyTorch"),
        ("h5py", "HDF5 support"),
        ("yaml", "YAML support"),
        ("pickle", "Pickle support"),
        ("pathlib", "Path utilities"),
        ("dataclasses", "Dataclasses"),
        ("typing", "Type hints"),
    ]
    
    all_available = True
    for module_name, description in dependencies:
        if not check_import(module_name, description):
            all_available = False
    
    return all_available


def check_data_directories():
    """Check data directory structure."""
    print("\n💾 DATA DIRECTORIES")
    print("-" * 40)
    
    data_dirs = [
        ("data", "Base data directory"),
        ("data/raw", "Raw data directory"),
        ("data/processed", "Processed data directory"),
    ]
    
    for dir_path, description in data_dirs:
        path = Path(dir_path)
        if path.exists():
            files = list(path.glob("*"))
            if files:
                print(f"✅ {description}: {dir_path} ({len(files)} files)")
            else:
                print(f"⚠️  {description}: {dir_path} (empty)")
        else:
            print(f"📁 {description}: {dir_path} (will be created)")


def run_pipeline_test():
    """Run a quick pipeline test."""
    print("\n🧪 PIPELINE TEST")
    print("-" * 40)
    
    try:
        # Test data generation components
        from data_generation.data_orchestrator import DataOrchestrator, DataGenerationConfig
        from data_generation.sabr_mc_generator import MCConfig
        from data_generation.sabr_params import GridConfig
        
        print("✅ Data generation components can be imported")
        
        # Test preprocessing components
        from preprocessing.data_preprocessor import DataPreprocessor, PreprocessingConfig
        from preprocessing.data_loader import SABRDataset
        
        print("✅ Preprocessing components can be imported")
        
        # Test model components
        from models.mdacnn_model import MDACNNModel
        from models.baseline_models import FunahashiBaselineModel
        
        print("✅ Model components can be imported")
        
        # Test configuration
        from config.training_config import TrainingConfig, load_config
        
        config = load_config("config/training_config.yaml")
        print("✅ Configuration can be loaded")
        
        # Test model creation
        model = MDACNNModel(
            patch_size=config.patch_size,
            point_features_dim=config.point_features_dim
        )
        print(f"✅ MDA-CNN model can be created ({model.count_parameters():,} parameters)")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False


def main():
    """Main pipeline checker."""
    print("SABR VOLATILITY SURFACE MODELING - PIPELINE STATUS CHECK")
    print("=" * 60)
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Configuration Files", check_config_files),
        ("Main Scripts", check_main_scripts),
        ("Core Modules", check_core_modules),
        ("Dependencies", check_dependencies),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        if not check_func():
            all_passed = False
    
    # Check data directories (informational)
    check_data_directories()
    
    # Run pipeline test
    if all_passed:
        pipeline_test_passed = run_pipeline_test()
        all_passed = all_passed and pipeline_test_passed
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED! Pipeline is ready to run.")
        print("\nNext steps:")
        print("1. Test data generation: python test_data_generation.py")
        print("2. Generate full dataset: python generate_training_data.py")
        print("3. Run complete experiment: python run_experiment.py")
        print("4. Or run individual components:")
        print("   - python generate_training_data.py")
        print("   - python main_training.py --data-dir data/processed")
        print("   - python main_evaluation.py --experiment-dir results/[experiment]")
    else:
        print("❌ SOME CHECKS FAILED! Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install missing dependencies: pip install torch numpy scipy h5py pyyaml")
        print("- Check file paths and ensure all files are in the correct locations")
        print("- Verify that you're running from the 'new/' directory")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())