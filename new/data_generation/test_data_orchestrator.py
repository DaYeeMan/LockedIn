"""
Integration tests for complete data generation workflow.

Tests the data orchestrator, validation, and end-to-end pipeline
to ensure all components work together correctly.
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path
import pickle
import json

from data_generation.data_orchestrator import (
    DataOrchestrator, DataGenerationConfig, DataQualityValidator,
    create_default_config, quick_generation_example
)
from data_generation.sabr_params import SABRParams, GridConfig
from data_generation.sabr_mc_generator import MCConfig, MCResult
from data_generation.hagan_surface_generator import HaganConfig, HaganResult
from visualization.data_visualizer import VisualizationConfig


class TestDataQualityValidator:
    """Test data quality validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataQualityValidator()
        self.sabr_params = SABRParams(F0=1.0, alpha=0.2, beta=0.5, nu=0.3, rho=-0.2)
        
        # Create mock results
        self.mock_mc_result = self._create_mock_mc_result()
        self.mock_hagan_result = self._create_mock_hagan_result()
    
    def _create_mock_mc_result(self) -> MCResult:
        """Create mock MC result for testing."""
        strikes = np.array([[0.8, 0.9, 1.0, 1.1, 1.2]])
        maturities = np.array([1.0])
        volatilities = np.array([[0.25, 0.22, 0.20, 0.22, 0.25]])
        
        return MCResult(
            strikes=strikes,
            maturities=maturities,
            volatility_surface=volatilities,
            option_prices=np.array([[0.1, 0.08, 0.05, 0.08, 0.1]]),
            convergence_info={'overall': {'convergence_achieved': True}},
            computation_time=1.0
        )
    
    def _create_mock_hagan_result(self) -> HaganResult:
        """Create mock Hagan result for testing."""
        strikes = np.array([[0.8, 0.9, 1.0, 1.1, 1.2]])
        maturities = np.array([1.0])
        volatilities = np.array([[0.24, 0.21, 0.19, 0.21, 0.24]])
        
        return HaganResult(
            strikes=strikes,
            maturities=maturities,
            volatility_surface=volatilities,
            computation_time=0.1,
            numerical_warnings=[],
            grid_info={}
        )
    
    def test_validate_surface_data_success(self):
        """Test successful validation of good surface data."""
        result = self.validator.validate_surface_data(
            self.mock_mc_result, self.mock_hagan_result, self.sabr_params
        )
        
        assert result['passed'] is True
        assert result['quality_score'] > 0.8
        assert 'checks' in result
        assert len(result['errors']) == 0
    
    def test_validate_surface_data_with_issues(self):
        """Test validation with data quality issues."""
        # Create problematic data
        bad_mc_result = self.mock_mc_result
        bad_mc_result.volatility_surface = np.array([[np.nan, -0.1, 5.0, np.inf, 0.2]])
        
        result = self.validator.validate_surface_data(
            bad_mc_result, self.mock_hagan_result, self.sabr_params
        )
        
        assert result['passed'] is False
        assert result['quality_score'] < 0.8
        assert len(result['warnings']) > 0 or len(result['errors']) > 0
    
    def test_volatility_bounds_check(self):
        """Test volatility bounds validation."""
        # Test with good volatilities
        good_vols = np.array([[0.1, 0.2, 0.3]])
        result = self.validator._check_volatility_bounds(good_vols, "Test")
        assert result['passed'] == True
        
        # Test with bad volatilities
        bad_vols = np.array([[-0.1, 5.0, np.inf]])
        result = self.validator._check_volatility_bounds(bad_vols, "Test")
        assert result['passed'] == False
    
    def test_atm_consistency_check(self):
        """Test ATM volatility consistency with alpha."""
        result = self.validator._check_atm_consistency(
            self.mock_mc_result, self.mock_hagan_result, self.sabr_params
        )
        
        assert 'passed' in result
        assert 'stats' in result
        assert 'mc_atm_volatility' in result['stats']
        assert 'hagan_atm_volatility' in result['stats']
    
    def test_residual_outlier_detection(self):
        """Test residual outlier detection."""
        result = self.validator._detect_residual_outliers(
            self.mock_mc_result, self.mock_hagan_result
        )
        
        assert 'outlier_count' in result
        assert 'outlier_ratio' in result
        assert result['outlier_count'] >= 0
        assert 0 <= result['outlier_ratio'] <= 1


class TestDataOrchestrator:
    """Test data orchestrator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal configuration for fast testing
        self.config = DataGenerationConfig(
            output_dir=self.temp_dir,
            n_parameter_sets=3,
            sampling_strategy="uniform",
            parallel_processing=False,
            validation_enabled=True,
            create_visualizations=False,  # Disable for speed
            save_intermediate=False
        )
        
        # Use fast MC configuration
        self.config.mc_config = MCConfig(
            n_paths=1000,
            n_steps=50,
            random_seed=42
        )
        
        # Use simple grid configuration
        self.config.grid_config = GridConfig(
            n_strikes=5,
            n_maturities=2
        )
        
        self.orchestrator = DataOrchestrator(self.config)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator.config == self.config
        assert self.orchestrator.validator is not None
        assert self.orchestrator.mc_generator is not None
        assert self.orchestrator.hagan_generator is not None
        assert self.orchestrator.parameter_sampler is not None
    
    def test_generate_parameter_sets(self):
        """Test parameter set generation."""
        param_sets = self.orchestrator._generate_parameter_sets()
        
        assert len(param_sets) == self.config.n_parameter_sets
        assert all(isinstance(params, SABRParams) for params in param_sets)
        
        # Test parameter validation
        for params in param_sets:
            params.validate()  # Should not raise exception
    
    def test_generate_surfaces_sequential(self):
        """Test sequential surface generation."""
        param_sets = self.orchestrator._generate_parameter_sets()
        mc_results, hagan_results = self.orchestrator._generate_surfaces_sequential(param_sets)
        
        assert len(mc_results) == len(param_sets)
        assert len(hagan_results) == len(param_sets)
        
        # Check result types
        for mc_res, hagan_res in zip(mc_results, hagan_results):
            assert isinstance(mc_res, MCResult)
            assert isinstance(hagan_res, HaganResult)
    
    def test_validate_all_data(self):
        """Test comprehensive data validation."""
        param_sets = self.orchestrator._generate_parameter_sets()
        mc_results, hagan_results = self.orchestrator._generate_surfaces_sequential(param_sets)
        
        validation_results = self.orchestrator._validate_all_data(
            param_sets, mc_results, hagan_results
        )
        
        assert 'total_sets' in validation_results
        assert 'passed_sets' in validation_results
        assert 'failed_sets' in validation_results
        assert 'individual_results' in validation_results
        assert 'overall_quality_score' in validation_results
        
        assert validation_results['total_sets'] == len(param_sets)
        assert len(validation_results['individual_results']) == len(param_sets)
    
    def test_save_all_data(self):
        """Test data saving functionality."""
        param_sets = self.orchestrator._generate_parameter_sets()
        mc_results, hagan_results = self.orchestrator._generate_surfaces_sequential(param_sets)
        validation_results = {'test': 'data'}
        
        file_paths = self.orchestrator._save_all_data(
            param_sets, mc_results, hagan_results, validation_results
        )
        
        # Check that all expected files were created
        expected_files = [
            'parameter_sets', 'parameter_sets_csv', 'mc_results', 
            'hagan_results', 'validation_results', 'config'
        ]
        
        for file_key in expected_files:
            assert file_key in file_paths
            assert os.path.exists(file_paths[file_key])
        
        # Check ML-ready data directory
        assert 'ml_data_dir' in file_paths
        assert os.path.exists(file_paths['ml_data_dir'])
        assert os.path.exists(f"{file_paths['ml_data_dir']}/training_data.csv")
        assert os.path.exists(f"{file_paths['ml_data_dir']}/training_data.h5")
    
    def test_generate_complete_dataset(self):
        """Test complete end-to-end data generation."""
        result = self.orchestrator.generate_complete_dataset()
        
        # Check result structure
        assert len(result.parameter_sets) == self.config.n_parameter_sets
        assert len(result.mc_results) == self.config.n_parameter_sets
        assert len(result.hagan_results) == self.config.n_parameter_sets
        assert result.validation_results is not None
        assert result.generation_metadata is not None
        assert result.file_paths is not None
        assert result.computation_time > 0
        
        # Check that files were actually created
        for file_path in result.file_paths.values():
            if isinstance(file_path, str) and not file_path.endswith('_dir'):
                assert os.path.exists(file_path)
    
    def test_load_existing_data(self):
        """Test loading previously generated data."""
        # First generate some data
        result = self.orchestrator.generate_complete_dataset()
        
        # Extract run directory from file paths
        run_dir = os.path.dirname(result.file_paths['parameter_sets'])
        
        # Load the data back
        loaded_result = self.orchestrator.load_existing_data(run_dir)
        
        # Check that loaded data matches original
        assert len(loaded_result.parameter_sets) == len(result.parameter_sets)
        assert len(loaded_result.mc_results) == len(result.mc_results)
        assert len(loaded_result.hagan_results) == len(result.hagan_results)
        
        # Check parameter sets match
        for orig_params, loaded_params in zip(result.parameter_sets, loaded_result.parameter_sets):
            assert orig_params.F0 == loaded_params.F0
            assert orig_params.alpha == loaded_params.alpha
            assert orig_params.beta == loaded_params.beta
            assert orig_params.nu == loaded_params.nu
            assert orig_params.rho == loaded_params.rho
    
    def test_create_ml_ready_data(self):
        """Test ML-ready data creation."""
        param_sets = self.orchestrator._generate_parameter_sets()
        mc_results, hagan_results = self.orchestrator._generate_surfaces_sequential(param_sets)
        
        # Create temporary run directory
        run_dir = f"{self.temp_dir}/test_run"
        os.makedirs(run_dir, exist_ok=True)
        
        self.orchestrator._create_ml_ready_data(run_dir, param_sets, mc_results, hagan_results)
        
        # Check that ML-ready files were created
        ml_dir = f"{run_dir}/ml_ready"
        assert os.path.exists(f"{ml_dir}/training_data.csv")
        assert os.path.exists(f"{ml_dir}/training_data.h5")
        assert os.path.exists(f"{ml_dir}/data_summary.json")
        
        # Load and check training data
        import pandas as pd
        df = pd.read_csv(f"{ml_dir}/training_data.csv")
        
        expected_columns = [
            'param_set_id', 'F0', 'alpha', 'beta', 'nu', 'rho',
            'strike', 'maturity', 'moneyness', 'mc_volatility',
            'hagan_volatility', 'residual'
        ]
        
        for col in expected_columns:
            assert col in df.columns
        
        # Check data summary
        with open(f"{ml_dir}/data_summary.json", 'r') as f:
            summary = json.load(f)
        
        assert 'total_data_points' in summary
        assert 'residual_statistics' in summary
        assert summary['total_data_points'] > 0


class TestIntegrationWorkflow:
    """Test complete integration workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_create_default_config(self):
        """Test default configuration creation."""
        config = create_default_config(
            n_parameter_sets=50,
            output_dir=self.temp_dir,
            parallel=False
        )
        
        assert config.n_parameter_sets == 50
        assert config.output_dir == self.temp_dir
        assert config.parallel_processing is False
        assert config.validation_enabled is True
        assert config.mc_config is not None
        assert config.hagan_config is not None
        assert config.grid_config is not None
    
    def test_quick_generation_example(self):
        """Test quick generation example function."""
        # Temporarily change the output directory
        import data_generation.data_orchestrator as orchestrator_module
        original_create_default_config = orchestrator_module.create_default_config
        
        def mock_create_default_config(n_parameter_sets, output_dir="new/data/quick_test", parallel=False):
            return original_create_default_config(n_parameter_sets, self.temp_dir, parallel)
        
        orchestrator_module.create_default_config = mock_create_default_config
        
        try:
            result = quick_generation_example(n_sets=3)
            
            assert len(result.parameter_sets) == 3
            assert len(result.mc_results) == 3
            assert len(result.hagan_results) == 3
            assert result.computation_time > 0
            
        finally:
            # Restore original function
            orchestrator_module.create_default_config = original_create_default_config
    
    def test_end_to_end_workflow_with_validation(self):
        """Test complete end-to-end workflow with validation."""
        # Create configuration
        config = DataGenerationConfig(
            output_dir=self.temp_dir,
            n_parameter_sets=5,
            sampling_strategy="lhs",
            parallel_processing=False,
            validation_enabled=True,
            create_visualizations=False,
            save_intermediate=True
        )
        
        # Use fast settings for testing
        config.mc_config = MCConfig(n_paths=5000, n_steps=50, random_seed=42)
        config.grid_config = GridConfig(n_strikes=7, n_maturities=3)
        
        # Run complete workflow
        orchestrator = DataOrchestrator(config)
        result = orchestrator.generate_complete_dataset()
        
        # Comprehensive validation of results
        assert len(result.parameter_sets) == 5
        assert len(result.mc_results) == 5
        assert len(result.hagan_results) == 5
        
        # Check validation results
        validation = result.validation_results
        assert validation['total_sets'] == 5
        assert 'overall_quality_score' in validation
        assert validation['overall_quality_score'] >= 0.0
        
        # Check that all files exist
        for file_path in result.file_paths.values():
            if isinstance(file_path, str) and not file_path.endswith('_dir'):
                assert os.path.exists(file_path), f"File not found: {file_path}"
        
        # Check intermediate files if enabled
        if config.save_intermediate:
            intermediate_dir = f"{config.output_dir}/intermediate"
            if os.path.exists(intermediate_dir):
                intermediate_files = os.listdir(intermediate_dir)
                assert len(intermediate_files) > 0
        
        # Test loading the generated data
        run_dir = os.path.dirname(result.file_paths['parameter_sets'])
        loaded_result = orchestrator.load_existing_data(run_dir)
        
        assert len(loaded_result.parameter_sets) == len(result.parameter_sets)
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Create configuration that might cause some failures
        config = DataGenerationConfig(
            output_dir=self.temp_dir,
            n_parameter_sets=3,
            parallel_processing=False,
            validation_enabled=True
        )
        
        # Use extreme parameters that might cause numerical issues
        config.mc_config = MCConfig(n_paths=100, n_steps=10, random_seed=42)
        
        orchestrator = DataOrchestrator(config)
        
        # Override parameter sampler to generate potentially problematic parameters
        def problematic_sampling(n_samples):
            return [
                SABRParams(F0=1.0, alpha=0.01, beta=0.99, nu=0.01, rho=0.0),  # Extreme case
                SABRParams(F0=1.0, alpha=0.3, beta=0.5, nu=0.4, rho=-0.3),   # Normal case
                SABRParams(F0=1.0, alpha=0.6, beta=0.3, nu=0.9, rho=0.75)    # Boundary case
            ]
        
        orchestrator.parameter_sampler.uniform_sampling = problematic_sampling
        
        # Should complete without crashing, even with some failures
        result = orchestrator.generate_complete_dataset()
        
        assert len(result.parameter_sets) == 3
        assert len(result.mc_results) == 3
        assert len(result.hagan_results) == 3
        
        # Check that validation caught any issues
        validation = result.validation_results
        if validation['failed_sets'] > 0:
            assert len(validation['individual_results']) == 3
            # Some sets might fail validation but process should complete


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])