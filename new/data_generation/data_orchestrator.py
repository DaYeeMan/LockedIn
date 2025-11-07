"""
Main data generation orchestrator for SABR volatility surface modeling.

This module coordinates MC and Hagan surface creation, implements data quality
validation, provides data saving/loading utilities with proper file organization,
and includes progress tracking with estimated completion times.
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import time
import warnings
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def update(self, n=1):
            pass
import logging

from data_generation.sabr_params import SABRParams, GridConfig, ParameterSampler
from data_generation.sabr_mc_generator import SABRMCGenerator, MCConfig, MCResult, ParallelMCGenerator
from data_generation.hagan_surface_generator import HaganSurfaceGenerator, HaganConfig, HaganResult
from visualization.data_visualizer import DataVisualizer, VisualizationConfig


@dataclass
class DataGenerationConfig:
    """
    Configuration for data generation orchestrator.
    
    Attributes:
        output_dir: Base directory for generated data
        n_parameter_sets: Number of SABR parameter sets to generate
        sampling_strategy: Parameter sampling strategy ('uniform', 'lhs', 'stratified')
        mc_config: Monte Carlo configuration
        hagan_config: Hagan analytical configuration
        grid_config: Grid configuration for surfaces
        validation_enabled: Enable data quality validation
        save_intermediate: Save intermediate results during generation
        parallel_processing: Use parallel processing for multiple parameter sets
        n_workers: Number of worker processes (None = auto-detect)
        random_seed: Random seed for reproducibility
        create_visualizations: Generate visualization plots during data generation
        visualization_config: Configuration for visualizations
    """
    output_dir: str = "new/data"
    n_parameter_sets: int = 100
    sampling_strategy: str = "lhs"  # 'uniform', 'lhs', 'stratified'
    mc_config: MCConfig = None
    hagan_config: HaganConfig = None
    grid_config: GridConfig = None
    validation_enabled: bool = True
    save_intermediate: bool = True
    parallel_processing: bool = True
    n_workers: Optional[int] = None
    random_seed: Optional[int] = 42
    create_visualizations: bool = True
    visualization_config: VisualizationConfig = None
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.mc_config is None:
            self.mc_config = MCConfig(random_seed=self.random_seed)
        if self.hagan_config is None:
            self.hagan_config = HaganConfig()
        if self.grid_config is None:
            self.grid_config = GridConfig()
        if self.visualization_config is None:
            self.visualization_config = VisualizationConfig(
                output_dir=f"{self.output_dir}/plots"
            )


@dataclass
class DataGenerationResult:
    """
    Result from complete data generation process.
    
    Attributes:
        parameter_sets: List of SABR parameter sets used
        mc_results: List of Monte Carlo results
        hagan_results: List of Hagan analytical results
        validation_results: Data quality validation results
        generation_metadata: Metadata about the generation process
        file_paths: Paths to saved data files
        computation_time: Total computation time
        visualization_figures: Generated visualization figures (if created)
    """
    parameter_sets: List[SABRParams]
    mc_results: List[MCResult]
    hagan_results: List[HaganResult]
    validation_results: Dict[str, Any]
    generation_metadata: Dict[str, Any]
    file_paths: Dict[str, str]
    computation_time: float
    visualization_figures: Optional[Dict[str, Any]] = None


class DataQualityValidator:
    """
    Data quality validation and outlier detection for generated surfaces.
    
    Implements comprehensive validation checks to ensure generated data
    meets quality standards and identifies potential outliers or issues.
    """
    
    def __init__(self, tolerance_config: Dict[str, float] = None):
        """
        Initialize data quality validator.
        
        Args:
            tolerance_config: Dictionary of tolerance parameters for validation
        """
        self.tolerance_config = tolerance_config or {
            'max_volatility': 2.0,  # Maximum reasonable volatility
            'min_volatility': 0.001,  # Minimum reasonable volatility
            'atm_alpha_tolerance': 0.2,  # Tolerance for ATM vol vs alpha
            'residual_outlier_threshold': 3.0,  # Standard deviations for outlier detection
            'max_relative_error': 0.5,  # Maximum relative error between MC and Hagan
            'min_valid_points_ratio': 0.8  # Minimum ratio of valid points per surface
        }
    
    def validate_surface_data(self, mc_result: MCResult, hagan_result: HaganResult,
                            sabr_params: SABRParams) -> Dict[str, Any]:
        """
        Comprehensive validation of surface data quality.
        
        Args:
            mc_result: Monte Carlo result
            hagan_result: Hagan analytical result
            sabr_params: SABR parameters used
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'passed': True,
            'warnings': [],
            'errors': [],
            'checks': {},
            'quality_score': 1.0
        }
        
        # Check 1: Volatility bounds
        mc_vol_check = self._check_volatility_bounds(
            mc_result.volatility_surface, "Monte Carlo"
        )
        hagan_vol_check = self._check_volatility_bounds(
            hagan_result.volatility_surface, "Hagan"
        )
        
        validation_result['checks']['mc_volatility_bounds'] = mc_vol_check
        validation_result['checks']['hagan_volatility_bounds'] = hagan_vol_check
        
        if not mc_vol_check['passed'] or not hagan_vol_check['passed']:
            validation_result['passed'] = False
            validation_result['quality_score'] *= 0.7
        
        # Check 2: ATM volatility vs alpha consistency
        atm_check = self._check_atm_consistency(mc_result, hagan_result, sabr_params)
        validation_result['checks']['atm_consistency'] = atm_check
        
        if not atm_check['passed']:
            validation_result['warnings'].append(
                f"ATM volatility inconsistent with alpha: {atm_check['message']}"
            )
            validation_result['quality_score'] *= 0.9
        
        # Check 3: Data completeness
        completeness_check = self._check_data_completeness(mc_result, hagan_result)
        validation_result['checks']['data_completeness'] = completeness_check
        
        if not completeness_check['passed']:
            validation_result['warnings'].append(
                f"Data completeness issue: {completeness_check['message']}"
            )
            validation_result['quality_score'] *= 0.8
        
        # Check 4: Residual outlier detection
        outlier_check = self._detect_residual_outliers(mc_result, hagan_result)
        validation_result['checks']['residual_outliers'] = outlier_check
        
        if outlier_check['outlier_count'] > 0:
            validation_result['warnings'].append(
                f"Found {outlier_check['outlier_count']} residual outliers"
            )
            validation_result['quality_score'] *= (1 - 0.1 * outlier_check['outlier_ratio'])
        
        # Check 5: Surface smoothness
        smoothness_check = self._check_surface_smoothness(hagan_result)
        validation_result['checks']['surface_smoothness'] = smoothness_check
        
        if not smoothness_check['passed']:
            validation_result['warnings'].append(
                f"Surface smoothness issue: {smoothness_check['message']}"
            )
            validation_result['quality_score'] *= 0.9
        
        # Check 6: Numerical stability
        stability_check = self._check_numerical_stability(mc_result, hagan_result)
        validation_result['checks']['numerical_stability'] = stability_check
        
        if not stability_check['passed']:
            validation_result['errors'].append(
                f"Numerical stability issue: {stability_check['message']}"
            )
            validation_result['passed'] = False
            validation_result['quality_score'] *= 0.5
        
        # Overall quality assessment
        if validation_result['quality_score'] < 0.5:
            validation_result['passed'] = False
            validation_result['errors'].append("Overall quality score too low")
        
        return validation_result
    
    def _check_volatility_bounds(self, volatility_surface: np.ndarray, 
                               surface_type: str) -> Dict[str, Any]:
        """Check if volatilities are within reasonable bounds."""
        valid_vols = volatility_surface[~np.isnan(volatility_surface)]
        
        if len(valid_vols) == 0:
            return {
                'passed': False,
                'message': f"No valid volatilities in {surface_type} surface",
                'stats': {}
            }
        
        min_vol = np.min(valid_vols)
        max_vol = np.max(valid_vols)
        
        bounds_ok = (
            min_vol >= self.tolerance_config['min_volatility'] and
            max_vol <= self.tolerance_config['max_volatility']
        )
        
        negative_count = np.sum(valid_vols <= 0)
        infinite_count = np.sum(~np.isfinite(valid_vols))
        
        return {
            'passed': bounds_ok and negative_count == 0 and infinite_count == 0,
            'message': f"{surface_type} volatilities in range [{min_vol:.4f}, {max_vol:.4f}]",
            'stats': {
                'min_volatility': min_vol,
                'max_volatility': max_vol,
                'negative_count': negative_count,
                'infinite_count': infinite_count,
                'total_valid': len(valid_vols)
            }
        }
    
    def _check_atm_consistency(self, mc_result: MCResult, hagan_result: HaganResult,
                             sabr_params: SABRParams) -> Dict[str, Any]:
        """Check ATM volatility consistency with alpha parameter."""
        # Find ATM volatility for shortest maturity
        if len(mc_result.maturities) == 0:
            return {'passed': False, 'message': "No maturities available"}
        
        maturity_idx = 0  # Shortest maturity
        strikes = mc_result.strikes[maturity_idx]
        mc_vols = mc_result.volatility_surface[maturity_idx]
        hagan_vols = hagan_result.volatility_surface[maturity_idx]
        
        valid_mask = ~np.isnan(strikes) & ~np.isnan(mc_vols) & ~np.isnan(hagan_vols)
        
        if np.sum(valid_mask) == 0:
            return {'passed': False, 'message': "No valid ATM data"}
        
        valid_strikes = strikes[valid_mask]
        valid_mc_vols = mc_vols[valid_mask]
        valid_hagan_vols = hagan_vols[valid_mask]
        
        # Find closest to ATM
        atm_idx = np.argmin(np.abs(valid_strikes - sabr_params.F0))
        
        mc_atm_vol = valid_mc_vols[atm_idx]
        hagan_atm_vol = valid_hagan_vols[atm_idx]
        
        # Check consistency with alpha
        mc_error = abs(mc_atm_vol - sabr_params.alpha) / sabr_params.alpha
        hagan_error = abs(hagan_atm_vol - sabr_params.alpha) / sabr_params.alpha
        
        tolerance = self.tolerance_config['atm_alpha_tolerance']
        
        return {
            'passed': mc_error < tolerance and hagan_error < tolerance,
            'message': f"MC ATM error: {mc_error:.2%}, Hagan ATM error: {hagan_error:.2%}",
            'stats': {
                'mc_atm_volatility': mc_atm_vol,
                'hagan_atm_volatility': hagan_atm_vol,
                'alpha': sabr_params.alpha,
                'mc_relative_error': mc_error,
                'hagan_relative_error': hagan_error
            }
        }
    
    def _check_data_completeness(self, mc_result: MCResult, 
                               hagan_result: HaganResult) -> Dict[str, Any]:
        """Check data completeness and coverage."""
        mc_valid_ratio = np.sum(~np.isnan(mc_result.volatility_surface)) / mc_result.volatility_surface.size
        hagan_valid_ratio = np.sum(~np.isnan(hagan_result.volatility_surface)) / hagan_result.volatility_surface.size
        
        min_ratio = self.tolerance_config['min_valid_points_ratio']
        
        return {
            'passed': mc_valid_ratio >= min_ratio and hagan_valid_ratio >= min_ratio,
            'message': f"MC coverage: {mc_valid_ratio:.1%}, Hagan coverage: {hagan_valid_ratio:.1%}",
            'stats': {
                'mc_valid_ratio': mc_valid_ratio,
                'hagan_valid_ratio': hagan_valid_ratio,
                'mc_total_points': mc_result.volatility_surface.size,
                'hagan_total_points': hagan_result.volatility_surface.size
            }
        }
    
    def _detect_residual_outliers(self, mc_result: MCResult, 
                                hagan_result: HaganResult) -> Dict[str, Any]:
        """Detect outliers in MC-Hagan residuals."""
        residuals = mc_result.volatility_surface - hagan_result.volatility_surface
        valid_residuals = residuals[~np.isnan(residuals)]
        
        if len(valid_residuals) == 0:
            return {
                'outlier_count': 0,
                'outlier_ratio': 0.0,
                'message': "No valid residuals for outlier detection"
            }
        
        # Use z-score method for outlier detection
        mean_residual = np.mean(valid_residuals)
        std_residual = np.std(valid_residuals)
        
        if std_residual == 0:
            return {
                'outlier_count': 0,
                'outlier_ratio': 0.0,
                'message': "Zero residual variance"
            }
        
        z_scores = np.abs((valid_residuals - mean_residual) / std_residual)
        outliers = z_scores > self.tolerance_config['residual_outlier_threshold']
        
        outlier_count = np.sum(outliers)
        outlier_ratio = outlier_count / len(valid_residuals)
        
        return {
            'outlier_count': outlier_count,
            'outlier_ratio': outlier_ratio,
            'message': f"Found {outlier_count} outliers ({outlier_ratio:.1%})",
            'stats': {
                'mean_residual': mean_residual,
                'std_residual': std_residual,
                'max_z_score': np.max(z_scores),
                'threshold': self.tolerance_config['residual_outlier_threshold']
            }
        }
    
    def _check_surface_smoothness(self, hagan_result: HaganResult) -> Dict[str, Any]:
        """Check volatility surface smoothness."""
        # Check for excessive oscillations in volatility smiles
        smoothness_issues = 0
        total_smiles = 0
        
        for i in range(len(hagan_result.maturities)):
            strikes = hagan_result.strikes[i]
            vols = hagan_result.volatility_surface[i]
            
            valid_mask = ~np.isnan(strikes) & ~np.isnan(vols)
            
            if np.sum(valid_mask) >= 3:
                valid_strikes = strikes[valid_mask]
                valid_vols = vols[valid_mask]
                
                # Sort by strike
                sort_idx = np.argsort(valid_strikes)
                sorted_vols = valid_vols[sort_idx]
                
                # Check for excessive sign changes in first derivative
                vol_diffs = np.diff(sorted_vols)
                sign_changes = np.sum(np.diff(np.sign(vol_diffs)) != 0)
                
                # Allow some oscillation but flag excessive changes
                if sign_changes > len(vol_diffs) // 2:
                    smoothness_issues += 1
                
                total_smiles += 1
        
        smoothness_ratio = 1 - (smoothness_issues / max(total_smiles, 1))
        
        return {
            'passed': smoothness_ratio >= 0.8,
            'message': f"Smoothness issues in {smoothness_issues}/{total_smiles} smiles",
            'stats': {
                'smoothness_ratio': smoothness_ratio,
                'issues_count': smoothness_issues,
                'total_smiles': total_smiles
            }
        }
    
    def _check_numerical_stability(self, mc_result: MCResult, 
                                 hagan_result: HaganResult) -> Dict[str, Any]:
        """Check for numerical stability issues."""
        # Check for NaN/Inf propagation
        mc_nan_count = np.sum(np.isnan(mc_result.volatility_surface))
        mc_inf_count = np.sum(np.isinf(mc_result.volatility_surface))
        
        hagan_nan_count = np.sum(np.isnan(hagan_result.volatility_surface))
        hagan_inf_count = np.sum(np.isinf(hagan_result.volatility_surface))
        
        total_points = mc_result.volatility_surface.size
        
        # Allow some NaN values (e.g., extreme strikes) but not too many
        nan_ratio = (mc_nan_count + hagan_nan_count) / (2 * total_points)
        inf_ratio = (mc_inf_count + hagan_inf_count) / (2 * total_points)
        
        stability_ok = nan_ratio < 0.3 and inf_ratio < 0.01
        
        return {
            'passed': stability_ok,
            'message': f"NaN ratio: {nan_ratio:.1%}, Inf ratio: {inf_ratio:.1%}",
            'stats': {
                'mc_nan_count': mc_nan_count,
                'mc_inf_count': mc_inf_count,
                'hagan_nan_count': hagan_nan_count,
                'hagan_inf_count': hagan_inf_count,
                'nan_ratio': nan_ratio,
                'inf_ratio': inf_ratio
            }
        }


class DataOrchestrator:
    """
    Main orchestrator for SABR volatility surface data generation.
    
    Coordinates Monte Carlo and Hagan surface generation, implements
    comprehensive data validation, and provides organized data storage
    with progress tracking and visualization capabilities.
    """
    
    def __init__(self, config: DataGenerationConfig):
        """
        Initialize data generation orchestrator.
        
        Args:
            config: Data generation configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        self.validator = DataQualityValidator()
        
        # Create output directories
        self._create_output_directories()
        
        # Initialize generators
        self.mc_generator = SABRMCGenerator(config.mc_config)
        self.hagan_generator = HaganSurfaceGenerator(config.hagan_config)
        self.parameter_sampler = ParameterSampler(config.random_seed)
        
        if config.create_visualizations:
            self.visualizer = DataVisualizer(config.visualization_config)
        else:
            self.visualizer = None
    
    def generate_complete_dataset(self) -> DataGenerationResult:
        """
        Generate complete SABR volatility surface dataset.
        
        Returns:
            DataGenerationResult with all generated data and metadata
        """
        start_time = time.time()
        
        self.logger.info(f"Starting data generation with {self.config.n_parameter_sets} parameter sets")
        
        # Step 1: Generate parameter sets
        self.logger.info("Generating SABR parameter sets...")
        parameter_sets = self._generate_parameter_sets()
        
        # Step 2: Generate surfaces
        self.logger.info("Generating volatility surfaces...")
        mc_results, hagan_results = self._generate_surfaces(parameter_sets)
        
        # Step 3: Validate data quality
        validation_results = {}
        if self.config.validation_enabled:
            self.logger.info("Validating data quality...")
            validation_results = self._validate_all_data(parameter_sets, mc_results, hagan_results)
        
        # Step 4: Save data
        self.logger.info("Saving generated data...")
        file_paths = self._save_all_data(parameter_sets, mc_results, hagan_results, validation_results)
        
        # Step 5: Create visualizations
        visualization_figures = None
        if self.config.create_visualizations and self.visualizer:
            self.logger.info("Creating visualizations...")
            visualization_figures = self._create_visualizations(parameter_sets, mc_results, hagan_results)
        
        # Step 6: Generate metadata
        computation_time = time.time() - start_time
        metadata = self._generate_metadata(parameter_sets, mc_results, hagan_results, computation_time)
        
        self.logger.info(f"Data generation completed in {computation_time:.2f} seconds")
        
        return DataGenerationResult(
            parameter_sets=parameter_sets,
            mc_results=mc_results,
            hagan_results=hagan_results,
            validation_results=validation_results,
            generation_metadata=metadata,
            file_paths=file_paths,
            computation_time=computation_time,
            visualization_figures=visualization_figures
        )    

    def _generate_parameter_sets(self) -> List[SABRParams]:
        """Generate SABR parameter sets using specified sampling strategy."""
        if self.config.sampling_strategy == "uniform":
            return self.parameter_sampler.uniform_sampling(self.config.n_parameter_sets)
        elif self.config.sampling_strategy == "lhs":
            return self.parameter_sampler.latin_hypercube_sampling(self.config.n_parameter_sets)
        elif self.config.sampling_strategy == "stratified":
            return self.parameter_sampler.stratified_sampling(self.config.n_parameter_sets)
        elif self.config.sampling_strategy == "funahashi_exact":
            return self.parameter_sampler.funahashi_exact_sampling(self.config.n_parameter_sets)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.config.sampling_strategy}")
    
    def _generate_surfaces(self, parameter_sets: List[SABRParams]) -> Tuple[List[MCResult], List[HaganResult]]:
        """Generate MC and Hagan surfaces for all parameter sets."""
        if self.config.parallel_processing and len(parameter_sets) > 1:
            return self._generate_surfaces_parallel(parameter_sets)
        else:
            return self._generate_surfaces_sequential(parameter_sets)
    
    def _generate_surfaces_sequential(self, parameter_sets: List[SABRParams]) -> Tuple[List[MCResult], List[HaganResult]]:
        """Generate surfaces sequentially with progress tracking."""
        mc_results = []
        hagan_results = []
        
        with tqdm(total=len(parameter_sets), desc="Generating surfaces") as pbar:
            for i, params in enumerate(parameter_sets):
                try:
                    # Generate MC surface
                    mc_result = self.mc_generator.generate_volatility_surface(params, self.config.grid_config)
                    
                    # Generate Hagan surface
                    hagan_result = self.hagan_generator.generate_volatility_surface(params, self.config.grid_config)
                    
                    mc_results.append(mc_result)
                    hagan_results.append(hagan_result)
                    
                    # Save intermediate results if requested
                    if self.config.save_intermediate:
                        self._save_intermediate_result(i, params, mc_result, hagan_result)
                    
                    pbar.update(1)
                    
                except Exception as e:
                    self.logger.error(f"Error generating surfaces for parameter set {i}: {e}")
                    # Create dummy results to maintain indexing
                    dummy_mc = self._create_dummy_mc_result()
                    dummy_hagan = self._create_dummy_hagan_result()
                    mc_results.append(dummy_mc)
                    hagan_results.append(dummy_hagan)
                    pbar.update(1)
        
        return mc_results, hagan_results
    
    def _generate_surfaces_parallel(self, parameter_sets: List[SABRParams]) -> Tuple[List[MCResult], List[HaganResult]]:
        """Generate surfaces in parallel using multiprocessing."""
        n_workers = self.config.n_workers or min(mp.cpu_count(), len(parameter_sets))
        
        self.logger.info(f"Using {n_workers} workers for parallel surface generation")
        
        # Create parallel generators
        parallel_mc = ParallelMCGenerator(self.config.mc_config, n_workers)
        
        # Generate MC surfaces in parallel
        self.logger.info("Generating MC surfaces in parallel...")
        mc_results = parallel_mc.generate_surfaces_parallel(parameter_sets, self.config.grid_config)
        
        # Generate Hagan surfaces (these are fast, so we can do them sequentially)
        self.logger.info("Generating Hagan surfaces...")
        hagan_results = []
        
        with tqdm(total=len(parameter_sets), desc="Generating Hagan surfaces") as pbar:
            for params in parameter_sets:
                try:
                    hagan_result = self.hagan_generator.generate_volatility_surface(params, self.config.grid_config)
                    hagan_results.append(hagan_result)
                except Exception as e:
                    self.logger.error(f"Error generating Hagan surface: {e}")
                    hagan_results.append(self._create_dummy_hagan_result())
                pbar.update(1)
        
        return mc_results, hagan_results
    
    def _validate_all_data(self, parameter_sets: List[SABRParams], 
                          mc_results: List[MCResult], 
                          hagan_results: List[HaganResult]) -> Dict[str, Any]:
        """Validate all generated data for quality and consistency."""
        validation_summary = {
            'total_sets': len(parameter_sets),
            'passed_sets': 0,
            'failed_sets': 0,
            'warning_sets': 0,
            'individual_results': [],
            'overall_quality_score': 0.0,
            'common_issues': {}
        }
        
        quality_scores = []
        issue_counts = {}
        
        self.logger.info("Validating data quality...")
        
        with tqdm(total=len(parameter_sets), desc="Validating data") as pbar:
            for i, (params, mc_res, hagan_res) in enumerate(zip(parameter_sets, mc_results, hagan_results)):
                try:
                    validation_result = self.validator.validate_surface_data(mc_res, hagan_res, params)
                    validation_result['parameter_set_index'] = i
                    
                    validation_summary['individual_results'].append(validation_result)
                    quality_scores.append(validation_result['quality_score'])
                    
                    if validation_result['passed']:
                        validation_summary['passed_sets'] += 1
                    else:
                        validation_summary['failed_sets'] += 1
                    
                    if validation_result['warnings']:
                        validation_summary['warning_sets'] += 1
                    
                    # Count common issues
                    for warning in validation_result['warnings']:
                        issue_type = warning.split(':')[0]  # Extract issue type
                        issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
                    
                except Exception as e:
                    self.logger.error(f"Error validating parameter set {i}: {e}")
                    validation_summary['failed_sets'] += 1
                    quality_scores.append(0.0)
                
                pbar.update(1)
        
        # Calculate overall statistics
        validation_summary['overall_quality_score'] = np.mean(quality_scores) if quality_scores else 0.0
        validation_summary['common_issues'] = dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True))
        
        # Log summary
        self.logger.info(f"Validation complete: {validation_summary['passed_sets']}/{validation_summary['total_sets']} sets passed")
        self.logger.info(f"Overall quality score: {validation_summary['overall_quality_score']:.3f}")
        
        return validation_summary
    
    def _save_all_data(self, parameter_sets: List[SABRParams], 
                      mc_results: List[MCResult], 
                      hagan_results: List[HaganResult],
                      validation_results: Dict[str, Any]) -> Dict[str, str]:
        """Save all generated data with organized file structure."""
        file_paths = {}
        
        # Create timestamp for this generation run
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = f"{self.config.output_dir}/run_{timestamp}"
        os.makedirs(run_dir, exist_ok=True)
        
        # Save parameter sets
        params_file = f"{run_dir}/parameter_sets.pkl"
        with open(params_file, 'wb') as f:
            pickle.dump(parameter_sets, f)
        file_paths['parameter_sets'] = params_file
        
        # Save parameter sets as CSV for easy inspection
        params_csv = f"{run_dir}/parameter_sets.csv"
        params_df = pd.DataFrame([asdict(params) for params in parameter_sets])
        params_df.to_csv(params_csv, index=False)
        file_paths['parameter_sets_csv'] = params_csv
        
        # Save MC results
        mc_file = f"{run_dir}/mc_results.pkl"
        with open(mc_file, 'wb') as f:
            pickle.dump(mc_results, f)
        file_paths['mc_results'] = mc_file
        
        # Save Hagan results
        hagan_file = f"{run_dir}/hagan_results.pkl"
        with open(hagan_file, 'wb') as f:
            pickle.dump(hagan_results, f)
        file_paths['hagan_results'] = hagan_file
        
        # Save validation results
        validation_file = f"{run_dir}/validation_results.json"
        with open(validation_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            validation_json = self._convert_for_json(validation_results)
            json.dump(validation_json, f, indent=2)
        file_paths['validation_results'] = validation_file
        
        # Save configuration
        config_file = f"{run_dir}/generation_config.json"
        with open(config_file, 'w') as f:
            config_dict = asdict(self.config)
            config_json = self._convert_for_json(config_dict)
            json.dump(config_json, f, indent=2)
        file_paths['config'] = config_file
        
        # Create organized data structure for ML training
        self._create_ml_ready_data(run_dir, parameter_sets, mc_results, hagan_results)
        file_paths['ml_data_dir'] = f"{run_dir}/ml_ready"
        
        self.logger.info(f"Data saved to {run_dir}")
        
        return file_paths
    
    def _create_ml_ready_data(self, run_dir: str, parameter_sets: List[SABRParams],
                            mc_results: List[MCResult], hagan_results: List[HaganResult]):
        """Create ML-ready data structure for training."""
        ml_dir = f"{run_dir}/ml_ready"
        os.makedirs(ml_dir, exist_ok=True)
        
        # Prepare training data
        training_data = []
        
        for i, (params, mc_res, hagan_res) in enumerate(zip(parameter_sets, mc_results, hagan_results)):
            # Calculate residuals
            residuals = mc_res.volatility_surface - hagan_res.volatility_surface
            
            # Extract valid data points
            for mat_idx in range(len(mc_res.maturities)):
                maturity = mc_res.maturities[mat_idx]
                
                for strike_idx in range(mc_res.strikes.shape[1]):
                    if (strike_idx < len(mc_res.strikes[mat_idx]) and 
                        not np.isnan(mc_res.strikes[mat_idx, strike_idx]) and
                        not np.isnan(residuals[mat_idx, strike_idx])):
                        
                        strike = mc_res.strikes[mat_idx, strike_idx]
                        mc_vol = mc_res.volatility_surface[mat_idx, strike_idx]
                        hagan_vol = hagan_res.volatility_surface[mat_idx, strike_idx]
                        residual = residuals[mat_idx, strike_idx]
                        
                        training_data.append({
                            'param_set_id': i,
                            'F0': params.F0,
                            'alpha': params.alpha,
                            'beta': params.beta,
                            'nu': params.nu,
                            'rho': params.rho,
                            'strike': strike,
                            'maturity': maturity,
                            'moneyness': strike / params.F0,
                            'mc_volatility': mc_vol,
                            'hagan_volatility': hagan_vol,
                            'residual': residual
                        })
        
        # Save as CSV and HDF5
        training_df = pd.DataFrame(training_data)
        
        # CSV for inspection
        training_df.to_csv(f"{ml_dir}/training_data.csv", index=False)
        
        # HDF5 for efficient loading (if tables is available)
        try:
            training_df.to_hdf(f"{ml_dir}/training_data.h5", key='data', mode='w')
        except ImportError:
            self.logger.warning("pytables not available, skipping HDF5 export")
        
        # Save summary statistics
        summary_stats = {
            'total_data_points': len(training_data),
            'parameter_sets': len(parameter_sets),
            'residual_statistics': {
                'mean': float(training_df['residual'].mean()),
                'std': float(training_df['residual'].std()),
                'min': float(training_df['residual'].min()),
                'max': float(training_df['residual'].max()),
                'percentiles': {
                    '5%': float(training_df['residual'].quantile(0.05)),
                    '25%': float(training_df['residual'].quantile(0.25)),
                    '50%': float(training_df['residual'].quantile(0.50)),
                    '75%': float(training_df['residual'].quantile(0.75)),
                    '95%': float(training_df['residual'].quantile(0.95))
                }
            },
            'moneyness_range': {
                'min': float(training_df['moneyness'].min()),
                'max': float(training_df['moneyness'].max())
            },
            'maturity_range': {
                'min': float(training_df['maturity'].min()),
                'max': float(training_df['maturity'].max())
            }
        }
        
        with open(f"{ml_dir}/data_summary.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
    
    def _create_visualizations(self, parameter_sets: List[SABRParams],
                             mc_results: List[MCResult], 
                             hagan_results: List[HaganResult]) -> Dict[str, Any]:
        """Create comprehensive visualizations of generated data."""
        if not self.visualizer:
            return {}
        
        try:
            # Create comprehensive report
            figures = self.visualizer.create_comprehensive_report(
                parameter_sets, mc_results, hagan_results
            )
            
            # Save summary statistics
            summary_df = self.visualizer.save_summary_statistics(
                parameter_sets, mc_results, hagan_results
            )
            
            return {
                'figures': figures,
                'summary_statistics': summary_df
            }
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
            return {}
    
    def _generate_metadata(self, parameter_sets: List[SABRParams],
                          mc_results: List[MCResult], 
                          hagan_results: List[HaganResult],
                          computation_time: float) -> Dict[str, Any]:
        """Generate comprehensive metadata about the generation process."""
        metadata = {
            'generation_info': {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'total_computation_time': computation_time,
                'n_parameter_sets': len(parameter_sets),
                'sampling_strategy': self.config.sampling_strategy,
                'parallel_processing': self.config.parallel_processing,
                'n_workers': self.config.n_workers or mp.cpu_count()
            },
            'configuration': {
                'mc_config': asdict(self.config.mc_config),
                'hagan_config': asdict(self.config.hagan_config),
                'grid_config': asdict(self.config.grid_config)
            },
            'data_statistics': {},
            'performance_metrics': {}
        }
        
        # Calculate data statistics
        if mc_results and hagan_results:
            # MC statistics
            mc_times = [res.computation_time for res in mc_results if hasattr(res, 'computation_time')]
            mc_valid_points = [np.sum(~np.isnan(res.volatility_surface)) for res in mc_results]
            
            # Hagan statistics
            hagan_times = [res.computation_time for res in hagan_results if hasattr(res, 'computation_time')]
            hagan_valid_points = [np.sum(~np.isnan(res.volatility_surface)) for res in hagan_results]
            
            # Residual statistics
            all_residuals = []
            for mc_res, hagan_res in zip(mc_results, hagan_results):
                residuals = mc_res.volatility_surface - hagan_res.volatility_surface
                valid_residuals = residuals[~np.isnan(residuals)]
                all_residuals.extend(valid_residuals)
            
            metadata['data_statistics'] = {
                'mc_computation_times': {
                    'mean': float(np.mean(mc_times)) if mc_times else 0.0,
                    'std': float(np.std(mc_times)) if mc_times else 0.0,
                    'total': float(np.sum(mc_times)) if mc_times else 0.0
                },
                'hagan_computation_times': {
                    'mean': float(np.mean(hagan_times)) if hagan_times else 0.0,
                    'std': float(np.std(hagan_times)) if hagan_times else 0.0,
                    'total': float(np.sum(hagan_times)) if hagan_times else 0.0
                },
                'data_coverage': {
                    'mc_avg_valid_points': float(np.mean(mc_valid_points)) if mc_valid_points else 0.0,
                    'hagan_avg_valid_points': float(np.mean(hagan_valid_points)) if hagan_valid_points else 0.0
                },
                'residual_statistics': {
                    'count': len(all_residuals),
                    'mean': float(np.mean(all_residuals)) if all_residuals else 0.0,
                    'std': float(np.std(all_residuals)) if all_residuals else 0.0,
                    'rmse': float(np.sqrt(np.mean(np.array(all_residuals)**2))) if all_residuals else 0.0
                }
            }
        
        # Performance metrics
        if computation_time > 0:
            metadata['performance_metrics'] = {
                'surfaces_per_second': len(parameter_sets) / computation_time,
                'avg_time_per_surface': computation_time / len(parameter_sets) if parameter_sets else 0.0,
                'parallel_efficiency': self._calculate_parallel_efficiency(mc_results, hagan_results)
            }
        
        return metadata
    
    def _calculate_parallel_efficiency(self, mc_results: List[MCResult], 
                                     hagan_results: List[HaganResult]) -> float:
        """Calculate parallel processing efficiency."""
        if not self.config.parallel_processing:
            return 1.0
        
        # Simple efficiency estimate based on computation times
        mc_times = [res.computation_time for res in mc_results if hasattr(res, 'computation_time')]
        
        if not mc_times:
            return 1.0
        
        # Theoretical sequential time vs actual parallel time
        theoretical_sequential = sum(mc_times)
        actual_parallel = max(mc_times) if mc_times else 0.0
        
        if actual_parallel > 0:
            return min(1.0, theoretical_sequential / (actual_parallel * (self.config.n_workers or 1)))
        else:
            return 1.0
    
    def _save_intermediate_result(self, index: int, params: SABRParams, 
                                mc_result: MCResult, hagan_result: HaganResult):
        """Save intermediate results during generation."""
        intermediate_dir = f"{self.config.output_dir}/intermediate"
        os.makedirs(intermediate_dir, exist_ok=True)
        
        result_data = {
            'index': index,
            'parameters': asdict(params),
            'mc_result': mc_result,
            'hagan_result': hagan_result
        }
        
        with open(f"{intermediate_dir}/result_{index:04d}.pkl", 'wb') as f:
            pickle.dump(result_data, f)
    
    def _create_dummy_mc_result(self) -> MCResult:
        """Create dummy MC result for failed computations."""
        return MCResult(
            strikes=np.array([]),
            maturities=np.array([]),
            volatility_surface=np.array([]),
            option_prices=np.array([]),
            convergence_info={'error': 'Failed computation'},
            computation_time=0.0
        )
    
    def _create_dummy_hagan_result(self) -> HaganResult:
        """Create dummy Hagan result for failed computations."""
        return HaganResult(
            strikes=np.array([]),
            maturities=np.array([]),
            volatility_surface=np.array([]),
            computation_time=0.0,
            numerical_warnings=['Failed computation'],
            grid_info={}
        )
    
    def _convert_for_json(self, obj: Any) -> Any:
        """Convert numpy types and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, '__dict__'):
            return self._convert_for_json(asdict(obj) if hasattr(obj, '__dataclass_fields__') else obj.__dict__)
        else:
            return obj
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the orchestrator."""
        logger = logging.getLogger('DataOrchestrator')
        logger.setLevel(logging.INFO)
        
        # Create console handler if not already exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_output_directories(self):
        """Create necessary output directories."""
        directories = [
            self.config.output_dir,
            f"{self.config.output_dir}/raw",
            f"{self.config.output_dir}/processed",
            f"{self.config.output_dir}/intermediate",
            self.config.visualization_config.output_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def load_existing_data(self, run_dir: str) -> DataGenerationResult:
        """
        Load previously generated data from a run directory.
        
        Args:
            run_dir: Directory containing saved data
            
        Returns:
            DataGenerationResult with loaded data
        """
        self.logger.info(f"Loading data from {run_dir}")
        
        # Load parameter sets
        with open(f"{run_dir}/parameter_sets.pkl", 'rb') as f:
            parameter_sets = pickle.load(f)
        
        # Load MC results
        with open(f"{run_dir}/mc_results.pkl", 'rb') as f:
            mc_results = pickle.load(f)
        
        # Load Hagan results
        with open(f"{run_dir}/hagan_results.pkl", 'rb') as f:
            hagan_results = pickle.load(f)
        
        # Load validation results
        with open(f"{run_dir}/validation_results.json", 'r') as f:
            validation_results = json.load(f)
        
        # Load configuration
        with open(f"{run_dir}/generation_config.json", 'r') as f:
            config_dict = json.load(f)
        
        # Generate file paths
        file_paths = {
            'parameter_sets': f"{run_dir}/parameter_sets.pkl",
            'mc_results': f"{run_dir}/mc_results.pkl",
            'hagan_results': f"{run_dir}/hagan_results.pkl",
            'validation_results': f"{run_dir}/validation_results.json",
            'config': f"{run_dir}/generation_config.json",
            'ml_data_dir': f"{run_dir}/ml_ready"
        }
        
        return DataGenerationResult(
            parameter_sets=parameter_sets,
            mc_results=mc_results,
            hagan_results=hagan_results,
            validation_results=validation_results,
            generation_metadata=config_dict,
            file_paths=file_paths,
            computation_time=0.0,  # Not tracked for loaded data
            visualization_figures=None
        )


# Utility functions for data generation

def create_default_config(n_parameter_sets: int = 100, 
                         output_dir: str = "new/data",
                         parallel: bool = True) -> DataGenerationConfig:
    """
    Create default data generation configuration.
    
    Args:
        n_parameter_sets: Number of parameter sets to generate
        output_dir: Output directory for data
        parallel: Use parallel processing
        
    Returns:
        DataGenerationConfig with sensible defaults
    """
    return DataGenerationConfig(
        output_dir=output_dir,
        n_parameter_sets=n_parameter_sets,
        sampling_strategy="lhs",
        parallel_processing=parallel,
        validation_enabled=True,
        create_visualizations=True,
        save_intermediate=False  # Disable for large datasets
    )


def quick_generation_example(n_sets: int = 10) -> DataGenerationResult:
    """
    Quick example of data generation for testing.
    
    Args:
        n_sets: Number of parameter sets (small for quick testing)
        
    Returns:
        DataGenerationResult
    """
    config = create_default_config(
        n_parameter_sets=n_sets,
        output_dir="new/data/quick_test",
        parallel=False  # Disable parallel for small datasets
    )
    
    # Use faster MC configuration for testing
    config.mc_config.n_paths = 10000  # Reduced for speed
    config.mc_config.n_steps = 100    # Reduced for speed
    
    orchestrator = DataOrchestrator(config)
    return orchestrator.generate_complete_dataset()


if __name__ == "__main__":
    # Example usage
    print("Running quick data generation example...")
    result = quick_generation_example(n_sets=5)
    print(f"Generated {len(result.parameter_sets)} parameter sets")
    print(f"Computation time: {result.computation_time:.2f} seconds")
    print(f"Data saved to: {result.file_paths['parameter_sets']}")