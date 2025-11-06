"""
Evaluation pipeline for comparing MDA-CNN against Funahashi baseline.

This module provides a high-level interface for running comprehensive
model comparisons using the same HF budget constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import tensorflow as tf
from pathlib import Path
import json
import time
from datetime import datetime

from metrics import ModelEvaluator, EvaluationResults
from model_comparison import (
    ModelComparator, ModelComparisonConfig, ModelResults,
    FunahashiComparisonTable, HFBudgetComparison
)


class EvaluationPipeline:
    """
    High-level evaluation pipeline for comprehensive model comparison.
    """
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize evaluation pipeline.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.comparator = ModelComparator()
        self.table_generator = FunahashiComparisonTable()
        self.budget_comparator = HFBudgetComparison()
        
        self.results_history = []
    
    def run_single_comparison(self, 
                            models: Dict[str, tf.keras.Model],
                            test_data: tf.data.Dataset,
                            experiment_name: str = None,
                            hf_budget: int = 200,
                            strikes: Optional[np.ndarray] = None,
                            forward_prices: Optional[np.ndarray] = None,
                            residuals: Optional[np.ndarray] = None,
                            save_results: bool = True) -> Dict[str, ModelResults]:
        """
        Run a single model comparison experiment.
        
        Args:
            models: Dictionary of model_name -> trained_model
            test_data: Test dataset
            experiment_name: Name for this experiment
            hf_budget: HF budget constraint
            strikes: Strike prices for regional analysis
            forward_prices: Forward prices for regional analysis
            residuals: MC-Hagan residuals for wing analysis
            save_results: Whether to save results to files
            
        Returns:
            Dictionary of model_name -> ModelResults
        """
        if experiment_name is None:
            experiment_name = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"Running model comparison: {experiment_name}")
        print(f"HF Budget: {hf_budget}")
        print(f"Models: {list(models.keys())}")
        
        # Configure comparator
        config = ModelComparisonConfig(hf_budget=hf_budget)
        self.comparator.config = config
        
        # Run comparison
        start_time = time.time()
        comparison_results = self.comparator.compare_models(
            models=models,
            test_data=test_data,
            strikes=strikes,
            forward_prices=forward_prices,
            residuals=residuals
        )
        total_time = time.time() - start_time
        
        print(f"Comparison completed in {total_time:.2f} seconds")
        
        # Save results if requested
        if save_results:
            self._save_comparison_results(
                comparison_results, experiment_name, 
                additional_info={'hf_budget': hf_budget, 'total_time': total_time}
            )
        
        # Store in history
        self.results_history.append({
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'hf_budget': hf_budget,
            'models': list(models.keys()),
            'results': comparison_results
        })
        
        return comparison_results
    
    def run_budget_comparison(self,
                            models: Dict[str, tf.keras.Model],
                            test_data: tf.data.Dataset,
                            hf_budgets: List[int],
                            experiment_name: str = None,
                            strikes: Optional[np.ndarray] = None,
                            forward_prices: Optional[np.ndarray] = None,
                            residuals: Optional[np.ndarray] = None,
                            save_results: bool = True) -> Dict[int, Dict[str, ModelResults]]:
        """
        Run model comparison across multiple HF budget constraints.
        
        Args:
            models: Dictionary of model_name -> trained_model
            test_data: Test dataset
            hf_budgets: List of HF budget values to test
            experiment_name: Name for this experiment
            strikes: Strike prices for regional analysis
            forward_prices: Forward prices for regional analysis
            residuals: MC-Hagan residuals for wing analysis
            save_results: Whether to save results to files
            
        Returns:
            Dictionary of budget -> {model_name -> ModelResults}
        """
        if experiment_name is None:
            experiment_name = f"budget_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"Running budget comparison: {experiment_name}")
        print(f"HF Budgets: {hf_budgets}")
        print(f"Models: {list(models.keys())}")
        
        # Run budget comparison
        start_time = time.time()
        budget_results = self.budget_comparator.compare_with_budget_constraints(
            models=models,
            test_data=test_data,
            hf_budgets=hf_budgets,
            strikes=strikes,
            forward_prices=forward_prices,
            residuals=residuals
        )
        total_time = time.time() - start_time
        
        print(f"Budget comparison completed in {total_time:.2f} seconds")
        
        # Save results if requested
        if save_results:
            self._save_budget_comparison_results(
                budget_results, experiment_name,
                additional_info={'hf_budgets': hf_budgets, 'total_time': total_time}
            )
        
        return budget_results
    
    def generate_funahashi_comparison_report(self,
                                           comparison_results: Dict[str, ModelResults],
                                           experiment_name: str,
                                           include_detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive comparison report in Funahashi's format.
        
        Args:
            comparison_results: Results from model comparison
            experiment_name: Name for the report
            include_detailed_analysis: Whether to include detailed regional/wing analysis
            
        Returns:
            Dictionary containing report data and file paths
        """
        print(f"Generating Funahashi comparison report: {experiment_name}")
        
        report_data = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'models_compared': list(comparison_results.keys()),
            'summary_table': None,
            'detailed_tables': {},
            'file_paths': {}
        }
        
        # Generate main comparison table
        funahashi_table = self.table_generator.create_funahashi_format_table(comparison_results)
        report_data['summary_table'] = funahashi_table
        
        # Save main table
        main_table_path = self.output_dir / f"{experiment_name}_funahashi_comparison.csv"
        funahashi_table.to_csv(main_table_path, index=False)
        report_data['file_paths']['main_table'] = main_table_path
        
        # Generate detailed analysis if requested
        if include_detailed_analysis:
            # Comprehensive table with regions and wings
            detailed_table = self.table_generator.create_comparison_table(
                comparison_results, include_regions=True, include_wings=True
            )
            report_data['detailed_tables']['comprehensive'] = detailed_table
            
            # Save detailed table
            detailed_path = self.output_dir / f"{experiment_name}_detailed_comparison.csv"
            detailed_table.to_csv(detailed_path, index=False)
            report_data['file_paths']['detailed_table'] = detailed_path
        
        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(comparison_results)
        report_data['summary_statistics'] = summary_stats
        
        # Save summary statistics
        stats_path = self.output_dir / f"{experiment_name}_summary_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        report_data['file_paths']['summary_stats'] = stats_path
        
        # Save complete report
        report_path = self.output_dir / f"{experiment_name}_complete_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        report_data['file_paths']['complete_report'] = report_path
        
        print(f"Report generated and saved to: {self.output_dir}")
        return report_data
    
    def _save_comparison_results(self, 
                               comparison_results: Dict[str, ModelResults],
                               experiment_name: str,
                               additional_info: Dict[str, Any] = None):
        """Save comparison results to files."""
        # Save using table generator
        saved_files = self.table_generator.save_comparison_tables(
            comparison_results, self.output_dir, experiment_name
        )
        
        # Save additional experiment info
        if additional_info:
            info_path = self.output_dir / f"{experiment_name}_experiment_info.json"
            with open(info_path, 'w') as f:
                json.dump(additional_info, f, indent=2, default=str)
            saved_files['experiment_info'] = info_path
        
        print(f"Results saved to: {list(saved_files.values())}")
    
    def _save_budget_comparison_results(self,
                                      budget_results: Dict[int, Dict[str, ModelResults]],
                                      experiment_name: str,
                                      additional_info: Dict[str, Any] = None):
        """Save budget comparison results to files."""
        # Create budget performance table
        budget_table = self.budget_comparator.create_budget_performance_table(budget_results)
        budget_path = self.output_dir / f"{experiment_name}_budget_performance.csv"
        budget_table.to_csv(budget_path, index=False)
        
        # Save detailed results for each budget
        for budget, results in budget_results.items():
            budget_experiment_name = f"{experiment_name}_budget_{budget}"
            self._save_comparison_results(results, budget_experiment_name)
        
        # Save additional info
        if additional_info:
            info_path = self.output_dir / f"{experiment_name}_budget_experiment_info.json"
            with open(info_path, 'w') as f:
                json.dump(additional_info, f, indent=2, default=str)
        
        print(f"Budget comparison results saved to: {self.output_dir}")
    
    def _generate_summary_statistics(self, 
                                   comparison_results: Dict[str, ModelResults]) -> Dict[str, Any]:
        """Generate summary statistics for the comparison."""
        stats = {
            'model_count': len(comparison_results),
            'models': list(comparison_results.keys()),
            'best_model_by_metric': {},
            'performance_gaps': {},
            'model_info': {}
        }
        
        # Find best model for each metric
        metrics = ['mse', 'rmse', 'mae', 'relative_percentage_error']
        
        for metric in metrics:
            metric_values = {}
            for model_name, results in comparison_results.items():
                metric_value = getattr(results.overall_results, metric)
                if not np.isnan(metric_value):
                    metric_values[model_name] = metric_value
            
            if metric_values:
                best_model = min(metric_values.items(), key=lambda x: x[1])
                stats['best_model_by_metric'][metric] = {
                    'model': best_model[0],
                    'value': best_model[1]
                }
                
                # Calculate performance gaps
                worst_value = max(metric_values.values())
                best_value = best_model[1]
                gap = ((worst_value - best_value) / best_value) * 100 if best_value > 0 else 0
                stats['performance_gaps'][metric] = gap
        
        # Collect model info
        for model_name, results in comparison_results.items():
            stats['model_info'][model_name] = {
                'parameters': results.model_parameters,
                'inference_time': results.inference_time,
                'has_regional_results': bool(results.regional_results),
                'has_wing_results': bool(results.wing_results)
            }
        
        return stats
    
    def print_comparison_summary(self, comparison_results: Dict[str, ModelResults]):
        """Print a formatted summary of comparison results."""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Create summary table
        funahashi_table = self.table_generator.create_funahashi_format_table(comparison_results)
        print(funahashi_table.to_string(index=False))
        
        print("\n" + "-"*80)
        print("KEY FINDINGS:")
        
        # Find best performing model for each metric
        metrics = ['MSE', 'RMSE', 'MAE', 'Rel. % Error']
        for metric in metrics:
            if metric in funahashi_table.columns:
                # Convert back to numeric for comparison
                numeric_values = pd.to_numeric(funahashi_table[metric], errors='coerce')
                if not numeric_values.isna().all():
                    best_idx = numeric_values.idxmin()
                    best_model = funahashi_table.loc[best_idx, 'Model']
                    best_value = funahashi_table.loc[best_idx, metric]
                    print(f"Best {metric}: {best_model} ({best_value})")
        
        print("="*80)