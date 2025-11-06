"""
Model comparison utilities for comparing MDA-CNN against Funahashi baseline.

This module provides functionality to compare different models using the same
HF budget and generate comparison tables matching Funahashi's paper format.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import tensorflow as tf
from pathlib import Path
import json
from dataclasses import dataclass, asdict

from metrics import ModelEvaluator, EvaluationResults, WingRegionAnalyzer


@dataclass
class ModelComparisonConfig:
    """Configuration for model comparison experiments."""
    hf_budget: int = 200
    test_size: float = 0.2
    random_seed: int = 42
    evaluation_regions: List[str] = None
    wing_analysis: bool = True
    wing_threshold_percentile: float = 90
    
    def __post_init__(self):
        if self.evaluation_regions is None:
            self.evaluation_regions = ['Overall', 'ITM', 'ATM', 'OTM']


@dataclass
class ModelResults:
    """Container for individual model evaluation results."""
    model_name: str
    overall_results: EvaluationResults
    regional_results: Dict[str, EvaluationResults]
    wing_results: Optional[Dict[str, EvaluationResults]] = None
    training_time: Optional[float] = None
    inference_time: Optional[float] = None
    model_parameters: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result_dict = {
            'model_name': self.model_name,
            'overall_results': self.overall_results.to_dict(),
            'regional_results': {k: v.to_dict() for k, v in self.regional_results.items()},
            'training_time': self.training_time,
            'inference_time': self.inference_time,
            'model_parameters': self.model_parameters
        }
        
        if self.wing_results:
            result_dict['wing_results'] = {k: v.to_dict() for k, v in self.wing_results.items()}
        
        return result_dict


class ModelComparator:
    """
    Main class for comparing MDA-CNN against Funahashi baseline and other models.
    """
    
    def __init__(self, config: ModelComparisonConfig = None):
        """
        Initialize model comparator.
        
        Args:
            config: Configuration for comparison experiments
        """
        self.config = config or ModelComparisonConfig()
        self.evaluator = ModelEvaluator()
        self.wing_analyzer = WingRegionAnalyzer(
            wing_threshold_percentile=self.config.wing_threshold_percentile
        )
        self.comparison_results = {}
    
    def evaluate_single_model(self, model: tf.keras.Model, 
                            test_data: tf.data.Dataset,
                            model_name: str,
                            strikes: Optional[np.ndarray] = None,
                            forward_prices: Optional[np.ndarray] = None,
                            residuals: Optional[np.ndarray] = None) -> ModelResults:
        """
        Evaluate a single model comprehensively.
        
        Args:
            model: Trained TensorFlow model
            test_data: Test dataset
            model_name: Name identifier for the model
            strikes: Strike prices for regional analysis
            forward_prices: Forward prices for regional analysis
            residuals: MC-Hagan residuals for wing analysis
            
        Returns:
            ModelResults containing comprehensive evaluation
        """
        import time
        
        # Collect all predictions and true values
        y_true_list = []
        y_pred_list = []
        
        # Measure inference time
        start_time = time.time()
        
        for batch in test_data:
            if isinstance(batch, tuple) and len(batch) == 2:
                x_batch, y_batch = batch
            else:
                raise ValueError("Expected test_data to yield (x, y) tuples")
            
            # Get predictions
            y_pred_batch = model(x_batch, training=False)
            
            # Convert to numpy and collect
            y_true_list.append(y_batch.numpy())
            y_pred_list.append(y_pred_batch.numpy())
        
        inference_time = time.time() - start_time
        
        # Concatenate all batches
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        
        # Overall evaluation
        overall_results = self.evaluator.evaluate_model(y_true, y_pred)
        
        # Regional evaluation (if strike and forward price data available)
        regional_results = {}
        if strikes is not None and forward_prices is not None:
            # Ensure arrays have same length as predictions
            if len(strikes) == len(y_true) and len(forward_prices) == len(y_true):
                for i, forward_price in enumerate(np.unique(forward_prices)):
                    fp_mask = forward_prices == forward_price
                    if np.any(fp_mask):
                        regional_eval = self.evaluator.evaluate_by_region(
                            y_true[fp_mask], y_pred[fp_mask], 
                            strikes[fp_mask], forward_price
                        )
                        for region, results in regional_eval.items():
                            if region not in regional_results:
                                regional_results[region] = []
                            regional_results[region].append(results)
                
                # Average results across different forward prices
                for region in regional_results:
                    if regional_results[region]:
                        # Average the metrics
                        avg_results = self._average_evaluation_results(regional_results[region])
                        regional_results[region] = avg_results
        
        # Wing analysis (if residual data available)
        wing_results = None
        if self.config.wing_analysis and residuals is not None:
            if len(residuals) == len(y_true) and strikes is not None and forward_prices is not None:
                wing_results = {}
                for i, forward_price in enumerate(np.unique(forward_prices)):
                    fp_mask = forward_prices == forward_price
                    if np.any(fp_mask):
                        wing_eval = self.wing_analyzer.evaluate_wing_performance(
                            y_true[fp_mask], y_pred[fp_mask],
                            residuals[fp_mask], strikes[fp_mask], forward_price
                        )
                        for region, results in wing_eval.items():
                            if region not in wing_results:
                                wing_results[region] = []
                            wing_results[region].append(results)
                
                # Average wing results
                for region in wing_results:
                    if wing_results[region]:
                        avg_results = self._average_evaluation_results(wing_results[region])
                        wing_results[region] = avg_results
        
        # Count model parameters
        model_parameters = model.count_params() if hasattr(model, 'count_params') else None
        
        return ModelResults(
            model_name=model_name,
            overall_results=overall_results,
            regional_results=regional_results,
            wing_results=wing_results,
            inference_time=inference_time,
            model_parameters=model_parameters
        )  
  
    def compare_models(self, models: Dict[str, tf.keras.Model],
                      test_data: tf.data.Dataset,
                      strikes: Optional[np.ndarray] = None,
                      forward_prices: Optional[np.ndarray] = None,
                      residuals: Optional[np.ndarray] = None) -> Dict[str, ModelResults]:
        """
        Compare multiple models using the same test data.
        
        Args:
            models: Dictionary of model_name -> model
            test_data: Test dataset
            strikes: Strike prices for regional analysis
            forward_prices: Forward prices for regional analysis
            residuals: MC-Hagan residuals for wing analysis
            
        Returns:
            Dictionary of model_name -> ModelResults
        """
        comparison_results = {}
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            
            # Reset test data iterator for each model
            test_data_copy = test_data
            
            model_results = self.evaluate_single_model(
                model=model,
                test_data=test_data_copy,
                model_name=model_name,
                strikes=strikes,
                forward_prices=forward_prices,
                residuals=residuals
            )
            
            comparison_results[model_name] = model_results
        
        self.comparison_results = comparison_results
        return comparison_results
    
    def _average_evaluation_results(self, results_list: List[EvaluationResults]) -> EvaluationResults:
        """
        Average multiple EvaluationResults objects.
        
        Args:
            results_list: List of EvaluationResults to average
            
        Returns:
            Averaged EvaluationResults
        """
        if not results_list:
            return EvaluationResults(
                mse=np.nan, rmse=np.nan, mae=np.nan,
                relative_percentage_error=np.nan, max_absolute_error=np.nan,
                mean_relative_error=np.nan, std_relative_error=np.nan
            )
        
        # Convert to dictionaries and average
        dicts = [result.to_dict() for result in results_list]
        averaged_dict = {}
        
        for key in dicts[0].keys():
            values = [d[key] for d in dicts if not np.isnan(d[key])]
            averaged_dict[key] = np.mean(values) if values else np.nan
        
        return EvaluationResults(
            mse=averaged_dict['MSE'],
            rmse=averaged_dict['RMSE'],
            mae=averaged_dict['MAE'],
            relative_percentage_error=averaged_dict['Relative_Percentage_Error'],
            max_absolute_error=averaged_dict['Max_Absolute_Error'],
            mean_relative_error=averaged_dict['Mean_Relative_Error'],
            std_relative_error=averaged_dict['Std_Relative_Error']
        )


class FunahashiComparisonTable:
    """
    Generate comparison tables matching Funahashi's paper format.
    """
    
    def __init__(self):
        self.metric_names = {
            'MSE': 'MSE',
            'RMSE': 'RMSE', 
            'MAE': 'MAE',
            'Relative_Percentage_Error': 'Rel. % Error'
        }
    
    def create_comparison_table(self, comparison_results: Dict[str, ModelResults],
                              include_regions: bool = True,
                              include_wings: bool = True) -> pd.DataFrame:
        """
        Create a comprehensive comparison table.
        
        Args:
            comparison_results: Results from ModelComparator.compare_models()
            include_regions: Whether to include regional breakdowns
            include_wings: Whether to include wing analysis
            
        Returns:
            Formatted comparison DataFrame
        """
        rows = []
        
        for model_name, results in comparison_results.items():
            # Overall results
            overall_row = self._create_result_row(
                model_name, 'Overall', results.overall_results
            )
            rows.append(overall_row)
            
            # Regional results
            if include_regions and results.regional_results:
                for region, regional_result in results.regional_results.items():
                    regional_row = self._create_result_row(
                        model_name, region, regional_result
                    )
                    rows.append(regional_row)
            
            # Wing results
            if include_wings and results.wing_results:
                for wing_region, wing_result in results.wing_results.items():
                    wing_row = self._create_result_row(
                        model_name, f'Wing_{wing_region}', wing_result
                    )
                    rows.append(wing_row)
        
        df = pd.DataFrame(rows)
        return df
    
    def create_funahashi_format_table(self, comparison_results: Dict[str, ModelResults]) -> pd.DataFrame:
        """
        Create a table specifically matching Funahashi's paper format.
        
        Args:
            comparison_results: Results from ModelComparator.compare_models()
            
        Returns:
            DataFrame formatted like Funahashi's comparison tables
        """
        # Focus on key metrics that Funahashi reports
        key_metrics = ['MSE', 'Relative_Percentage_Error', 'RMSE', 'MAE']
        
        table_data = []
        
        for model_name, results in comparison_results.items():
            row = {'Model': model_name}
            
            # Add overall metrics
            overall_dict = results.overall_results.to_dict()
            for metric in key_metrics:
                if metric in overall_dict:
                    row[self.metric_names.get(metric, metric)] = overall_dict[metric]
            
            # Add model info
            if results.model_parameters:
                row['Parameters'] = results.model_parameters
            if results.inference_time:
                row['Inference_Time_s'] = results.inference_time
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Format numbers for readability
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'Parameters':
                df[col] = df[col].apply(lambda x: f"{x:.6f}" if not np.isnan(x) else "N/A")
        
        return df   
 
    def _create_result_row(self, model_name: str, region: str, 
                          results: EvaluationResults) -> Dict[str, Any]:
        """
        Create a single row for the comparison table.
        
        Args:
            model_name: Name of the model
            region: Region name (Overall, ITM, ATM, OTM, etc.)
            results: EvaluationResults for this model/region
            
        Returns:
            Dictionary representing a table row
        """
        row = {
            'Model': model_name,
            'Region': region
        }
        
        # Add all metrics
        results_dict = results.to_dict()
        for metric, value in results_dict.items():
            row[metric] = value
        
        return row
    
    def save_comparison_tables(self, comparison_results: Dict[str, ModelResults],
                             output_dir: Path, 
                             experiment_name: str = "model_comparison") -> Dict[str, Path]:
        """
        Save comparison tables to files.
        
        Args:
            comparison_results: Results from ModelComparator.compare_models()
            output_dir: Directory to save tables
            experiment_name: Name for the experiment files
            
        Returns:
            Dictionary of table_type -> file_path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Comprehensive table
        comprehensive_table = self.create_comparison_table(comparison_results)
        comprehensive_path = output_dir / f"{experiment_name}_comprehensive.csv"
        comprehensive_table.to_csv(comprehensive_path, index=False)
        saved_files['comprehensive'] = comprehensive_path
        
        # Funahashi format table
        funahashi_table = self.create_funahashi_format_table(comparison_results)
        funahashi_path = output_dir / f"{experiment_name}_funahashi_format.csv"
        funahashi_table.to_csv(funahashi_path, index=False)
        saved_files['funahashi_format'] = funahashi_path
        
        # Save raw results as JSON
        raw_results = {name: result.to_dict() for name, result in comparison_results.items()}
        json_path = output_dir / f"{experiment_name}_raw_results.json"
        with open(json_path, 'w') as f:
            json.dump(raw_results, f, indent=2, default=str)
        saved_files['raw_json'] = json_path
        
        return saved_files


class HFBudgetComparison:
    """
    Compare models using different HF budget constraints.
    """
    
    def __init__(self):
        self.comparator = ModelComparator()
    
    def compare_with_budget_constraints(self, models: Dict[str, tf.keras.Model],
                                      test_data: tf.data.Dataset,
                                      hf_budgets: List[int],
                                      strikes: Optional[np.ndarray] = None,
                                      forward_prices: Optional[np.ndarray] = None,
                                      residuals: Optional[np.ndarray] = None) -> Dict[int, Dict[str, ModelResults]]:
        """
        Compare models across different HF budget constraints.
        
        Args:
            models: Dictionary of model_name -> model
            test_data: Test dataset
            hf_budgets: List of HF budget values to test
            strikes: Strike prices for regional analysis
            forward_prices: Forward prices for regional analysis
            residuals: MC-Hagan residuals for wing analysis
            
        Returns:
            Dictionary of budget -> {model_name -> ModelResults}
        """
        budget_results = {}
        
        for budget in hf_budgets:
            print(f"Evaluating models with HF budget: {budget}")
            
            # Update config for this budget
            config = ModelComparisonConfig(hf_budget=budget)
            comparator = ModelComparator(config)
            
            # Compare models with this budget
            budget_comparison = comparator.compare_models(
                models=models,
                test_data=test_data,
                strikes=strikes,
                forward_prices=forward_prices,
                residuals=residuals
            )
            
            budget_results[budget] = budget_comparison
        
        return budget_results
    
    def create_budget_performance_table(self, budget_results: Dict[int, Dict[str, ModelResults]]) -> pd.DataFrame:
        """
        Create a table showing performance vs HF budget.
        
        Args:
            budget_results: Results from compare_with_budget_constraints()
            
        Returns:
            DataFrame with budget performance comparison
        """
        rows = []
        
        for budget, model_results in budget_results.items():
            for model_name, results in model_results.items():
                row = {
                    'HF_Budget': budget,
                    'Model': model_name,
                    'MSE': results.overall_results.mse,
                    'RMSE': results.overall_results.rmse,
                    'MAE': results.overall_results.mae,
                    'Rel_Pct_Error': results.overall_results.relative_percentage_error
                }
                rows.append(row)
        
        return pd.DataFrame(rows)