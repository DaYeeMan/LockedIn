"""
Test suite for evaluation metrics and model comparison functionality.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import pytest
from pathlib import Path
import tempfile
import shutil

from metrics import (
    FunahashiMetrics, ModelEvaluator, EvaluationResults, WingRegionAnalyzer
)
from model_comparison import (
    ModelComparator, ModelComparisonConfig, ModelResults, 
    FunahashiComparisonTable, HFBudgetComparison
)
from evaluation_pipeline import EvaluationPipeline


class TestFunahashiMetrics:
    """Test Funahashi's exact metrics implementation."""
    
    def test_mean_squared_error(self):
        """Test MSE calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9])
        
        expected_mse = np.mean((y_true - y_pred) ** 2)
        calculated_mse = FunahashiMetrics.mean_squared_error(y_true, y_pred)
        
        assert np.isclose(calculated_mse, expected_mse)
    
    def test_root_mean_squared_error(self):
        """Test RMSE calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9])
        
        expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        calculated_rmse = FunahashiMetrics.root_mean_squared_error(y_true, y_pred)
        
        assert np.isclose(calculated_rmse, expected_rmse)
    
    def test_mean_absolute_error(self):
        """Test MAE calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9])
        
        expected_mae = np.mean(np.abs(y_true - y_pred))
        calculated_mae = FunahashiMetrics.mean_absolute_error(y_true, y_pred)
        
        assert np.isclose(calculated_mae, expected_mae)
    
    def test_relative_percentage_error(self):
        """Test relative percentage error calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9])
        
        expected_rpe = np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100
        calculated_rpe = FunahashiMetrics.relative_percentage_error(y_true, y_pred)
        
        assert np.isclose(calculated_rpe, expected_rpe)
    
    def test_relative_percentage_error_with_zeros(self):
        """Test relative percentage error with zero values."""
        y_true = np.array([0.0, 2.0, 3.0, 4.0])
        y_pred = np.array([0.1, 1.9, 3.1, 3.9])
        
        # Should handle zeros gracefully with epsilon
        rpe = FunahashiMetrics.relative_percentage_error(y_true, y_pred)
        assert not np.isnan(rpe)
        assert not np.isinf(rpe)


class TestModelEvaluator:
    """Test ModelEvaluator functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.evaluator = ModelEvaluator()
        self.y_true = np.array([0.2, 0.25, 0.3, 0.35, 0.4])
        self.y_pred = np.array([0.21, 0.24, 0.31, 0.34, 0.39])
    
    def test_evaluate_model(self):
        """Test complete model evaluation."""
        results = self.evaluator.evaluate_model(self.y_true, self.y_pred)
        
        assert isinstance(results, EvaluationResults)
        assert results.mse > 0
        assert results.rmse > 0
        assert results.mae > 0
        assert results.relative_percentage_error > 0
        assert not np.isnan(results.mse)
    
    def test_evaluate_by_region(self):
        """Test regional evaluation."""
        strikes = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        forward_price = 1.0
        
        regional_results = self.evaluator.evaluate_by_region(
            self.y_true, self.y_pred, strikes, forward_price
        )
        
        assert 'ITM' in regional_results
        assert 'ATM' in regional_results
        assert 'OTM' in regional_results
        
        for region_result in regional_results.values():
            assert isinstance(region_result, EvaluationResults)


class TestWingRegionAnalyzer:
    """Test wing region analysis functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.analyzer = WingRegionAnalyzer(wing_threshold_percentile=80)
        self.residuals = np.array([0.001, 0.002, 0.05, 0.001, 0.08])  # Wing regions have large residuals
        self.strikes = np.array([0.5, 0.8, 1.0, 1.2, 2.0])
        self.forward_price = 1.0
    
    def test_identify_wing_regions(self):
        """Test wing region identification."""
        wing_mask = self.analyzer.identify_wing_regions(
            self.residuals, self.strikes, self.forward_price
        )
        
        assert isinstance(wing_mask, np.ndarray)
        assert wing_mask.dtype == bool
        assert len(wing_mask) == len(self.residuals)
        
        # Should identify points with largest residuals as wings
        assert wing_mask[2] or wing_mask[4]  # Points with 0.05 and 0.08 residuals
    
    def test_evaluate_wing_performance(self):
        """Test wing performance evaluation."""
        y_true = np.array([0.2, 0.25, 0.3, 0.35, 0.4])
        y_pred = np.array([0.21, 0.24, 0.31, 0.34, 0.39])
        
        wing_results = self.analyzer.evaluate_wing_performance(
            y_true, y_pred, self.residuals, self.strikes, self.forward_price
        )
        
        assert isinstance(wing_results, dict)
        # Should have wing and/or non-wing results
        assert len(wing_results) > 0


def create_dummy_model():
    """Create a simple dummy model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def create_dummy_dataset():
    """Create a dummy dataset for testing."""
    n_samples = 100
    x_data = np.random.randn(n_samples, 8)
    y_data = np.random.randn(n_samples, 1) * 0.01
    
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    return dataset.batch(16)


class TestModelComparator:
    """Test ModelComparator functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.config = ModelComparisonConfig(hf_budget=100)
        self.comparator = ModelComparator(self.config)
        self.test_data = create_dummy_dataset()
    
    def test_evaluate_single_model(self):
        """Test single model evaluation."""
        model = create_dummy_model()
        
        results = self.comparator.evaluate_single_model(
            model=model,
            test_data=self.test_data,
            model_name="test_model"
        )
        
        assert isinstance(results, ModelResults)
        assert results.model_name == "test_model"
        assert isinstance(results.overall_results, EvaluationResults)
        assert results.model_parameters > 0
        assert results.inference_time > 0
    
    def test_compare_models(self):
        """Test multi-model comparison."""
        models = {
            'model_1': create_dummy_model(),
            'model_2': create_dummy_model()
        }
        
        comparison_results = self.comparator.compare_models(
            models=models,
            test_data=self.test_data
        )
        
        assert isinstance(comparison_results, dict)
        assert len(comparison_results) == 2
        assert 'model_1' in comparison_results
        assert 'model_2' in comparison_results
        
        for result in comparison_results.values():
            assert isinstance(result, ModelResults)


class TestFunahashiComparisonTable:
    """Test comparison table generation."""
    
    def setup_method(self):
        """Set up test data."""
        self.table_generator = FunahashiComparisonTable()
        
        # Create dummy comparison results
        dummy_results = EvaluationResults(
            mse=0.001, rmse=0.032, mae=0.025,
            relative_percentage_error=2.5, max_absolute_error=0.1,
            mean_relative_error=0.02, std_relative_error=0.01
        )
        
        self.comparison_results = {
            'MDA-CNN': ModelResults(
                model_name='MDA-CNN',
                overall_results=dummy_results,
                regional_results={},
                model_parameters=10000,
                inference_time=0.5
            ),
            'Funahashi_Baseline': ModelResults(
                model_name='Funahashi_Baseline',
                overall_results=dummy_results,
                regional_results={},
                model_parameters=5000,
                inference_time=0.3
            )
        }
    
    def test_create_funahashi_format_table(self):
        """Test Funahashi format table creation."""
        table = self.table_generator.create_funahashi_format_table(self.comparison_results)
        
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 2
        assert 'Model' in table.columns
        assert 'MSE' in table.columns
        assert 'RMSE' in table.columns
        assert 'MAE' in table.columns
        assert 'Rel. % Error' in table.columns
    
    def test_save_comparison_tables(self):
        """Test saving comparison tables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = self.table_generator.save_comparison_tables(
                self.comparison_results, Path(temp_dir), "test_experiment"
            )
            
            assert 'comprehensive' in saved_files
            assert 'funahashi_format' in saved_files
            assert 'raw_json' in saved_files
            
            # Check files exist
            for file_path in saved_files.values():
                assert Path(file_path).exists()


class TestEvaluationPipeline:
    """Test EvaluationPipeline functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = EvaluationPipeline(output_dir=self.temp_dir)
        self.test_data = create_dummy_dataset()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_run_single_comparison(self):
        """Test single comparison run."""
        models = {'test_model': create_dummy_model()}
        
        results = self.pipeline.run_single_comparison(
            models=models,
            test_data=self.test_data,
            experiment_name="test_experiment",
            save_results=True
        )
        
        assert isinstance(results, dict)
        assert 'test_model' in results
        
        # Check that files were saved
        output_files = list(Path(self.temp_dir).glob("test_experiment*"))
        assert len(output_files) > 0
    
    def test_generate_funahashi_comparison_report(self):
        """Test report generation."""
        # Create dummy results
        dummy_results = EvaluationResults(
            mse=0.001, rmse=0.032, mae=0.025,
            relative_percentage_error=2.5, max_absolute_error=0.1,
            mean_relative_error=0.02, std_relative_error=0.01
        )
        
        comparison_results = {
            'test_model': ModelResults(
                model_name='test_model',
                overall_results=dummy_results,
                regional_results={},
                model_parameters=1000,
                inference_time=0.1
            )
        }
        
        report = self.pipeline.generate_funahashi_comparison_report(
            comparison_results=comparison_results,
            experiment_name="test_report"
        )
        
        assert isinstance(report, dict)
        assert 'experiment_name' in report
        assert 'summary_table' in report
        assert 'file_paths' in report
        
        # Check that report files were created
        for file_path in report['file_paths'].values():
            assert Path(file_path).exists()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])