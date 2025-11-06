"""
Comprehensive test of the evaluation system functionality.
"""

import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
import shutil

# Add evaluation module to path
sys.path.append(str(Path(__file__).parent / "evaluation"))

from metrics import FunahashiMetrics, ModelEvaluator, EvaluationResults
from model_comparison import ModelComparator, ModelComparisonConfig, FunahashiComparisonTable
from evaluation_pipeline import EvaluationPipeline


def create_dummy_model(name="test_model"):
    """Create a simple dummy model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear', name=f'{name}_output')
    ], name=name)
    model.compile(optimizer='adam', loss='mse')
    return model


def create_dummy_dataset(n_samples=100):
    """Create a dummy dataset for testing."""
    x_data = np.random.randn(n_samples, 8)
    y_data = np.random.randn(n_samples, 1) * 0.01
    
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    return dataset.batch(16)


def test_complete_evaluation_pipeline():
    """Test the complete evaluation pipeline."""
    print("Testing Complete Evaluation Pipeline")
    print("=" * 50)
    
    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize pipeline
        pipeline = EvaluationPipeline(output_dir=temp_dir)
        
        # Create dummy models
        models = {
            'MDA-CNN': create_dummy_model('mda_cnn'),
            'Funahashi_Baseline': create_dummy_model('funahashi'),
            'Simple_MLP': create_dummy_model('simple_mlp')
        }
        
        print(f"Created {len(models)} test models")
        
        # Create test data
        test_data = create_dummy_dataset(200)
        
        # Create dummy metadata for regional analysis
        n_samples = 200
        strikes = np.random.uniform(0.5, 2.0, n_samples)
        forward_prices = np.ones(n_samples)
        residuals = np.random.randn(n_samples) * 0.02
        
        print("Created test dataset and metadata")
        
        # Run single comparison
        print("\n1. Running single model comparison...")
        comparison_results = pipeline.run_single_comparison(
            models=models,
            test_data=test_data,
            experiment_name="test_comparison",
            hf_budget=200,
            strikes=strikes,
            forward_prices=forward_prices,
            residuals=residuals,
            save_results=True
        )
        
        print(f"✓ Comparison completed for {len(comparison_results)} models")
        
        # Generate Funahashi format report
        print("\n2. Generating Funahashi format report...")
        report = pipeline.generate_funahashi_comparison_report(
            comparison_results=comparison_results,
            experiment_name="test_funahashi_report",
            include_detailed_analysis=True
        )
        
        print("✓ Funahashi format report generated")
        print(f"Report files: {list(report['file_paths'].keys())}")
        
        # Print summary
        print("\n3. Comparison Summary:")
        pipeline.print_comparison_summary(comparison_results)
        
        # Test budget comparison
        print("\n4. Running budget comparison...")
        budget_results = pipeline.run_budget_comparison(
            models={'Funahashi_Baseline': models['Funahashi_Baseline'], 
                   'Simple_MLP': models['Simple_MLP']},
            test_data=test_data,
            hf_budgets=[50, 100, 200],
            experiment_name="test_budget_comparison",
            strikes=strikes,
            forward_prices=forward_prices,
            residuals=residuals,
            save_results=True
        )
        
        print(f"✓ Budget comparison completed for {len(budget_results)} budgets")
        
        # Verify output files were created
        output_files = list(Path(temp_dir).glob("*"))
        print(f"\n5. Generated {len(output_files)} output files:")
        for file_path in output_files[:5]:  # Show first 5 files
            print(f"   - {file_path.name}")
        if len(output_files) > 5:
            print(f"   ... and {len(output_files) - 5} more files")
        
        print("\n✅ Complete evaluation pipeline test passed!")
        
        return comparison_results, report, budget_results
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


def test_funahashi_metrics_accuracy():
    """Test that Funahashi metrics match expected calculations."""
    print("\nTesting Funahashi Metrics Accuracy")
    print("=" * 40)
    
    # Create test data with known results
    y_true = np.array([0.2, 0.25, 0.3, 0.35, 0.4])
    y_pred = np.array([0.21, 0.24, 0.31, 0.34, 0.39])
    
    # Calculate expected values manually
    expected_mse = np.mean((y_true - y_pred) ** 2)
    expected_rmse = np.sqrt(expected_mse)
    expected_mae = np.mean(np.abs(y_true - y_pred))
    expected_rpe = np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100
    
    # Calculate using our implementation
    calculated_mse = FunahashiMetrics.mean_squared_error(y_true, y_pred)
    calculated_rmse = FunahashiMetrics.root_mean_squared_error(y_true, y_pred)
    calculated_mae = FunahashiMetrics.mean_absolute_error(y_true, y_pred)
    calculated_rpe = FunahashiMetrics.relative_percentage_error(y_true, y_pred)
    
    # Verify accuracy
    assert np.isclose(calculated_mse, expected_mse), f"MSE mismatch: {calculated_mse} vs {expected_mse}"
    assert np.isclose(calculated_rmse, expected_rmse), f"RMSE mismatch: {calculated_rmse} vs {expected_rmse}"
    assert np.isclose(calculated_mae, expected_mae), f"MAE mismatch: {calculated_mae} vs {expected_mae}"
    assert np.isclose(calculated_rpe, expected_rpe), f"RPE mismatch: {calculated_rpe} vs {expected_rpe}"
    
    print(f"✓ MSE: {calculated_mse:.6f} (expected: {expected_mse:.6f})")
    print(f"✓ RMSE: {calculated_rmse:.6f} (expected: {expected_rmse:.6f})")
    print(f"✓ MAE: {calculated_mae:.6f} (expected: {expected_mae:.6f})")
    print(f"✓ Rel % Error: {calculated_rpe:.6f} (expected: {expected_rpe:.6f})")
    
    print("✅ All Funahashi metrics accuracy tests passed!")


def test_comparison_table_generation():
    """Test comparison table generation."""
    print("\nTesting Comparison Table Generation")
    print("=" * 40)
    
    # Create dummy results
    dummy_results = EvaluationResults(
        mse=0.001, rmse=0.032, mae=0.025,
        relative_percentage_error=2.5, max_absolute_error=0.1,
        mean_relative_error=0.02, std_relative_error=0.01
    )
    
    from model_comparison import ModelResults
    comparison_results = {
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
    
    # Test table generation
    table_generator = FunahashiComparisonTable()
    
    # Generate Funahashi format table
    funahashi_table = table_generator.create_funahashi_format_table(comparison_results)
    
    print("Generated Funahashi Format Table:")
    print(funahashi_table.to_string(index=False))
    
    # Verify table structure
    assert len(funahashi_table) == 2, f"Expected 2 rows, got {len(funahashi_table)}"
    assert 'Model' in funahashi_table.columns
    assert 'MSE' in funahashi_table.columns
    assert 'RMSE' in funahashi_table.columns
    assert 'MAE' in funahashi_table.columns
    
    print("\n✅ Comparison table generation test passed!")


if __name__ == "__main__":
    print("SABR Volatility Surface Model Evaluation - Comprehensive Test")
    print("=" * 70)
    
    try:
        # Test individual components
        test_funahashi_metrics_accuracy()
        test_comparison_table_generation()
        
        # Test complete pipeline
        comparison_results, report, budget_results = test_complete_evaluation_pipeline()
        
        print("\n" + "=" * 70)
        print("🎉 ALL EVALUATION TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)
        print("\nThe evaluation system is ready for:")
        print("- Comparing MDA-CNN against Funahashi baseline")
        print("- Generating Funahashi-format comparison tables")
        print("- Regional and wing analysis")
        print("- HF budget constraint comparisons")
        print("- Comprehensive reporting and visualization")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()