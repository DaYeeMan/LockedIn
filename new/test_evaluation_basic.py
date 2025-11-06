"""
Basic test of evaluation functionality.
"""

import sys
import numpy as np
from pathlib import Path

# Add evaluation module to path
sys.path.append(str(Path(__file__).parent / "evaluation"))

from metrics import FunahashiMetrics, ModelEvaluator, EvaluationResults

def test_funahashi_metrics():
    """Test basic Funahashi metrics."""
    print("Testing Funahashi Metrics...")
    
    # Create test data
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9])
    
    # Test metrics
    mse = FunahashiMetrics.mean_squared_error(y_true, y_pred)
    rmse = FunahashiMetrics.root_mean_squared_error(y_true, y_pred)
    mae = FunahashiMetrics.mean_absolute_error(y_true, y_pred)
    rpe = FunahashiMetrics.relative_percentage_error(y_true, y_pred)
    
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Relative % Error: {rpe:.6f}")
    
    # Verify calculations
    expected_mse = np.mean((y_true - y_pred) ** 2)
    assert np.isclose(mse, expected_mse), f"MSE mismatch: {mse} vs {expected_mse}"
    
    expected_rmse = np.sqrt(expected_mse)
    assert np.isclose(rmse, expected_rmse), f"RMSE mismatch: {rmse} vs {expected_rmse}"
    
    expected_mae = np.mean(np.abs(y_true - y_pred))
    assert np.isclose(mae, expected_mae), f"MAE mismatch: {mae} vs {expected_mae}"
    
    print("✓ All Funahashi metrics tests passed!")


def test_model_evaluator():
    """Test ModelEvaluator functionality."""
    print("\nTesting ModelEvaluator...")
    
    evaluator = ModelEvaluator()
    
    # Create test data
    y_true = np.array([0.2, 0.25, 0.3, 0.35, 0.4])
    y_pred = np.array([0.21, 0.24, 0.31, 0.34, 0.39])
    
    # Test evaluation
    results = evaluator.evaluate_model(y_true, y_pred)
    
    assert isinstance(results, EvaluationResults)
    assert results.mse > 0
    assert results.rmse > 0
    assert results.mae > 0
    assert results.relative_percentage_error > 0
    assert not np.isnan(results.mse)
    
    print(f"MSE: {results.mse:.6f}")
    print(f"RMSE: {results.rmse:.6f}")
    print(f"MAE: {results.mae:.6f}")
    print(f"Relative % Error: {results.relative_percentage_error:.6f}")
    
    # Test regional evaluation
    strikes = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    forward_price = 1.0
    
    regional_results = evaluator.evaluate_by_region(
        y_true, y_pred, strikes, forward_price
    )
    
    assert 'ITM' in regional_results
    assert 'ATM' in regional_results
    assert 'OTM' in regional_results
    
    print("Regional results:")
    for region, result in regional_results.items():
        print(f"  {region}: MSE={result.mse:.6f}, MAE={result.mae:.6f}")
    
    print("✓ All ModelEvaluator tests passed!")


def test_evaluation_results():
    """Test EvaluationResults data structure."""
    print("\nTesting EvaluationResults...")
    
    results = EvaluationResults(
        mse=0.001,
        rmse=0.032,
        mae=0.025,
        relative_percentage_error=2.5,
        max_absolute_error=0.1,
        mean_relative_error=0.02,
        std_relative_error=0.01
    )
    
    # Test to_dict conversion
    results_dict = results.to_dict()
    
    assert isinstance(results_dict, dict)
    assert 'MSE' in results_dict
    assert 'RMSE' in results_dict
    assert 'MAE' in results_dict
    assert 'Relative_Percentage_Error' in results_dict
    
    assert results_dict['MSE'] == 0.001
    assert results_dict['RMSE'] == 0.032
    
    print("EvaluationResults dict:", results_dict)
    print("✓ EvaluationResults tests passed!")


if __name__ == "__main__":
    print("Running Basic Evaluation Tests")
    print("=" * 50)
    
    try:
        test_funahashi_metrics()
        test_model_evaluator()
        test_evaluation_results()
        
        print("\n" + "=" * 50)
        print("✓ All basic evaluation tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()