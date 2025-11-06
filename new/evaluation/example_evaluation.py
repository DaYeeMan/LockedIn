"""
Example script demonstrating comprehensive model evaluation and comparison.

This script shows how to use the evaluation pipeline to compare MDA-CNN
against Funahashi baseline using the same HF budget constraints.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.evaluation_pipeline import EvaluationPipeline
from evaluation.model_comparison import ModelComparisonConfig
from models.mda_cnn import MDACNN
from models.funahashi_baseline import FunahashiBaseline
from preprocessing.data_loader import create_test_dataset


def create_dummy_models():
    """Create dummy models for demonstration."""
    # MDA-CNN model
    mda_cnn = MDACNN(
        patch_size=(9, 9),
        n_point_features=8,
        cnn_filters=[32, 64, 128],
        mlp_units=[64, 64],
        fusion_units=[128, 64],
        dropout_rate=0.2
    )
    
    # Funahashi baseline model
    funahashi_baseline = FunahashiBaseline(
        input_dim=8,
        hidden_units=32,
        n_layers=5,
        activation='relu',
        use_residual_learning=True
    )
    
    # Simple MLP baseline
    simple_mlp = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    return {
        'MDA-CNN': mda_cnn,
        'Funahashi_Baseline': funahashi_baseline,
        'Simple_MLP': simple_mlp
    }


def create_dummy_test_data():
    """Create dummy test data for demonstration."""
    n_samples = 1000
    
    # Create dummy patch data (for MDA-CNN)
    patch_data = np.random.randn(n_samples, 9, 9, 1)
    
    # Create dummy point features
    point_features = np.random.randn(n_samples, 8)
    
    # Create dummy targets (residuals)
    targets = np.random.randn(n_samples, 1) * 0.01  # Small residuals
    
    # Create dummy metadata for regional analysis
    strikes = np.random.uniform(0.5, 2.0, n_samples)  # Strike range
    forward_prices = np.ones(n_samples)  # Normalized forward price
    residuals = np.random.randn(n_samples) * 0.02  # MC-Hagan residuals
    
    # Create TensorFlow dataset
    # For MDA-CNN: input is (patch, point_features)
    # For baselines: input is just point_features
    def create_dataset_for_model(model_name):
        if model_name == 'MDA-CNN':
            dataset = tf.data.Dataset.from_tensor_slices(
                ((patch_data, point_features), targets)
            )
        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                (point_features, targets)
            )
        return dataset.batch(32)
    
    return {
        'datasets': {name: create_dataset_for_model(name) for name in ['MDA-CNN', 'Funahashi_Baseline', 'Simple_MLP']},
        'strikes': strikes,
        'forward_prices': forward_prices,
        'residuals': residuals
    }


def run_basic_comparison_example():
    """Run a basic model comparison example."""
    print("Running Basic Model Comparison Example")
    print("="*50)
    
    # Initialize evaluation pipeline
    pipeline = EvaluationPipeline(output_dir="example_evaluation_results")
    
    # Create dummy models (in practice, these would be trained models)
    models = create_dummy_models()
    
    # Create dummy test data
    test_data_info = create_dummy_test_data()
    
    # Run comparison for each model with appropriate dataset
    comparison_results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Get appropriate test dataset
        test_dataset = test_data_info['datasets'][model_name]
        
        # Run single model evaluation
        single_result = pipeline.run_single_comparison(
            models={model_name: model},
            test_data=test_dataset,
            experiment_name=f"single_{model_name.lower()}",
            hf_budget=200,
            strikes=test_data_info['strikes'],
            forward_prices=test_data_info['forward_prices'],
            residuals=test_data_info['residuals'],
            save_results=True
        )
        
        comparison_results.update(single_result)
    
    # Generate comprehensive comparison report
    report = pipeline.generate_funahashi_comparison_report(
        comparison_results=comparison_results,
        experiment_name="basic_comparison_example",
        include_detailed_analysis=True
    )
    
    # Print summary
    pipeline.print_comparison_summary(comparison_results)
    
    return comparison_results, report


def run_budget_comparison_example():
    """Run a budget comparison example."""
    print("\n\nRunning Budget Comparison Example")
    print("="*50)
    
    # Initialize evaluation pipeline
    pipeline = EvaluationPipeline(output_dir="example_budget_results")
    
    # Create dummy models
    models = create_dummy_models()
    
    # Create dummy test data
    test_data_info = create_dummy_test_data()
    
    # Define HF budgets to test
    hf_budgets = [50, 100, 200, 500]
    
    # Note: In practice, you would need to retrain models with different HF budgets
    # For this example, we'll use the same models but simulate different budgets
    
    # Run budget comparison (using Simple_MLP dataset for all models for simplicity)
    budget_results = pipeline.run_budget_comparison(
        models={'Funahashi_Baseline': models['Funahashi_Baseline'], 
                'Simple_MLP': models['Simple_MLP']},
        test_data=test_data_info['datasets']['Simple_MLP'],
        hf_budgets=hf_budgets,
        experiment_name="budget_comparison_example",
        strikes=test_data_info['strikes'],
        forward_prices=test_data_info['forward_prices'],
        residuals=test_data_info['residuals'],
        save_results=True
    )
    
    print(f"\nBudget comparison completed for budgets: {hf_budgets}")
    
    return budget_results


def run_funahashi_format_example():
    """Run an example that specifically matches Funahashi's paper format."""
    print("\n\nRunning Funahashi Format Example")
    print("="*50)
    
    # Initialize evaluation pipeline
    pipeline = EvaluationPipeline(output_dir="funahashi_format_results")
    
    # Create models that match Funahashi's setup
    models = {
        'MDA-CNN': create_dummy_models()['MDA-CNN'],
        'Funahashi_5Layer_32Neurons': create_dummy_models()['Funahashi_Baseline']
    }
    
    # Create test data
    test_data_info = create_dummy_test_data()
    
    # Run comparison with Funahashi's exact HF budget (200 points)
    comparison_results = {}
    
    # Evaluate MDA-CNN
    mda_result = pipeline.run_single_comparison(
        models={'MDA-CNN': models['MDA-CNN']},
        test_data=test_data_info['datasets']['MDA-CNN'],
        experiment_name="funahashi_mda_cnn",
        hf_budget=200,
        strikes=test_data_info['strikes'],
        forward_prices=test_data_info['forward_prices'],
        residuals=test_data_info['residuals'],
        save_results=False
    )
    
    # Evaluate Funahashi baseline
    funahashi_result = pipeline.run_single_comparison(
        models={'Funahashi_5Layer_32Neurons': models['Funahashi_5Layer_32Neurons']},
        test_data=test_data_info['datasets']['Funahashi_Baseline'],
        experiment_name="funahashi_baseline",
        hf_budget=200,
        strikes=test_data_info['strikes'],
        forward_prices=test_data_info['forward_prices'],
        residuals=test_data_info['residuals'],
        save_results=False
    )
    
    # Combine results
    comparison_results.update(mda_result)
    comparison_results.update(funahashi_result)
    
    # Generate Funahashi-format report
    report = pipeline.generate_funahashi_comparison_report(
        comparison_results=comparison_results,
        experiment_name="funahashi_format_comparison",
        include_detailed_analysis=True
    )
    
    print("\nFunahashi Format Comparison Table:")
    print(report['summary_table'].to_string(index=False))
    
    return comparison_results, report


if __name__ == "__main__":
    print("SABR Volatility Surface Model Evaluation Examples")
    print("="*60)
    
    try:
        # Run basic comparison
        basic_results, basic_report = run_basic_comparison_example()
        
        # Run budget comparison
        budget_results = run_budget_comparison_example()
        
        # Run Funahashi format comparison
        funahashi_results, funahashi_report = run_funahashi_format_example()
        
        print("\n" + "="*60)
        print("All evaluation examples completed successfully!")
        print("Check the output directories for detailed results:")
        print("- example_evaluation_results/")
        print("- example_budget_results/")
        print("- funahashi_format_results/")
        
    except Exception as e:
        print(f"Error running evaluation examples: {e}")
        import traceback
        traceback.print_exc()