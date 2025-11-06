# Evaluation Module

This module provides comprehensive evaluation metrics and model comparison functionality for SABR volatility surface models, specifically designed to compare MDA-CNN against Funahashi baseline using the same HF budget constraints.

## Overview

The evaluation module implements:

1. **Funahashi's Exact Metrics**: MSE, RMSE, MAE, and relative percentage error as used in the original paper
2. **Model Comparison Pipeline**: Direct comparison between MDA-CNN and Funahashi baseline
3. **Regional Analysis**: Performance evaluation by moneyness regions (ITM, ATM, OTM)
4. **Wing Region Analysis**: Specialized analysis for volatility wing regions where MC-Hagan residuals are largest
5. **HF Budget Comparison**: Compare models using different high-fidelity data budget constraints
6. **Funahashi Format Tables**: Generate comparison tables matching the original paper's format

## Key Components

### 1. Metrics (`metrics.py`)

- `FunahashiMetrics`: Implementation of exact metrics from Funahashi's paper
- `ModelEvaluator`: Comprehensive model evaluation with regional analysis
- `WingRegionAnalyzer`: Specialized analysis for volatility wing regions
- `EvaluationResults`: Data structure for storing evaluation results

### 2. Model Comparison (`model_comparison.py`)

- `ModelComparator`: Main class for comparing multiple models
- `FunahashiComparisonTable`: Generate tables matching Funahashi's paper format
- `HFBudgetComparison`: Compare models across different HF budget constraints
- `ModelResults`: Data structure for storing individual model results

### 3. Evaluation Pipeline (`evaluation_pipeline.py`)

- `EvaluationPipeline`: High-level interface for running comprehensive evaluations
- Automated report generation
- File management and result saving
- Summary statistics and visualization

## Usage Examples

### Basic Model Comparison

```python
from evaluation import EvaluationPipeline, ModelComparator

# Initialize pipeline
pipeline = EvaluationPipeline(output_dir="evaluation_results")

# Compare models
models = {
    'MDA-CNN': trained_mda_cnn_model,
    'Funahashi_Baseline': trained_funahashi_model
}

results = pipeline.run_single_comparison(
    models=models,
    test_data=test_dataset,
    experiment_name="mda_cnn_vs_funahashi",
    hf_budget=200,
    strikes=strike_prices,
    forward_prices=forward_prices,
    residuals=mc_hagan_residuals
)

# Generate Funahashi format report
report = pipeline.generate_funahashi_comparison_report(
    comparison_results=results,
    experiment_name="funahashi_comparison"
)
```

### HF Budget Analysis

```python
# Compare across different HF budgets
budget_results = pipeline.run_budget_comparison(
    models=models,
    test_data=test_dataset,
    hf_budgets=[50, 100, 200, 500],
    experiment_name="budget_analysis"
)
```

### Custom Metrics Evaluation

```python
from evaluation import ModelEvaluator, FunahashiMetrics

evaluator = ModelEvaluator()

# Evaluate single model
results = evaluator.evaluate_model(y_true, y_pred)

# Regional evaluation
regional_results = evaluator.evaluate_by_region(
    y_true, y_pred, strikes, forward_price
)

# Wing region analysis
from evaluation import WingRegionAnalyzer
wing_analyzer = WingRegionAnalyzer()
wing_results = wing_analyzer.evaluate_wing_performance(
    y_true, y_pred, residuals, strikes, forward_price
)
```

## Metrics Implemented

### Funahashi's Exact Metrics

1. **Mean Squared Error (MSE)**: `mean((y_true - y_pred)^2)`
2. **Root Mean Squared Error (RMSE)**: `sqrt(MSE)`
3. **Mean Absolute Error (MAE)**: `mean(|y_true - y_pred|)`
4. **Relative Percentage Error**: `mean(|y_true - y_pred| / |y_true|) * 100`

### Additional Metrics

- Maximum Absolute Error
- Mean and Standard Deviation of Relative Errors
- Regional performance breakdown (ITM/ATM/OTM)
- Wing region performance analysis

## Output Files

The evaluation pipeline generates several output files:

1. **Funahashi Format Table** (`*_funahashi_format.csv`): Main comparison table matching paper format
2. **Comprehensive Table** (`*_comprehensive.csv`): Detailed results with regional breakdowns
3. **Raw Results** (`*_raw_results.json`): Complete results in JSON format
4. **Summary Statistics** (`*_summary_stats.json`): Statistical summary of comparison
5. **Complete Report** (`*_complete_report.json`): Full report with all analysis

## Regional Analysis

The evaluation system supports analysis by moneyness regions:

- **ITM (In-The-Money)**: Strikes < (Forward - threshold)
- **ATM (At-The-Money)**: Strikes within threshold of Forward
- **OTM (Out-Of-The-Money)**: Strikes > (Forward + threshold)

Default ATM threshold is 5% of forward price.

## Wing Region Analysis

Wing regions are identified as areas where MC-Hagan residuals are largest (default: top 10% by absolute residual magnitude). This analysis helps evaluate model performance in the most challenging regions of the volatility surface.

## Testing

Run the test suite to verify functionality:

```bash
cd new/evaluation
python -m pytest test_evaluation.py -v
```

## Requirements

- numpy
- pandas
- tensorflow
- pathlib
- json
- pytest (for testing)

## Integration with Training Pipeline

The evaluation module integrates seamlessly with the training pipeline:

1. Train models using the same HF budget
2. Evaluate on identical test sets
3. Generate comparison reports
4. Analyze performance across different budget constraints

This ensures fair comparison between MDA-CNN and Funahashi baseline models as required by the research objectives.