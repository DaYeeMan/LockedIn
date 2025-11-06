"""
Evaluation module for SABR volatility surface models.

This module provides comprehensive evaluation metrics and model comparison
functionality for comparing MDA-CNN against Funahashi baseline and other models.
"""

from metrics import (
    FunahashiMetrics,
    ModelEvaluator, 
    EvaluationResults,
    WingRegionAnalyzer
)

from model_comparison import (
    ModelComparator,
    ModelComparisonConfig,
    ModelResults,
    FunahashiComparisonTable,
    HFBudgetComparison
)

from evaluation_pipeline import EvaluationPipeline

__all__ = [
    # Metrics
    'FunahashiMetrics',
    'ModelEvaluator',
    'EvaluationResults', 
    'WingRegionAnalyzer',
    
    # Model Comparison
    'ModelComparator',
    'ModelComparisonConfig',
    'ModelResults',
    'FunahashiComparisonTable',
    'HFBudgetComparison',
    
    # Pipeline
    'EvaluationPipeline'
]