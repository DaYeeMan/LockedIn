"""Simple test to verify evaluation module works."""

import sys
from pathlib import Path
import numpy as np

# Add evaluation to path
sys.path.append(str(Path(__file__).parent / "evaluation"))

try:
    from metrics import FunahashiMetrics
    print("✓ Successfully imported FunahashiMetrics")
    
    # Test basic functionality
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.1])
    
    mse = FunahashiMetrics.mean_squared_error(y_true, y_pred)
    print(f"✓ MSE calculation works: {mse}")
    
    from metrics import ModelEvaluator
    print("✓ Successfully imported ModelEvaluator")
    
    from model_comparison import ModelComparator
    print("✓ Successfully imported ModelComparator")
    
    from evaluation_pipeline import EvaluationPipeline
    print("✓ Successfully imported EvaluationPipeline")
    
    print("\n✅ All evaluation module imports successful!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()