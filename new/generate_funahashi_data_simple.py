#!/usr/bin/env python3
"""
Simple Funahashi comparison data generation using config file.

This script generates data for direct comparison with Funahashi's results
using the configuration file approach.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from generate_training_data import main as generate_main


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def main():
    """Generate Funahashi comparison data using config file."""
    print("FUNAHASHI COMPARISON DATA GENERATION (Simple)")
    print("=" * 50)
    
    setup_logging()
    
    # Override sys.argv to use Funahashi config
    original_argv = sys.argv.copy()
    
    try:
        sys.argv = [
            'generate_training_data.py',
            '--config', 'config/funahashi_comparison_config.yaml',
            '--output-dir', 'data_funahashi_comparison',
            '--seed', '12345'
        ]
        
        # Run the main data generation
        generate_main()
        
        print("\n" + "=" * 50)
        print("🎉 FUNAHASHI COMPARISON DATA READY!")
        print("=" * 50)
        print("\nGenerated Funahashi's exact test cases with 3-year maturity:")
        print("  - Case A: f=1, α=0.5, β=0.6, ν=0.3, ρ=-0.2")
        print("  - Case B: f=1, α=0.5, β=0.9, ν=0.3, ρ=-0.2")
        print("  - Case C: f=1, α=0.5, β=0.3, ν=0.3, ρ=-0.2")
        print("  - Case D: f=1, α=0.5, β=0.6, ν=0.3, ρ=-0.5")
        print("\nNext steps:")
        print("  python main_training.py --data-dir data_funahashi_comparison/processed")
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        return 1
    
    finally:
        # Restore original argv
        sys.argv = original_argv
    
    return 0


if __name__ == "__main__":
    sys.exit(main())