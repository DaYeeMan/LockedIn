"""
Test suite for data visualization tools.

This module tests the data visualizer functionality with sample data
to ensure all visualization methods work correctly.
"""

import numpy as np
import pytest
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generation.sabr_params import SABRParams, GridConfig
from data_generation.sabr_mc_generator import MCResult
from data_generation.hagan_surface_generator import HaganResult
from visualization.data_visualizer import DataVisualizer, VisualizationConfig


def create_sample_data():
    """Create sample data for testing visualization."""
    # Sample SABR parameters
    params = SABRParams(
        F0=100.0,
        alpha=0.3,
        beta=0.7,
        nu=0.4,
        rho=-0.2
    )
    
    # Sample grid
    n_strikes = 10
    n_maturities = 5
    strikes = np.linspace(80, 120, n_strikes)
    maturities = np.linspace(0.5, 2.0, n_maturities)
    
    # Create sample surfaces
    mc_strikes = np.tile(strikes, (n_maturities, 1))
    hagan_strikes = np.tile(strikes, (n_maturities, 1))
    
    # Generate realistic volatility surfaces
    mc_vols = np.zeros((n_maturities, n_strikes))
    hagan_vols = np.zeros((n_maturities, n_strikes))
    
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            # Simple volatility smile pattern
            moneyness = K / params.F0
            base_vol = 0.2 + 0.1 * np.sqrt(T)
            smile_effect = 0.05 * (moneyness - 1.0)**2
            
            # MC with some noise
            mc_vols[i, j] = base_vol + smile_effect + np.random.normal(0, 0.01)
            
            # Hagan slightly different
            hagan_vols[i, j] = base_vol + smile_effect * 0.9
    
    # Create result objects
    mc_result = MCResult(
        strikes=mc_strikes,
        maturities=maturities,
        volatility_surface=mc_vols,
        option_prices=np.zeros_like(mc_vols),  # Dummy option prices
        convergence_info={'converged': True},
        computation_time=1.5
    )
    
    hagan_result = HaganResult(
        strikes=hagan_strikes,
        maturities=maturities,
        volatility_surface=hagan_vols,
        computation_time=0.1,
        numerical_warnings=[],
        grid_info={'n_strikes': n_strikes, 'n_maturities': n_maturities}
    )
    
    return params, mc_result, hagan_result


class TestDataVisualizer:
    """Test class for DataVisualizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use non-interactive config for testing
        config = VisualizationConfig(
            show_plots=False,
            save_plots=False,
            interactive=True
        )
        self.visualizer = DataVisualizer(config)
        self.params, self.mc_result, self.hagan_result = create_sample_data()
    
    def test_surface_comparison_3d(self):
        """Test 3D surface comparison plot."""
        fig = self.visualizer.plot_surface_comparison_3d(
            self.mc_result, self.hagan_result, self.params
        )
        assert fig is not None
        print("✓ 3D surface comparison plot created successfully")
    
    def test_residual_heatmap(self):
        """Test residual heatmap plot."""
        fig = self.visualizer.plot_residual_heatmap(
            self.mc_result, self.hagan_result, self.params
        )
        assert fig is not None
        print("✓ Residual heatmap created successfully")
    
    def test_volatility_smiles(self):
        """Test volatility smile plots."""
        fig = self.visualizer.plot_volatility_smiles(
            self.mc_result, self.hagan_result, self.params
        )
        assert fig is not None
        print("✓ Volatility smiles plot created successfully")
    
    def test_residual_distribution(self):
        """Test residual distribution analysis."""
        fig = self.visualizer.plot_residual_distribution(
            self.mc_result, self.hagan_result, self.params
        )
        assert fig is not None
        print("✓ Residual distribution analysis created successfully")
    
    def test_parameter_space_exploration(self):
        """Test parameter space exploration with multiple parameter sets."""
        # Create multiple parameter sets
        param_sets = []
        mc_results = []
        hagan_results = []
        
        for i in range(3):
            params = SABRParams(
                F0=100.0,
                alpha=0.2 + i * 0.1,
                beta=0.6 + i * 0.1,
                nu=0.3 + i * 0.1,
                rho=-0.3 + i * 0.1
            )
            param_sets.append(params)
            
            # Create slightly different results for each parameter set
            _, mc_res, hagan_res = create_sample_data()
            mc_results.append(mc_res)
            hagan_results.append(hagan_res)
        
        fig = self.visualizer.plot_parameter_space_exploration(
            param_sets, mc_results, hagan_results
        )
        assert fig is not None
        print("✓ Parameter space exploration created successfully")
    
    def test_comprehensive_report(self):
        """Test comprehensive report generation."""
        # Create multiple parameter sets for comprehensive report
        param_sets = []
        mc_results = []
        hagan_results = []
        
        for i in range(2):
            params = SABRParams(
                F0=100.0,
                alpha=0.25 + i * 0.05,
                beta=0.65 + i * 0.05,
                nu=0.35 + i * 0.05,
                rho=-0.25 + i * 0.05
            )
            param_sets.append(params)
            
            _, mc_res, hagan_res = create_sample_data()
            mc_results.append(mc_res)
            hagan_results.append(hagan_res)
        
        figures = self.visualizer.create_comprehensive_report(
            param_sets, mc_results, hagan_results, save_individual=False
        )
        
        assert isinstance(figures, dict)
        assert len(figures) > 0
        print(f"✓ Comprehensive report created with {len(figures)} figures")
    
    def test_calculate_residuals(self):
        """Test residual calculation."""
        residuals = self.visualizer._calculate_residuals(
            self.mc_result, self.hagan_result
        )
        
        assert residuals.shape == self.mc_result.volatility_surface.shape
        assert not np.all(np.isnan(residuals))
        print("✓ Residual calculation working correctly")
    
    def test_save_summary_statistics(self):
        """Test summary statistics saving."""
        param_sets = [self.params]
        mc_results = [self.mc_result]
        hagan_results = [self.hagan_result]
        
        # Use a temporary filename for testing
        test_filename = "test_summary.csv"
        
        df = self.visualizer.save_summary_statistics(
            param_sets, mc_results, hagan_results, filename=test_filename
        )
        
        assert df is not None
        assert len(df) == 1
        assert 'alpha' in df.columns
        assert 'rmse' in df.columns
        print("✓ Summary statistics saved successfully")
        
        # Clean up test file
        import os
        test_path = f"{self.visualizer.config.output_dir}/{test_filename}"
        if os.path.exists(test_path):
            os.remove(test_path)


def run_visualization_tests():
    """Run all visualization tests."""
    print("Running data visualization tests...")
    print("=" * 50)
    
    try:
        # Create test instance
        test_viz = TestDataVisualizer()
        test_viz.setup_method()
        
        # Run individual tests
        test_viz.test_calculate_residuals()
        test_viz.test_surface_comparison_3d()
        test_viz.test_residual_heatmap()
        test_viz.test_volatility_smiles()
        test_viz.test_residual_distribution()
        test_viz.test_parameter_space_exploration()
        test_viz.test_comprehensive_report()
        test_viz.test_save_summary_statistics()
        
        print("=" * 50)
        print("✅ All visualization tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_visualization_tests()
    sys.exit(0 if success else 1)