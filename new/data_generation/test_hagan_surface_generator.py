"""
Unit tests for Hagan analytical SABR volatility surface generator.

Tests cover numerical accuracy, edge cases, and validation against
literature benchmarks and known analytical properties.
"""

import pytest
import numpy as np
import warnings
from unittest.mock import patch

from data_generation.hagan_surface_generator import (
    HaganSurfaceGenerator, HaganConfig, HaganResult,
    compare_with_literature_benchmarks
)
from data_generation.sabr_params import SABRParams, GridConfig


class TestHaganSurfaceGenerator:
    """Test suite for HaganSurfaceGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = HaganConfig()
        self.generator = HaganSurfaceGenerator(self.config)
        
        # Standard test parameters
        self.sabr_params = SABRParams(
            F0=1.0,
            alpha=0.2,
            beta=0.5,
            nu=0.4,
            rho=-0.3
        )
        
        self.grid_config = GridConfig(
            maturity_range=(0.5, 2.0),
            n_strikes=11,
            n_maturities=3
        )
    
    def test_initialization(self):
        """Test generator initialization."""
        # Default config
        gen1 = HaganSurfaceGenerator()
        assert gen1.config is not None
        assert isinstance(gen1.config, HaganConfig)
        
        # Custom config
        custom_config = HaganConfig(use_pde_correction=False)
        gen2 = HaganSurfaceGenerator(custom_config)
        assert gen2.config.use_pde_correction is False
    
    def test_atm_volatility_basic(self):
        """Test ATM volatility calculation."""
        maturity = 1.0
        atm_vol = self.generator._hagan_atm_volatility(self.sabr_params, maturity)
        
        # ATM volatility should be positive and close to alpha
        assert atm_vol > 0
        assert abs(atm_vol - self.sabr_params.alpha) / self.sabr_params.alpha < 0.5
    
    def test_atm_volatility_no_correction(self):
        """Test ATM volatility without PDE correction."""
        config = HaganConfig(use_pde_correction=False)
        generator = HaganSurfaceGenerator(config)
        
        maturity = 1.0
        atm_vol = generator._hagan_atm_volatility(self.sabr_params, maturity)
        
        # Without correction, should equal alpha exactly
        assert abs(atm_vol - self.sabr_params.alpha) < 1e-10
    
    def test_atm_volatility_short_maturity(self):
        """Test ATM volatility for very short maturity."""
        maturity = 0.01  # 1 day
        atm_vol = self.generator._hagan_atm_volatility(self.sabr_params, maturity)
        
        # For short maturity, correction should be small
        assert abs(atm_vol - self.sabr_params.alpha) / self.sabr_params.alpha < 0.1
    
    def test_hagan_volatility_atm_case(self):
        """Test Hagan volatility for ATM strikes."""
        strikes = np.array([self.sabr_params.F0])  # Exactly ATM
        maturity = 1.0
        
        vols = self.generator.hagan_volatility(self.sabr_params, strikes, maturity)
        
        assert len(vols) == 1
        assert vols[0] > 0
        assert not np.isnan(vols[0])
    
    def test_hagan_volatility_near_atm(self):
        """Test Hagan volatility for near-ATM strikes."""
        F = self.sabr_params.F0
        strikes = np.array([F * 0.999, F, F * 1.001])  # Very close to ATM
        maturity = 1.0
        
        vols = self.generator.hagan_volatility(self.sabr_params, strikes, maturity)
        
        assert len(vols) == 3
        assert np.all(vols > 0)
        assert np.all(~np.isnan(vols))
        
        # Volatilities should be close to each other for near-ATM strikes
        vol_spread = np.max(vols) - np.min(vols)
        assert vol_spread < 0.1 * np.mean(vols)
    
    def test_hagan_volatility_smile_shape(self):
        """Test that volatility smile has reasonable shape."""
        F = self.sabr_params.F0
        strikes = np.linspace(0.7 * F, 1.3 * F, 13)
        maturity = 1.0
        
        vols = self.generator.hagan_volatility(self.sabr_params, strikes, maturity)
        
        # All volatilities should be positive and finite
        valid_vols = vols[~np.isnan(vols)]
        assert len(valid_vols) > 0
        assert np.all(valid_vols > 0)
        assert np.all(np.isfinite(valid_vols))
    
    def test_hagan_volatility_beta_one_case(self):
        """Test Hagan volatility for β = 1 (log-normal) case."""
        # Create parameters with β = 1 (use boundary value)
        lognormal_params = SABRParams(
            F0=1.0,
            alpha=0.2,
            beta=0.9,  # Close to log-normal case (boundary of valid range)
            nu=0.3,
            rho=-0.2
        )
        
        F = lognormal_params.F0
        strikes = np.array([0.8 * F, F, 1.2 * F])
        maturity = 1.0
        
        vols = self.generator.hagan_volatility(lognormal_params, strikes, maturity)
        
        assert len(vols) == 3
        assert np.all(vols > 0)
        assert np.all(~np.isnan(vols))
    
    def test_hagan_volatility_zero_nu_case(self):
        """Test Hagan volatility for ν = 0 case."""
        # Create parameters with ν ≈ 0 (use minimum valid value)
        zero_nu_params = SABRParams(
            F0=1.0,
            alpha=0.2,
            beta=0.5,
            nu=0.05,  # Minimum valid value (effectively small)
            rho=0.0
        )
        
        F = zero_nu_params.F0
        strikes = np.array([0.9 * F, F, 1.1 * F])
        maturity = 1.0
        
        vols = self.generator.hagan_volatility(zero_nu_params, strikes, maturity)
        
        # For ν ≈ 0, should reduce to displaced diffusion
        assert len(vols) == 3
        assert np.all(vols > 0)
        assert np.all(~np.isnan(vols))
    
    def test_hagan_volatility_extreme_strikes(self):
        """Test Hagan volatility for extreme strikes."""
        F = self.sabr_params.F0
        strikes = np.array([0.05 * F, 15.0 * F])  # Very extreme strikes (beyond max_moneyness=10)
        maturity = 1.0
        
        vols = self.generator.hagan_volatility(self.sabr_params, strikes, maturity)
        
        # Extreme strikes should return NaN due to numerical limits
        assert len(vols) == 2
        # At least one should be NaN due to extreme moneyness
        assert np.any(np.isnan(vols))
    
    def test_hagan_volatility_very_short_maturity(self):
        """Test Hagan volatility for very short maturity."""
        strikes = np.array([self.sabr_params.F0])
        maturity = 1e-8  # Extremely short
        
        vols = self.generator.hagan_volatility(self.sabr_params, strikes, maturity)
        
        # Should return NaN for maturity below threshold
        assert np.all(np.isnan(vols))
        assert len(self.generator.warnings) > 0
    
    def test_x_function_small_z(self):
        """Test x(z) function for small z values."""
        z = np.array([-1e-8, 0, 1e-8])
        rho = -0.3
        
        x_z = self.generator._calculate_x_function(z, rho)
        
        # For small z, x(z) ≈ z
        assert len(x_z) == 3
        assert np.all(np.isfinite(x_z))
        assert np.abs(x_z[1]) < 1e-10  # x(0) = 0
    
    def test_x_function_regular_z(self):
        """Test x(z) function for regular z values."""
        z = np.array([-0.5, 0.0, 0.5])
        rho = -0.3
        
        x_z = self.generator._calculate_x_function(z, rho)
        
        assert len(x_z) == 3
        assert np.all(np.isfinite(x_z))
        assert x_z[1] == 0  # x(0) = 0
    
    def test_x_function_extreme_rho(self):
        """Test x(z) function for extreme correlation values."""
        z = np.array([0.1, 0.2])
        
        # Test with ρ close to ±1
        for rho in [-0.99, 0.99]:
            x_z = self.generator._calculate_x_function(z, rho)
            assert np.all(np.isfinite(x_z))
    
    def test_pde_correction_calculation(self):
        """Test PDE correction term calculation."""
        FK = np.array([0.8, 1.0, 1.2])
        maturity = 1.0
        
        correction = self.generator._calculate_pde_correction(
            self.sabr_params, FK, maturity
        )
        
        assert len(correction) == 3
        assert np.all(correction > 0)  # Correction should be positive
        assert np.all(np.isfinite(correction))
    
    def test_generate_volatility_surface_basic(self):
        """Test basic volatility surface generation."""
        result = self.generator.generate_volatility_surface(
            self.sabr_params, self.grid_config
        )
        
        assert isinstance(result, HaganResult)
        assert result.strikes.shape[0] == self.grid_config.n_maturities
        assert result.volatility_surface.shape == result.strikes.shape
        assert len(result.maturities) == self.grid_config.n_maturities
        assert result.computation_time > 0
    
    def test_generate_volatility_surface_extended_wings(self):
        """Test volatility surface generation with extended wings."""
        extended_config = GridConfig(
            maturity_range=(1.0, 2.0),
            n_strikes=11,
            n_maturities=2,
            include_extended_wings=True
        )
        
        result = self.generator.generate_volatility_surface(
            self.sabr_params, extended_config
        )
        
        assert isinstance(result, HaganResult)
        # Should have more strikes due to extended wings
        assert result.strikes.shape[1] >= extended_config.n_strikes
    
    def test_validate_surface_basic(self):
        """Test basic surface validation."""
        result = self.generator.generate_volatility_surface(
            self.sabr_params, self.grid_config
        )
        
        validation = self.generator.validate_surface(result, self.sabr_params)
        
        assert isinstance(validation, dict)
        assert 'passed' in validation
        assert 'checks' in validation
        assert 'warnings' in validation
        assert isinstance(validation['checks'], list)
    
    def test_validate_surface_with_issues(self):
        """Test surface validation with problematic data."""
        # Create a result with some NaN and negative values
        result = self.generator.generate_volatility_surface(
            self.sabr_params, self.grid_config
        )
        
        # Introduce some issues
        result.volatility_surface[0, 0] = -0.1  # Negative volatility
        result.volatility_surface[0, 1] = np.inf  # Infinite volatility
        
        validation = self.generator.validate_surface(result, self.sabr_params)
        
        assert validation['passed'] is False
        assert len(validation['warnings']) > 0
    
    def test_numerical_stability_extreme_parameters(self):
        """Test numerical stability with extreme but valid parameters."""
        extreme_params = SABRParams(
            F0=1.0,
            alpha=0.05,  # Very low alpha
            beta=0.9,    # High beta
            nu=0.9,      # High nu
            rho=0.75     # High positive correlation
        )
        
        result = self.generator.generate_volatility_surface(
            extreme_params, self.grid_config
        )
        
        # Should complete without crashing
        assert isinstance(result, HaganResult)
        
        # Check that we get some valid volatilities
        valid_vols = result.volatility_surface[~np.isnan(result.volatility_surface)]
        assert len(valid_vols) > 0
    
    def test_consistency_across_maturities(self):
        """Test that volatility surface is consistent across maturities."""
        result = self.generator.generate_volatility_surface(
            self.sabr_params, self.grid_config
        )
        
        # ATM volatilities should generally increase with maturity (for positive ν)
        atm_vols = []
        for i in range(len(result.maturities)):
            strikes = result.strikes[i]
            vols = result.volatility_surface[i]
            
            valid_mask = ~np.isnan(strikes) & ~np.isnan(vols)
            if np.any(valid_mask):
                valid_strikes = strikes[valid_mask]
                valid_vols = vols[valid_mask]
                
                # Find closest to ATM
                atm_idx = np.argmin(np.abs(valid_strikes - self.sabr_params.F0))
                atm_vols.append(valid_vols[atm_idx])
        
        if len(atm_vols) >= 2:
            # For positive ν, ATM vol should generally increase with time
            if self.sabr_params.nu > 0:
                # Allow for some numerical noise
                assert atm_vols[-1] >= atm_vols[0] * 0.9


class TestHaganConfigEdgeCases:
    """Test edge cases for HaganConfig."""
    
    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Valid config
        config = HaganConfig(
            use_pde_correction=True,
            numerical_tolerance=1e-10,
            max_moneyness=5.0,
            min_maturity=1e-5
        )
        
        assert config.use_pde_correction is True
        assert config.numerical_tolerance == 1e-10
    
    def test_extreme_config_values(self):
        """Test configuration with extreme values."""
        config = HaganConfig(
            numerical_tolerance=1e-15,  # Very tight tolerance
            max_moneyness=100.0,        # Very wide moneyness range
            min_maturity=1e-10          # Very short minimum maturity
        )
        
        generator = HaganSurfaceGenerator(config)
        
        # Should still work with extreme config
        sabr_params = SABRParams(F0=1.0, alpha=0.2, beta=0.5, nu=0.3, rho=0.0)
        strikes = np.array([1.0])
        maturity = 1.0
        
        vols = generator.hagan_volatility(sabr_params, strikes, maturity)
        assert len(vols) == 1


class TestLiteratureBenchmarks:
    """Test against literature benchmarks."""
    
    def test_compare_with_literature_benchmarks_basic(self):
        """Test basic benchmark comparison functionality."""
        # Use parameters that might match a benchmark
        sabr_params = SABRParams(F0=1.0, alpha=0.2, beta=0.5, nu=0.4, rho=-0.3)
        
        # Create a simple result
        generator = HaganSurfaceGenerator()
        grid_config = GridConfig(maturity_range=(0.9, 1.1), n_maturities=1, n_strikes=5)
        result = generator.generate_volatility_surface(sabr_params, grid_config)
        
        benchmark_results = compare_with_literature_benchmarks(sabr_params, result)
        
        assert isinstance(benchmark_results, dict)
        assert 'comparisons' in benchmark_results
        assert 'overall_accuracy' in benchmark_results
    
    def test_benchmark_no_matches(self):
        """Test benchmark comparison when no parameters match."""
        # Use parameters that won't match any benchmark
        sabr_params = SABRParams(F0=1.0, alpha=0.15, beta=0.6, nu=0.35, rho=0.2)
        
        generator = HaganSurfaceGenerator()
        grid_config = GridConfig(maturity_range=(1.0, 2.0), n_maturities=2, n_strikes=5)
        result = generator.generate_volatility_surface(sabr_params, grid_config)
        
        benchmark_results = compare_with_literature_benchmarks(sabr_params, result)
        
        assert len(benchmark_results['comparisons']) == 0
        assert benchmark_results['overall_accuracy'] == 'unknown'


class TestHaganResultDataStructure:
    """Test HaganResult data structure."""
    
    def test_hagan_result_creation(self):
        """Test HaganResult creation and attributes."""
        strikes = np.array([[0.8, 1.0, 1.2]])
        maturities = np.array([1.0])
        volatility_surface = np.array([[0.18, 0.20, 0.22]])
        
        result = HaganResult(
            strikes=strikes,
            maturities=maturities,
            volatility_surface=volatility_surface,
            computation_time=0.1,
            numerical_warnings=[],
            grid_info={'test': 'info'}
        )
        
        assert np.array_equal(result.strikes, strikes)
        assert np.array_equal(result.maturities, maturities)
        assert np.array_equal(result.volatility_surface, volatility_surface)
        assert result.computation_time == 0.1
        assert result.numerical_warnings == []
        assert result.grid_info == {'test': 'info'}


class TestNumericalEdgeCases:
    """Test numerical edge cases and stability."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = HaganSurfaceGenerator()
    
    def test_zero_forward_price(self):
        """Test handling of zero forward price."""
        with pytest.raises(ValueError):
            SABRParams(F0=0.0, alpha=0.2, beta=0.5, nu=0.3, rho=0.0)
    
    def test_negative_forward_price(self):
        """Test handling of negative forward price."""
        with pytest.raises(ValueError):
            SABRParams(F0=-1.0, alpha=0.2, beta=0.5, nu=0.3, rho=0.0)
    
    def test_zero_strikes(self):
        """Test handling of zero strikes."""
        sabr_params = SABRParams(F0=1.0, alpha=0.2, beta=0.5, nu=0.3, rho=0.0)
        strikes = np.array([0.0, 1.0])
        maturity = 1.0
        
        vols = self.generator.hagan_volatility(sabr_params, strikes, maturity)
        
        # Zero strike should be handled gracefully
        assert len(vols) == 2
        # First volatility might be NaN or a large value
        assert np.isfinite(vols[1])  # Second should be finite
    
    def test_negative_strikes(self):
        """Test handling of negative strikes."""
        sabr_params = SABRParams(F0=1.0, alpha=0.2, beta=0.5, nu=0.3, rho=0.0)
        strikes = np.array([-0.5, 1.0])
        maturity = 1.0
        
        vols = self.generator.hagan_volatility(sabr_params, strikes, maturity)
        
        # Negative strike should be handled gracefully
        assert len(vols) == 2
        assert np.isfinite(vols[1])  # Positive strike should work
    
    def test_very_large_nu(self):
        """Test behavior with very large ν (at boundary)."""
        sabr_params = SABRParams(F0=1.0, alpha=0.2, beta=0.5, nu=0.9, rho=0.0)
        strikes = np.array([0.9, 1.0, 1.1])
        maturity = 1.0
        
        vols = self.generator.hagan_volatility(sabr_params, strikes, maturity)
        
        # Should handle large ν without crashing
        assert len(vols) == 3
        valid_vols = vols[~np.isnan(vols)]
        if len(valid_vols) > 0:
            assert np.all(valid_vols > 0)
    
    def test_extreme_correlation(self):
        """Test behavior with extreme correlation values."""
        for rho in [-0.75, 0.75]:  # Boundary values
            sabr_params = SABRParams(F0=1.0, alpha=0.2, beta=0.5, nu=0.3, rho=rho)
            strikes = np.array([0.9, 1.0, 1.1])
            maturity = 1.0
            
            vols = self.generator.hagan_volatility(sabr_params, strikes, maturity)
            
            # Should handle extreme correlation
            assert len(vols) == 3
            valid_vols = vols[~np.isnan(vols)]
            if len(valid_vols) > 0:
                assert np.all(valid_vols > 0)


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v'])