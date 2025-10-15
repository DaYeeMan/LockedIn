"""
Tests for SABR Monte Carlo simulation engine.

This module provides comprehensive tests for the Monte Carlo implementation,
including accuracy validation, convergence checks, and performance tests.
"""

import pytest
import numpy as np
import time
from typing import List

from data_generation.sabr_params import SABRParams, GridConfig
from data_generation.sabr_mc_generator import (
    SABRMCGenerator, ParallelMCGenerator, MCConfig, MCResult,
    validate_mc_accuracy
)


class TestMCConfig:
    """Test Monte Carlo configuration."""
    
    def test_mc_config_creation(self):
        """Test MCConfig creation with default values."""
        config = MCConfig()
        
        assert config.n_paths == 100000
        assert config.n_steps == 300
        assert config.random_seed is None
        assert config.use_antithetic is True
        assert config.convergence_check is True
        assert config.convergence_tolerance == 1e-4
        assert config.max_iterations == 5
    
    def test_mc_config_custom_values(self):
        """Test MCConfig creation with custom values."""
        config = MCConfig(
            n_paths=50000,
            n_steps=200,
            random_seed=42,
            use_antithetic=False,
            convergence_check=False
        )
        
        assert config.n_paths == 50000
        assert config.n_steps == 200
        assert config.random_seed == 42
        assert config.use_antithetic is False
        assert config.convergence_check is False


class TestSABRMCGenerator:
    """Test SABR Monte Carlo generator."""
    
    @pytest.fixture
    def sample_sabr_params(self):
        """Sample SABR parameters for testing."""
        return SABRParams(F0=1.0, alpha=0.3, beta=0.5, nu=0.4, rho=-0.2)
    
    @pytest.fixture
    def sample_grid_config(self):
        """Sample grid configuration for testing."""
        return GridConfig(
            maturity_range=(1.0, 3.0),
            n_strikes=11,
            n_maturities=3,
            include_extended_wings=False
        )
    
    @pytest.fixture
    def fast_mc_config(self):
        """Fast MC configuration for testing."""
        return MCConfig(
            n_paths=1000,
            n_steps=50,
            random_seed=42,
            use_antithetic=True
        )
    
    def test_generator_initialization(self, fast_mc_config):
        """Test generator initialization."""
        generator = SABRMCGenerator(fast_mc_config)
        
        assert generator.config == fast_mc_config
        assert generator.rng is not None
    
    def test_simulate_paths_shape(self, sample_sabr_params, fast_mc_config):
        """Test that simulated paths have correct shape."""
        generator = SABRMCGenerator(fast_mc_config)
        maturity = 1.0
        
        F_paths, alpha_paths = generator.simulate_paths(sample_sabr_params, maturity)
        
        expected_shape = (fast_mc_config.n_paths, fast_mc_config.n_steps + 1)
        assert F_paths.shape == expected_shape
        assert alpha_paths.shape == expected_shape
    
    def test_simulate_paths_initial_conditions(self, sample_sabr_params, fast_mc_config):
        """Test that paths start with correct initial conditions."""
        generator = SABRMCGenerator(fast_mc_config)
        maturity = 1.0
        
        F_paths, alpha_paths = generator.simulate_paths(sample_sabr_params, maturity)
        
        # Check initial conditions
        np.testing.assert_allclose(F_paths[:, 0], sample_sabr_params.F0)
        np.testing.assert_allclose(alpha_paths[:, 0], sample_sabr_params.alpha)
    
    def test_simulate_paths_positivity(self, sample_sabr_params, fast_mc_config):
        """Test that simulated paths remain positive."""
        generator = SABRMCGenerator(fast_mc_config)
        maturity = 1.0
        
        F_paths, alpha_paths = generator.simulate_paths(sample_sabr_params, maturity)
        
        # Forward prices should be positive
        assert np.all(F_paths > 0), "Forward prices should remain positive"
        
        # Volatilities should be positive
        assert np.all(alpha_paths > 0), "Volatilities should remain positive"
    
    def test_simulate_paths_antithetic_variates(self, sample_sabr_params):
        """Test antithetic variates functionality."""
        # Test with antithetic variates
        config_with_antithetic = MCConfig(n_paths=1000, n_steps=50, random_seed=42, use_antithetic=True)
        generator_with = SABRMCGenerator(config_with_antithetic)
        
        # Test without antithetic variates
        config_without_antithetic = MCConfig(n_paths=1000, n_steps=50, random_seed=42, use_antithetic=False)
        generator_without = SABRMCGenerator(config_without_antithetic)
        
        maturity = 1.0
        
        F_paths_with, _ = generator_with.simulate_paths(sample_sabr_params, maturity)
        F_paths_without, _ = generator_without.simulate_paths(sample_sabr_params, maturity)
        
        # Both should have same number of paths
        assert F_paths_with.shape[0] == F_paths_without.shape[0]
        
        # Variance should be different (antithetic should reduce variance)
        var_with = np.var(F_paths_with[:, -1])
        var_without = np.var(F_paths_without[:, -1])
        
        # This is a statistical test, so we use a loose check
        assert var_with != var_without, "Antithetic variates should affect variance"
    
    def test_calculate_option_price(self, sample_sabr_params, fast_mc_config):
        """Test option price calculation."""
        generator = SABRMCGenerator(fast_mc_config)
        maturity = 1.0
        
        F_paths, _ = generator.simulate_paths(sample_sabr_params, maturity)
        
        # Test ATM call option
        strike = sample_sabr_params.F0
        call_price = generator.calculate_option_price(F_paths, strike, is_call=True)
        put_price = generator.calculate_option_price(F_paths, strike, is_call=False)
        
        # Prices should be positive
        assert call_price >= 0, "Call price should be non-negative"
        assert put_price >= 0, "Put price should be non-negative"
        
        # ATM call and put should have similar prices (put-call parity, zero rates)
        assert abs(call_price - put_price) < 0.1, "ATM call and put prices should be similar"
    
    def test_implied_volatility_calculation(self, sample_sabr_params):
        """Test implied volatility calculation."""
        generator = SABRMCGenerator(MCConfig(n_paths=100, n_steps=10, random_seed=42))
        
        # Test with known Black-Scholes price
        forward = 1.0
        strike = 1.0
        maturity = 1.0
        true_vol = 0.2
        
        # Calculate Black-Scholes price
        from scipy.stats import norm
        d1 = (np.log(forward / strike) + 0.5 * true_vol**2 * maturity) / (true_vol * np.sqrt(maturity))
        d2 = d1 - true_vol * np.sqrt(maturity)
        bs_price = forward * norm.cdf(d1) - strike * norm.cdf(d2)
        
        # Calculate implied volatility
        impl_vol = generator.implied_volatility_from_price(bs_price, forward, strike, maturity)
        
        # Should recover the true volatility
        np.testing.assert_allclose(impl_vol, true_vol, rtol=1e-6)
    
    def test_generate_volatility_surface(self, sample_sabr_params, sample_grid_config, fast_mc_config):
        """Test complete volatility surface generation."""
        generator = SABRMCGenerator(fast_mc_config)
        
        result = generator.generate_volatility_surface(sample_sabr_params, sample_grid_config)
        
        # Check result structure
        assert isinstance(result, MCResult)
        assert result.strikes.shape[0] == sample_grid_config.n_maturities
        assert result.volatility_surface.shape[0] == sample_grid_config.n_maturities
        assert result.option_prices.shape[0] == sample_grid_config.n_maturities
        assert len(result.maturities) == sample_grid_config.n_maturities
        
        # Check that we have some valid volatilities
        valid_vols = result.volatility_surface[~np.isnan(result.volatility_surface)]
        assert len(valid_vols) > 0, "Should have some valid volatilities"
        
        # Check convergence info
        assert 'overall' in result.convergence_info
        assert result.computation_time > 0
    
    def test_beta_edge_cases(self, fast_mc_config):
        """Test simulation with edge case beta values."""
        # Test beta = 1.0 (log-normal case)
        params_lognormal = SABRParams(F0=1.0, alpha=0.3, beta=1.0, nu=0.4, rho=-0.2)
        generator = SABRMCGenerator(fast_mc_config)
        
        F_paths, alpha_paths = generator.simulate_paths(params_lognormal, 1.0)
        
        # Should still be positive and finite
        assert np.all(F_paths > 0)
        assert np.all(alpha_paths > 0)
        assert np.all(np.isfinite(F_paths))
        assert np.all(np.isfinite(alpha_paths))
        
        # Test beta close to 0
        params_normal = SABRParams(F0=1.0, alpha=0.3, beta=0.3, nu=0.4, rho=-0.2)
        F_paths, alpha_paths = generator.simulate_paths(params_normal, 1.0)
        
        assert np.all(F_paths > 0)
        assert np.all(alpha_paths > 0)


class TestParallelMCGenerator:
    """Test parallel Monte Carlo generator."""
    
    @pytest.fixture
    def sample_param_sets(self):
        """Sample parameter sets for parallel testing."""
        return [
            SABRParams(F0=1.0, alpha=0.2, beta=0.5, nu=0.3, rho=-0.1),
            SABRParams(F0=1.0, alpha=0.4, beta=0.7, nu=0.5, rho=0.2),
            SABRParams(F0=1.0, alpha=0.3, beta=0.6, nu=0.4, rho=-0.3)
        ]
    
    @pytest.fixture
    def fast_grid_config(self):
        """Fast grid configuration for testing."""
        return GridConfig(
            maturity_range=(1.0, 2.0),
            n_strikes=5,
            n_maturities=2
        )
    
    @pytest.fixture
    def fast_mc_config(self):
        """Fast MC configuration for testing."""
        return MCConfig(n_paths=100, n_steps=20, random_seed=42)
    
    def test_parallel_generator_initialization(self, fast_mc_config):
        """Test parallel generator initialization."""
        generator = ParallelMCGenerator(fast_mc_config, n_workers=2)
        
        assert generator.mc_config == fast_mc_config
        assert generator.n_workers == 2
    
    def test_generate_surfaces_parallel(self, sample_param_sets, fast_grid_config, fast_mc_config):
        """Test parallel surface generation."""
        generator = ParallelMCGenerator(fast_mc_config, n_workers=2)
        
        results = generator.generate_surfaces_parallel(sample_param_sets, fast_grid_config)
        
        # Should have one result per parameter set
        assert len(results) == len(sample_param_sets)
        
        # Each result should be valid
        for result in results:
            assert isinstance(result, MCResult)
            assert result.volatility_surface.shape[0] == fast_grid_config.n_maturities


class TestMCAccuracyValidation:
    """Test Monte Carlo accuracy validation."""
    
    @pytest.fixture
    def sample_sabr_params(self):
        """Sample SABR parameters for validation testing."""
        return SABRParams(F0=1.0, alpha=0.3, beta=0.5, nu=0.2, rho=0.0)
    
    @pytest.fixture
    def sample_mc_result(self, sample_sabr_params):
        """Sample MC result for validation testing."""
        # Create a simple mock result
        strikes = np.array([[0.8, 0.9, 1.0, 1.1, 1.2]])
        maturities = np.array([1.0])
        
        # Create realistic volatility surface (smile shape)
        volatilities = np.array([[0.35, 0.32, 0.30, 0.32, 0.35]])
        option_prices = np.array([[0.2, 0.15, 0.12, 0.15, 0.2]])
        
        convergence_info = {
            'overall': {
                'total_valid_volatilities': 5,
                'convergence_achieved': True
            }
        }
        
        return MCResult(
            strikes=strikes,
            maturities=maturities,
            volatility_surface=volatilities,
            option_prices=option_prices,
            convergence_info=convergence_info,
            computation_time=1.0
        )
    
    def test_validate_mc_accuracy_pass(self, sample_sabr_params, sample_mc_result):
        """Test validation with good MC result."""
        validation = validate_mc_accuracy(sample_sabr_params, sample_mc_result, tolerance=0.1)
        
        assert validation['passed'] is True
        assert len(validation['checks']) > 0
        assert len(validation['warnings']) == 0
    
    def test_validate_mc_accuracy_fail_atm(self, sample_sabr_params):
        """Test validation failure due to bad ATM volatility."""
        # Create result with bad ATM volatility
        strikes = np.array([[0.8, 0.9, 1.0, 1.1, 1.2]])
        maturities = np.array([0.5])  # Short maturity
        
        # ATM volatility very different from alpha
        volatilities = np.array([[0.35, 0.32, 0.8, 0.32, 0.35]])  # Bad ATM vol
        option_prices = np.array([[0.2, 0.15, 0.12, 0.15, 0.2]])
        
        convergence_info = {'overall': {'total_valid_volatilities': 5}}
        
        bad_result = MCResult(
            strikes=strikes,
            maturities=maturities,
            volatility_surface=volatilities,
            option_prices=option_prices,
            convergence_info=convergence_info,
            computation_time=1.0
        )
        
        validation = validate_mc_accuracy(sample_sabr_params, bad_result, tolerance=0.1)
        
        assert validation['passed'] is False
        assert len(validation['warnings']) > 0
    
    def test_validate_mc_accuracy_negative_vols(self, sample_sabr_params):
        """Test validation failure due to negative volatilities."""
        strikes = np.array([[0.8, 0.9, 1.0, 1.1, 1.2]])
        maturities = np.array([1.0])
        
        # Include negative volatility
        volatilities = np.array([[0.35, 0.32, -0.1, 0.32, 0.35]])
        option_prices = np.array([[0.2, 0.15, 0.12, 0.15, 0.2]])
        
        convergence_info = {'overall': {'total_valid_volatilities': 4}}
        
        bad_result = MCResult(
            strikes=strikes,
            maturities=maturities,
            volatility_surface=volatilities,
            option_prices=option_prices,
            convergence_info=convergence_info,
            computation_time=1.0
        )
        
        validation = validate_mc_accuracy(sample_sabr_params, bad_result)
        
        assert validation['passed'] is False
        assert any('negative' in warning.lower() for warning in validation['warnings'])


class TestMCPerformance:
    """Test Monte Carlo performance characteristics."""
    
    def test_convergence_with_path_count(self):
        """Test that results converge as path count increases."""
        sabr_params = SABRParams(F0=1.0, alpha=0.3, beta=0.5, nu=0.2, rho=0.0)
        grid_config = GridConfig(maturity_range=(1.0, 1.0), n_strikes=5, n_maturities=1)
        
        path_counts = [1000, 5000, 10000]
        atm_vols = []
        
        for n_paths in path_counts:
            config = MCConfig(n_paths=n_paths, n_steps=100, random_seed=42)
            generator = SABRMCGenerator(config)
            
            result = generator.generate_volatility_surface(sabr_params, grid_config)
            
            # Find ATM volatility
            strikes = result.strikes[0]
            valid_strikes = strikes[~np.isnan(strikes)]
            atm_idx = np.argmin(np.abs(valid_strikes - sabr_params.F0))
            atm_vol = result.volatility_surface[0, atm_idx]
            
            if not np.isnan(atm_vol):
                atm_vols.append(atm_vol)
        
        # Should have results for all path counts
        assert len(atm_vols) == len(path_counts)
        
        # Variance should decrease with more paths (statistical test)
        # We just check that we get reasonable values
        for vol in atm_vols:
            assert 0.1 < vol < 0.6, f"ATM volatility {vol} should be reasonable"
    
    def test_computation_time_scaling(self):
        """Test that computation time scales reasonably with problem size."""
        sabr_params = SABRParams(F0=1.0, alpha=0.3, beta=0.5, nu=0.2, rho=0.0)
        
        # Small problem
        small_config = MCConfig(n_paths=100, n_steps=20, random_seed=42)
        small_grid = GridConfig(maturity_range=(1.0, 1.0), n_strikes=3, n_maturities=1)
        
        generator = SABRMCGenerator(small_config)
        
        start_time = time.time()
        result_small = generator.generate_volatility_surface(sabr_params, small_grid)
        small_time = time.time() - start_time
        
        # Larger problem
        large_config = MCConfig(n_paths=500, n_steps=50, random_seed=42)
        large_grid = GridConfig(maturity_range=(1.0, 2.0), n_strikes=5, n_maturities=2)
        
        generator = SABRMCGenerator(large_config)
        
        start_time = time.time()
        result_large = generator.generate_volatility_surface(sabr_params, large_grid)
        large_time = time.time() - start_time
        
        # Larger problem should take more time
        assert large_time > small_time, "Larger problem should take more time"
        
        # But not excessively more (should be roughly linear in paths * steps * strikes * maturities)
        size_ratio = (500 * 50 * 5 * 2) / (100 * 20 * 3 * 1)
        time_ratio = large_time / small_time
        
        # Allow for some overhead, but should be roughly proportional
        assert time_ratio < size_ratio * 3, "Time scaling should be reasonable"


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])