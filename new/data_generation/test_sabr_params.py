"""
Unit tests for SABR parameter and grid configuration classes.

Tests parameter validation, sampling strategies, and grid generation
following Funahashi's approach.
"""

import pytest
import numpy as np
from typing import List

import sys
import os
sys.path.append(os.path.dirname(__file__))

from sabr_params import SABRParams, GridConfig, ParameterSampler


class TestSABRParams:
    """Test cases for SABRParams class."""
    
    def test_valid_parameters(self):
        """Test creation with valid parameters."""
        params = SABRParams(F0=1.0, alpha=0.3, beta=0.5, nu=0.4, rho=0.2)
        assert params.F0 == 1.0
        assert params.alpha == 0.3
        assert params.beta == 0.5
        assert params.nu == 0.4
        assert params.rho == 0.2
        assert params.validate()
    
    def test_invalid_forward_price(self):
        """Test validation fails for non-positive forward price."""
        with pytest.raises(ValueError, match="Forward price F0 must be positive"):
            SABRParams(F0=0.0, alpha=0.3, beta=0.5, nu=0.4, rho=0.2)
        
        with pytest.raises(ValueError, match="Forward price F0 must be positive"):
            SABRParams(F0=-1.0, alpha=0.3, beta=0.5, nu=0.4, rho=0.2)
    
    def test_alpha_range_validation(self):
        """Test alpha parameter range validation."""
        # Valid alpha values
        SABRParams(F0=1.0, alpha=0.05, beta=0.5, nu=0.4, rho=0.2)  # Min
        SABRParams(F0=1.0, alpha=0.6, beta=0.5, nu=0.4, rho=0.2)   # Max
        SABRParams(F0=1.0, alpha=0.3, beta=0.5, nu=0.4, rho=0.2)   # Mid
        
        # Invalid alpha values
        with pytest.raises(ValueError, match="Alpha must be in range"):
            SABRParams(F0=1.0, alpha=0.04, beta=0.5, nu=0.4, rho=0.2)  # Too low
        
        with pytest.raises(ValueError, match="Alpha must be in range"):
            SABRParams(F0=1.0, alpha=0.61, beta=0.5, nu=0.4, rho=0.2)  # Too high
    
    def test_beta_range_validation(self):
        """Test beta parameter range validation."""
        # Valid beta values
        SABRParams(F0=1.0, alpha=0.3, beta=0.3, nu=0.4, rho=0.2)  # Min
        SABRParams(F0=1.0, alpha=0.3, beta=0.9, nu=0.4, rho=0.2)  # Max
        
        # Invalid beta values
        with pytest.raises(ValueError, match="Beta must be in range"):
            SABRParams(F0=1.0, alpha=0.3, beta=0.29, nu=0.4, rho=0.2)  # Too low
        
        with pytest.raises(ValueError, match="Beta must be in range"):
            SABRParams(F0=1.0, alpha=0.3, beta=0.91, nu=0.4, rho=0.2)  # Too high
    
    def test_nu_range_validation(self):
        """Test nu parameter range validation."""
        # Valid nu values
        SABRParams(F0=1.0, alpha=0.3, beta=0.5, nu=0.05, rho=0.2)  # Min
        SABRParams(F0=1.0, alpha=0.3, beta=0.5, nu=0.9, rho=0.2)   # Max
        
        # Invalid nu values
        with pytest.raises(ValueError, match="Nu must be in range"):
            SABRParams(F0=1.0, alpha=0.3, beta=0.5, nu=0.04, rho=0.2)  # Too low
        
        with pytest.raises(ValueError, match="Nu must be in range"):
            SABRParams(F0=1.0, alpha=0.3, beta=0.5, nu=0.91, rho=0.2)  # Too high
    
    def test_rho_range_validation(self):
        """Test rho parameter range validation."""
        # Valid rho values
        SABRParams(F0=1.0, alpha=0.3, beta=0.5, nu=0.4, rho=-0.75)  # Min
        SABRParams(F0=1.0, alpha=0.3, beta=0.5, nu=0.4, rho=0.75)   # Max
        SABRParams(F0=1.0, alpha=0.3, beta=0.5, nu=0.4, rho=0.0)    # Zero
        
        # Invalid rho values
        with pytest.raises(ValueError, match="Rho must be in range"):
            SABRParams(F0=1.0, alpha=0.3, beta=0.5, nu=0.4, rho=-0.76)  # Too low
        
        with pytest.raises(ValueError, match="Rho must be in range"):
            SABRParams(F0=1.0, alpha=0.3, beta=0.5, nu=0.4, rho=0.76)   # Too high
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        params = SABRParams(F0=1.0, alpha=0.3, beta=0.5, nu=0.4, rho=0.2)
        params_dict = params.to_dict()
        
        expected = {
            'F0': 1.0,
            'alpha': 0.3,
            'beta': 0.5,
            'nu': 0.4,
            'rho': 0.2
        }
        assert params_dict == expected
    
    def test_from_dict_conversion(self):
        """Test creation from dictionary."""
        params_dict = {
            'F0': 1.0,
            'alpha': 0.3,
            'beta': 0.5,
            'nu': 0.4,
            'rho': 0.2
        }
        params = SABRParams.from_dict(params_dict)
        
        assert params.F0 == 1.0
        assert params.alpha == 0.3
        assert params.beta == 0.5
        assert params.nu == 0.4
        assert params.rho == 0.2


class TestGridConfig:
    """Test cases for GridConfig class."""
    
    def test_default_configuration(self):
        """Test default grid configuration."""
        config = GridConfig()
        assert config.maturity_range == (1.0, 10.0)
        assert config.n_strikes == 21  # Funahashi's choice
        assert config.n_maturities == 10
        assert config.include_extended_wings == False
        assert config.validate()
    
    def test_custom_configuration(self):
        """Test custom grid configuration."""
        config = GridConfig(
            maturity_range=(0.5, 5.0),
            n_strikes=15,
            n_maturities=8,
            include_extended_wings=True
        )
        assert config.maturity_range == (0.5, 5.0)
        assert config.n_strikes == 15
        assert config.n_maturities == 8
        assert config.include_extended_wings == True
        assert config.validate()
    
    def test_invalid_maturity_range(self):
        """Test validation fails for invalid maturity range."""
        with pytest.raises(ValueError, match="Minimum maturity must be positive"):
            GridConfig(maturity_range=(0.0, 5.0))
        
        with pytest.raises(ValueError, match="Maximum maturity must be greater than minimum"):
            GridConfig(maturity_range=(5.0, 3.0))
    
    def test_invalid_grid_sizes(self):
        """Test validation fails for invalid grid sizes."""
        with pytest.raises(ValueError, match="Must have at least 3 strikes"):
            GridConfig(n_strikes=2)
        
        with pytest.raises(ValueError, match="Must have at least 1 maturity"):
            GridConfig(n_maturities=0)
    
    def test_funahashi_strike_range_calculation(self):
        """Test Funahashi's strike range formula."""
        config = GridConfig()
        params = SABRParams(F0=100.0, alpha=0.3, beta=0.5, nu=0.4, rho=0.2)
        maturity = 2.0
        
        K1, K2 = config.calculate_strike_range(params, maturity)
        
        # Calculate expected values
        f = 100.0
        alpha = 0.3
        V = (alpha ** 2) * maturity  # 0.18
        sqrt_V = np.sqrt(V)  # ~0.424
        
        expected_K1 = max(f - 1.8 * sqrt_V, 0.4 * f)  # max(100 - 1.8*0.424, 40) = max(~23.7, 40) = 40
        expected_K2 = min(f + 2.0 * sqrt_V, 2.0 * f)  # min(100 + 2*0.424, 200) = min(~100.85, 200) = ~100.85
        
        assert abs(K1 - expected_K1) < 1e-10
        assert abs(K2 - expected_K2) < 1e-10
    
    def test_strike_grid_generation(self):
        """Test strike grid generation."""
        config = GridConfig(n_strikes=21)
        params = SABRParams(F0=100.0, alpha=0.3, beta=0.5, nu=0.4, rho=0.2)
        maturity = 2.0
        
        strikes = config.generate_strike_grid(params, maturity)
        
        assert len(strikes) == 21
        assert strikes[0] >= 0  # All strikes should be positive
        assert np.all(np.diff(strikes) > 0)  # Should be monotonically increasing
        
        # Check that strikes span the expected range
        K1, K2 = config.calculate_strike_range(params, maturity)
        assert abs(strikes[0] - K1) < 1e-10
        assert abs(strikes[-1] - K2) < 1e-10
    
    def test_extended_wings_generation(self):
        """Test extended wing strikes generation."""
        config = GridConfig(n_strikes=21, include_extended_wings=True)
        params = SABRParams(F0=100.0, alpha=0.3, beta=0.5, nu=0.4, rho=0.2)
        maturity = 2.0
        
        strikes = config.generate_strike_grid(params, maturity)
        
        # Should have more than 21 strikes due to extended wings
        assert len(strikes) > 21
        
        # Check for extended wing strikes
        f = params.F0
        left_wing = 0.3 * f  # 30
        right_wing = 2.5 * f  # 250
        
        assert left_wing in strikes
        assert right_wing in strikes
        assert np.all(np.diff(strikes) > 0)  # Still monotonically increasing
    
    def test_maturity_grid_generation(self):
        """Test maturity grid generation."""
        config = GridConfig(maturity_range=(1.0, 5.0), n_maturities=5)
        maturities = config.generate_maturity_grid()
        
        assert len(maturities) == 5
        assert maturities[0] == 1.0
        assert maturities[-1] == 5.0
        assert np.all(np.diff(maturities) > 0)  # Monotonically increasing
    
    def test_full_grid_generation(self):
        """Test full strike-maturity grid generation."""
        config = GridConfig(n_strikes=5, n_maturities=3)
        params = SABRParams(F0=100.0, alpha=0.3, beta=0.5, nu=0.4, rho=0.2)
        
        K_grid, T_grid = config.generate_full_grid(params)
        
        assert K_grid.shape == (5, 3)  # (n_strikes, n_maturities)
        assert T_grid.shape == (5, 3)
        
        # Check that grids are properly structured
        assert np.all(K_grid[:, 0] == K_grid[:, 1])  # Same strikes across maturities
        assert np.all(T_grid[0, :] == T_grid[1, :])  # Same maturities across strikes


class TestParameterSampler:
    """Test cases for ParameterSampler class."""
    
    def test_uniform_sampling(self):
        """Test uniform parameter sampling."""
        sampler = ParameterSampler(random_seed=42)
        params_list = sampler.uniform_sampling(n_samples=100, F0=1.0)
        
        assert len(params_list) == 100
        
        # Check all parameters are valid
        for params in params_list:
            assert isinstance(params, SABRParams)
            assert params.validate()
            assert params.F0 == 1.0
        
        # Check parameter ranges are covered
        alphas = [p.alpha for p in params_list]
        betas = [p.beta for p in params_list]
        nus = [p.nu for p in params_list]
        rhos = [p.rho for p in params_list]
        
        assert min(alphas) >= SABRParams.ALPHA_RANGE[0]
        assert max(alphas) <= SABRParams.ALPHA_RANGE[1]
        assert min(betas) >= SABRParams.BETA_RANGE[0]
        assert max(betas) <= SABRParams.BETA_RANGE[1]
        assert min(nus) >= SABRParams.NU_RANGE[0]
        assert max(nus) <= SABRParams.NU_RANGE[1]
        assert min(rhos) >= SABRParams.RHO_RANGE[0]
        assert max(rhos) <= SABRParams.RHO_RANGE[1]
    
    def test_latin_hypercube_sampling(self):
        """Test Latin Hypercube Sampling."""
        sampler = ParameterSampler(random_seed=42)
        params_list = sampler.latin_hypercube_sampling(n_samples=50, F0=1.0)
        
        assert len(params_list) == 50
        
        # Check all parameters are valid
        for params in params_list:
            assert isinstance(params, SABRParams)
            assert params.validate()
            assert params.F0 == 1.0
        
        # LHS should provide better coverage than uniform sampling
        # Check that we have good spread across parameter space
        alphas = np.array([p.alpha for p in params_list])
        betas = np.array([p.beta for p in params_list])
        
        # Check that samples are well distributed (not clustered)
        alpha_std = np.std(alphas)
        beta_std = np.std(betas)
        
        assert alpha_std > 0.1  # Should have reasonable spread
        assert beta_std > 0.1
    
    def test_stratified_sampling(self):
        """Test stratified parameter sampling."""
        sampler = ParameterSampler(random_seed=42)
        params_list = sampler.stratified_sampling(n_samples=90, F0=1.0)  # Divisible by 3
        
        assert len(params_list) == 90
        
        # Check all parameters are valid
        for params in params_list:
            assert isinstance(params, SABRParams)
            assert params.validate()
            assert params.F0 == 1.0
        
        # Check that we have samples from different volatility regimes
        alphas = [p.alpha for p in params_list]
        nus = [p.nu for p in params_list]
        
        # Should have samples in low, medium, and high vol regimes
        low_vol_count = sum(1 for a, n in zip(alphas, nus) if a <= 0.2 and n <= 0.3)
        med_vol_count = sum(1 for a, n in zip(alphas, nus) if 0.2 < a <= 0.4 and 0.3 < n <= 0.6)
        high_vol_count = sum(1 for a, n in zip(alphas, nus) if a > 0.4 and n > 0.6)
        
        # Each regime should have approximately equal representation
        assert low_vol_count >= 25  # Should have samples from each regime
        assert med_vol_count >= 25
        assert high_vol_count >= 25
    
    def test_sampling_reproducibility(self):
        """Test that sampling is reproducible with fixed seed."""
        sampler1 = ParameterSampler(random_seed=42)
        sampler2 = ParameterSampler(random_seed=42)
        
        params1 = sampler1.uniform_sampling(n_samples=10)
        params2 = sampler2.uniform_sampling(n_samples=10)
        
        # Should generate identical parameters
        for p1, p2 in zip(params1, params2):
            assert p1.alpha == p2.alpha
            assert p1.beta == p2.beta
            assert p1.nu == p2.nu
            assert p1.rho == p2.rho
    
    def test_lhs_generation(self):
        """Test internal LHS generation method."""
        sampler = ParameterSampler(random_seed=42)
        lhs_samples = sampler._generate_lhs(n_samples=10, n_dims=4)
        
        assert lhs_samples.shape == (10, 4)
        assert np.all(lhs_samples >= 0)
        assert np.all(lhs_samples <= 1)
        
        # Check that each dimension has good stratification
        for dim in range(4):
            sorted_samples = np.sort(lhs_samples[:, dim])
            # Gaps between consecutive samples should be roughly equal
            gaps = np.diff(sorted_samples)
            gap_std = np.std(gaps)
            assert gap_std < 0.05  # Should be well stratified


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])