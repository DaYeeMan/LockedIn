"""
SABR parameter and grid configuration classes for volatility surface modeling.

This module implements the core data structures for SABR model parameters
and grid configurations following Funahashi's approach with extensions for
deep wing analysis.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Union
import warnings


@dataclass
class SABRParams:
    """
    SABR model parameters with validation based on Funahashi's ranges.
    
    Parameters follow the SABR model: dF = α F^β dW1, dα = ν α dW2
    where dW1 and dW2 have correlation ρ.
    
    Attributes:
        F0: Forward price (must be positive)
        alpha: Initial volatility [0.05-0.6] (Funahashi range)
        beta: Elasticity parameter [0.3-0.9] (Funahashi range) 
        nu: Vol-of-vol parameter [0.05-0.9] (Funahashi range)
        rho: Correlation parameter [-0.75-0.75] (Funahashi range)
    """
    F0: float
    alpha: float
    beta: float
    nu: float
    rho: float
    
    # Funahashi's parameter ranges for validation
    ALPHA_RANGE: Tuple[float, float] = field(default=(0.05, 0.6), init=False)
    BETA_RANGE: Tuple[float, float] = field(default=(0.3, 0.9), init=False)
    NU_RANGE: Tuple[float, float] = field(default=(0.05, 0.9), init=False)
    RHO_RANGE: Tuple[float, float] = field(default=(-0.75, 0.75), init=False)
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate SABR parameters against Funahashi's ranges.
        
        Returns:
            bool: True if all parameters are valid
            
        Raises:
            ValueError: If any parameter is outside valid range
        """
        if self.F0 <= 0:
            raise ValueError(f"Forward price F0 must be positive, got {self.F0}")
            
        if not (self.ALPHA_RANGE[0] <= self.alpha <= self.ALPHA_RANGE[1]):
            raise ValueError(
                f"Alpha must be in range {self.ALPHA_RANGE}, got {self.alpha}"
            )
            
        if not (self.BETA_RANGE[0] <= self.beta <= self.BETA_RANGE[1]):
            raise ValueError(
                f"Beta must be in range {self.BETA_RANGE}, got {self.beta}"
            )
            
        if not (self.NU_RANGE[0] <= self.nu <= self.NU_RANGE[1]):
            raise ValueError(
                f"Nu must be in range {self.NU_RANGE}, got {self.nu}"
            )
            
        if not (self.RHO_RANGE[0] <= self.rho <= self.RHO_RANGE[1]):
            raise ValueError(
                f"Rho must be in range {self.RHO_RANGE}, got {self.rho}"
            )
            
        return True
    
    def to_dict(self) -> dict:
        """Convert parameters to dictionary."""
        return {
            'F0': self.F0,
            'alpha': self.alpha,
            'beta': self.beta,
            'nu': self.nu,
            'rho': self.rho
        }
    
    @classmethod
    def from_dict(cls, params_dict: dict) -> 'SABRParams':
        """Create SABRParams from dictionary."""
        return cls(
            F0=params_dict['F0'],
            alpha=params_dict['alpha'],
            beta=params_dict['beta'],
            nu=params_dict['nu'],
            rho=params_dict['rho']
        )


@dataclass
class GridConfig:
    """
    Grid configuration for SABR volatility surface discretization.
    
    Implements Funahashi's strike range calculation with extensions for
    deep wing analysis. Uses 21 strikes per Funahashi's approach.
    
    Attributes:
        maturity_range: (T_min, T_max) in years [1-10 per Funahashi]
        n_strikes: Number of strikes (21 per Funahashi)
        n_maturities: Number of maturities
        include_extended_wings: Add strikes at 0.3f and 2.5f for research
    """
    maturity_range: Tuple[float, float] = (1.0, 10.0)  # Funahashi range
    n_strikes: int = 21  # Funahashi's choice
    n_maturities: int = 10
    include_extended_wings: bool = False
    
    def __post_init__(self):
        """Validate grid configuration."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate grid configuration parameters.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.maturity_range[0] <= 0:
            raise ValueError("Minimum maturity must be positive")
            
        if self.maturity_range[1] <= self.maturity_range[0]:
            raise ValueError("Maximum maturity must be greater than minimum")
            
        if self.n_strikes < 3:
            raise ValueError("Must have at least 3 strikes")
            
        if self.n_maturities < 1:
            raise ValueError("Must have at least 1 maturity")
            
        return True
    
    def calculate_strike_range(self, sabr_params: SABRParams, maturity: float) -> Tuple[float, float]:
        """
        Calculate strike range using Funahashi's formula.
        
        Funahashi's strike range:
        K1 = max(f - 1.8√V, 0.4f)
        K2 = min(f + 2√V, 2f)
        
        where V is the total variance approximation.
        
        Args:
            sabr_params: SABR model parameters
            maturity: Time to maturity in years
            
        Returns:
            Tuple of (K_min, K_max)
        """
        f = sabr_params.F0
        alpha = sabr_params.alpha
        
        # Approximate total variance for strike range calculation
        # Using simplified approximation V ≈ α²T for ATM
        V = (alpha ** 2) * maturity
        sqrt_V = np.sqrt(V)
        
        # Funahashi's strike range formula
        K1 = max(f - 1.8 * sqrt_V, 0.4 * f)
        K2 = min(f + 2.0 * sqrt_V, 2.0 * f)
        
        return K1, K2
    
    def generate_strike_grid(self, sabr_params: SABRParams, maturity: float) -> np.ndarray:
        """
        Generate strike grid for given SABR parameters and maturity.
        
        Args:
            sabr_params: SABR model parameters
            maturity: Time to maturity in years
            
        Returns:
            Array of strike values
        """
        K1, K2 = self.calculate_strike_range(sabr_params, maturity)
        
        # Generate base strikes using Funahashi's range
        base_strikes = np.linspace(K1, K2, self.n_strikes)
        
        if self.include_extended_wings:
            # Add extended wing strikes for research
            f = sabr_params.F0
            left_wing = 0.3 * f
            right_wing = 2.5 * f
            
            # Only add if they're outside the base range
            extended_strikes = []
            if left_wing < K1:
                extended_strikes.append(left_wing)
            if right_wing > K2:
                extended_strikes.append(right_wing)
                
            if extended_strikes:
                all_strikes = np.concatenate([extended_strikes[:1], base_strikes, extended_strikes[1:]])
                return np.sort(all_strikes)
        
        return base_strikes
    
    def generate_maturity_grid(self) -> np.ndarray:
        """
        Generate maturity grid.
        
        Returns:
            Array of maturity values in years
        """
        return np.linspace(self.maturity_range[0], self.maturity_range[1], self.n_maturities)
    
    def generate_full_grid(self, sabr_params: SABRParams) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate full strike-maturity grid for given SABR parameters.
        
        Args:
            sabr_params: SABR model parameters
            
        Returns:
            Tuple of (strikes_2d, maturities_2d) meshgrids
        """
        maturities = self.generate_maturity_grid()
        
        # Generate strikes for each maturity (they may vary)
        all_strikes = []
        for T in maturities:
            strikes_T = self.generate_strike_grid(sabr_params, T)
            all_strikes.append(strikes_T)
        
        # For consistent grid, use the strikes from the first maturity
        # This ensures rectangular grid structure
        base_strikes = all_strikes[0]
        
        # Create meshgrid
        K_grid, T_grid = np.meshgrid(base_strikes, maturities, indexing='ij')
        
        return K_grid, T_grid


class ParameterSampler:
    """
    Parameter sampling strategies matching Funahashi's approach.
    
    Provides various sampling strategies for generating SABR parameter sets
    for training and evaluation.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize parameter sampler.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(random_seed)
    
    def uniform_sampling(self, n_samples: int, F0: float = 1.0) -> List[SABRParams]:
        """
        Generate parameter sets using uniform sampling within Funahashi's ranges.
        
        Args:
            n_samples: Number of parameter sets to generate
            F0: Forward price (fixed)
            
        Returns:
            List of SABRParams objects
        """
        params_list = []
        
        for _ in range(n_samples):
            alpha = self.rng.uniform(*SABRParams.ALPHA_RANGE)
            beta = self.rng.uniform(*SABRParams.BETA_RANGE)
            nu = self.rng.uniform(*SABRParams.NU_RANGE)
            rho = self.rng.uniform(*SABRParams.RHO_RANGE)
            
            params = SABRParams(F0=F0, alpha=alpha, beta=beta, nu=nu, rho=rho)
            params_list.append(params)
        
        return params_list
    
    def funahashi_exact_sampling(self, n_samples: int = 4) -> List[SABRParams]:
        """
        Generate Funahashi's exact test cases for direct comparison.
        
        Args:
            n_samples: Number of samples (should be 4 for Funahashi's cases)
            
        Returns:
            List of SABRParams with Funahashi's exact test cases
        """
        return FunahashiTestCases.get_test_cases()[:n_samples]
    
    def latin_hypercube_sampling(self, n_samples: int, F0: float = 1.0) -> List[SABRParams]:
        """
        Generate parameter sets using Latin Hypercube Sampling for better coverage.
        
        Args:
            n_samples: Number of parameter sets to generate
            F0: Forward price (fixed)
            
        Returns:
            List of SABRParams objects
        """
        # Generate LHS samples in [0,1]^4
        lhs_samples = self._generate_lhs(n_samples, 4)
        
        params_list = []
        for i in range(n_samples):
            # Map to parameter ranges
            alpha = SABRParams.ALPHA_RANGE[0] + lhs_samples[i, 0] * (
                SABRParams.ALPHA_RANGE[1] - SABRParams.ALPHA_RANGE[0]
            )
            beta = SABRParams.BETA_RANGE[0] + lhs_samples[i, 1] * (
                SABRParams.BETA_RANGE[1] - SABRParams.BETA_RANGE[0]
            )
            nu = SABRParams.NU_RANGE[0] + lhs_samples[i, 2] * (
                SABRParams.NU_RANGE[1] - SABRParams.NU_RANGE[0]
            )
            rho = SABRParams.RHO_RANGE[0] + lhs_samples[i, 3] * (
                SABRParams.RHO_RANGE[1] - SABRParams.RHO_RANGE[0]
            )
            
            params = SABRParams(F0=F0, alpha=alpha, beta=beta, nu=nu, rho=rho)
            params_list.append(params)
        
        return params_list
    
    def stratified_sampling(self, n_samples: int, F0: float = 1.0) -> List[SABRParams]:
        """
        Generate parameter sets using stratified sampling to ensure coverage
        of different parameter regimes.
        
        Args:
            n_samples: Number of parameter sets to generate
            F0: Forward price (fixed)
            
        Returns:
            List of SABRParams objects
        """
        params_list = []
        
        # Define parameter regimes for stratification
        regimes = [
            # Low vol regime
            {'alpha': (0.05, 0.2), 'nu': (0.05, 0.3)},
            # Medium vol regime  
            {'alpha': (0.2, 0.4), 'nu': (0.3, 0.6)},
            # High vol regime
            {'alpha': (0.4, 0.6), 'nu': (0.6, 0.9)}
        ]
        
        samples_per_regime = n_samples // len(regimes)
        remaining_samples = n_samples % len(regimes)
        
        for i, regime in enumerate(regimes):
            n_regime_samples = samples_per_regime
            if i < remaining_samples:
                n_regime_samples += 1
            
            for _ in range(n_regime_samples):
                alpha = self.rng.uniform(*regime['alpha'])
                beta = self.rng.uniform(*SABRParams.BETA_RANGE)
                nu = self.rng.uniform(*regime['nu'])
                rho = self.rng.uniform(*SABRParams.RHO_RANGE)
                
                params = SABRParams(F0=F0, alpha=alpha, beta=beta, nu=nu, rho=rho)
                params_list.append(params)
        
        return params_list
    
    def _generate_lhs(self, n_samples: int, n_dims: int) -> np.ndarray:
        """
        Generate Latin Hypercube Sampling points.
        
        Args:
            n_samples: Number of samples
            n_dims: Number of dimensions
            
        Returns:
            Array of shape (n_samples, n_dims) with LHS points in [0,1]
        """
        samples = np.zeros((n_samples, n_dims))
        
        for dim in range(n_dims):
            # Create stratified samples
            intervals = np.arange(n_samples) / n_samples
            # Add random jitter within each interval
            jitter = self.rng.uniform(0, 1/n_samples, n_samples)
            samples[:, dim] = intervals + jitter
            
            # Shuffle to break correlation between dimensions
            self.rng.shuffle(samples[:, dim])
        
        return samples

class FunahashiTestCases:
    """
    Funahashi's exact test cases for direct comparison.
    
    This class provides the exact SABR parameter sets used in Funahashi's paper
    for direct result comparison.
    """
    
    @staticmethod
    def get_test_cases() -> List[SABRParams]:
        """
        Get Funahashi's exact test cases from his paper.
        
        Returns:
            List of SABRParams for the 4 test cases
        """
        test_cases = [
            # Case A: f=1, α=0.5, β=0.6, ν=0.3, ρ=-0.2
            SABRParams(F0=1.0, alpha=0.5, beta=0.6, nu=0.3, rho=-0.2),
            
            # Case B: f=1, α=0.5, β=0.9, ν=0.3, ρ=-0.2
            SABRParams(F0=1.0, alpha=0.5, beta=0.9, nu=0.3, rho=-0.2),
            
            # Case C: f=1, α=0.5, β=0.3, ν=0.3, ρ=-0.2
            SABRParams(F0=1.0, alpha=0.5, beta=0.3, nu=0.3, rho=-0.2),
            
            # Case D: f=1, α=0.5, β=0.6, ν=0.3, ρ=-0.5
            SABRParams(F0=1.0, alpha=0.5, beta=0.6, nu=0.3, rho=-0.5),
        ]
        
        return test_cases
    
    @staticmethod
    def get_funahashi_strikes() -> np.ndarray:
        """
        Get Funahashi's exact strike values from Table 3.
        
        Returns:
            Array of strike values used in Funahashi's paper
        """
        return np.array([
            0.4, 0.485, 0.57, 0.655, 0.74, 0.825, 0.91, 0.995, 1.08,
            1.165, 1.25, 1.335, 1.42, 1.505, 1.59, 1.675, 1.76, 1.845,
            1.93, 2.015, 2.1
        ])
    
    @staticmethod
    def get_case_names() -> List[str]:
        """Get case names for labeling."""
        return ["Case A", "Case B", "Case C", "Case D"]


def create_funahashi_comparison_sampler() -> 'ParameterSampler':
    """
    Create a parameter sampler configured for Funahashi comparison.
    
    Returns:
        ParameterSampler instance with Funahashi's exact test cases
    """
    sampler = ParameterSampler(random_seed=12345)
    sampler._funahashi_mode = True
    return sampler