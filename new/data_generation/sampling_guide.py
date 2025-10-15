"""
Sampling strategy selection guide for SABR parameter generation.

This module provides guidance on when to use different sampling strategies
and implements a smart sampler that selects the appropriate method based
on use case requirements.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
import numpy as np

from data_generation.sabr_params import SABRParams, ParameterSampler


class UseCase(Enum):
    """Different use cases for SABR parameter sampling."""
    PROTOTYPING = "prototyping"
    TRAINING = "training"
    EVALUATION = "evaluation"
    REGIME_ANALYSIS = "regime_analysis"
    DEBUGGING = "debugging"
    RESEARCH = "research"


class SamplingStrategy(Enum):
    """Available sampling strategies."""
    UNIFORM = "uniform"
    LATIN_HYPERCUBE = "latin_hypercube"
    STRATIFIED = "stratified"


class SmartSampler:
    """
    Intelligent sampler that selects appropriate sampling strategy based on use case.
    
    This class encapsulates the decision logic for choosing the right sampling
    strategy based on the intended use case, dataset size, and research objectives.
    """
    
    # Strategy recommendations based on use case and dataset size
    STRATEGY_RECOMMENDATIONS = {
        UseCase.PROTOTYPING: {
            "small": SamplingStrategy.UNIFORM,
            "medium": SamplingStrategy.UNIFORM,
            "large": SamplingStrategy.LATIN_HYPERCUBE
        },
        UseCase.TRAINING: {
            "small": SamplingStrategy.LATIN_HYPERCUBE,
            "medium": SamplingStrategy.LATIN_HYPERCUBE,
            "large": SamplingStrategy.LATIN_HYPERCUBE
        },
        UseCase.EVALUATION: {
            "small": SamplingStrategy.LATIN_HYPERCUBE,
            "medium": SamplingStrategy.LATIN_HYPERCUBE,
            "large": SamplingStrategy.LATIN_HYPERCUBE
        },
        UseCase.REGIME_ANALYSIS: {
            "small": SamplingStrategy.STRATIFIED,
            "medium": SamplingStrategy.STRATIFIED,
            "large": SamplingStrategy.STRATIFIED
        },
        UseCase.DEBUGGING: {
            "small": SamplingStrategy.UNIFORM,
            "medium": SamplingStrategy.UNIFORM,
            "large": SamplingStrategy.UNIFORM
        },
        UseCase.RESEARCH: {
            "small": SamplingStrategy.STRATIFIED,
            "medium": SamplingStrategy.LATIN_HYPERCUBE,
            "large": SamplingStrategy.LATIN_HYPERCUBE
        }
    }
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize smart sampler.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.sampler = ParameterSampler(random_seed)
    
    def recommend_strategy(self, use_case: UseCase, n_samples: int) -> SamplingStrategy:
        """
        Recommend sampling strategy based on use case and dataset size.
        
        Args:
            use_case: The intended use case
            n_samples: Number of samples to generate
            
        Returns:
            Recommended sampling strategy
        """
        # Categorize dataset size
        if n_samples < 100:
            size_category = "small"
        elif n_samples < 1000:
            size_category = "medium"
        else:
            size_category = "large"
        
        return self.STRATEGY_RECOMMENDATIONS[use_case][size_category]
    
    def generate_samples(
        self, 
        use_case: UseCase, 
        n_samples: int, 
        F0: float = 1.0,
        strategy_override: Optional[SamplingStrategy] = None
    ) -> List[SABRParams]:
        """
        Generate SABR parameter samples using recommended or specified strategy.
        
        Args:
            use_case: The intended use case
            n_samples: Number of samples to generate
            F0: Forward price (default: 1.0)
            strategy_override: Override automatic strategy selection
            
        Returns:
            List of SABRParams objects
        """
        # Use override or get recommendation
        strategy = strategy_override or self.recommend_strategy(use_case, n_samples)
        
        # Generate samples using selected strategy
        if strategy == SamplingStrategy.UNIFORM:
            return self.sampler.uniform_sampling(n_samples, F0)
        elif strategy == SamplingStrategy.LATIN_HYPERCUBE:
            return self.sampler.latin_hypercube_sampling(n_samples, F0)
        elif strategy == SamplingStrategy.STRATIFIED:
            return self.sampler.stratified_sampling(n_samples, F0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def get_strategy_info(self, strategy: SamplingStrategy) -> Dict[str, Any]:
        """
        Get detailed information about a sampling strategy.
        
        Args:
            strategy: The sampling strategy
            
        Returns:
            Dictionary with strategy information
        """
        info = {
            SamplingStrategy.UNIFORM: {
                "name": "Uniform Sampling",
                "description": "Simple random sampling with equal probability across parameter space",
                "advantages": [
                    "Fast and simple",
                    "Easy to understand and debug",
                    "Good for initial exploration"
                ],
                "disadvantages": [
                    "Can have clustering issues",
                    "Poor space-filling in high dimensions",
                    "May miss important parameter regions"
                ],
                "best_for": [
                    "Prototyping and debugging",
                    "Small datasets",
                    "Initial exploration"
                ]
            },
            SamplingStrategy.LATIN_HYPERCUBE: {
                "name": "Latin Hypercube Sampling",
                "description": "Stratified sampling ensuring good coverage of each parameter dimension",
                "advantages": [
                    "Excellent space-filling properties",
                    "Better coverage than uniform sampling",
                    "Reduces clustering and gaps",
                    "Efficient for training ML models"
                ],
                "disadvantages": [
                    "Slightly more complex than uniform",
                    "May not ensure regime balance"
                ],
                "best_for": [
                    "Training neural networks",
                    "Model evaluation",
                    "Production datasets",
                    "Research studies"
                ]
            },
            SamplingStrategy.STRATIFIED: {
                "name": "Stratified Sampling",
                "description": "Sampling across predefined volatility regimes (low/medium/high)",
                "advantages": [
                    "Ensures balanced regime representation",
                    "Prevents bias toward specific market conditions",
                    "Excellent for regime analysis",
                    "Robust across market cycles"
                ],
                "disadvantages": [
                    "May not fill parameter space optimally",
                    "Requires domain knowledge for regime definition"
                ],
                "best_for": [
                    "Volatility regime analysis",
                    "Risk management applications",
                    "Comparative studies",
                    "Balanced training sets"
                ]
            }
        }
        
        return info[strategy]


def print_sampling_guide():
    """Print comprehensive sampling strategy guide."""
    print("SABR Parameter Sampling Strategy Guide")
    print("=" * 50)
    
    print("\n📋 USE CASE RECOMMENDATIONS:")
    print("-" * 30)
    
    use_case_descriptions = {
        UseCase.PROTOTYPING: "Quick testing and initial development",
        UseCase.TRAINING: "Training neural networks and ML models",
        UseCase.EVALUATION: "Model evaluation and validation",
        UseCase.REGIME_ANALYSIS: "Studying volatility regime behavior",
        UseCase.DEBUGGING: "Testing and debugging code",
        UseCase.RESEARCH: "Academic research and comprehensive studies"
    }
    
    for use_case, description in use_case_descriptions.items():
        print(f"\n🎯 {use_case.value.upper()}: {description}")
        
        for size, size_name in [("small", "< 100 samples"), ("medium", "100-1000 samples"), ("large", "> 1000 samples")]:
            strategy = SmartSampler.STRATEGY_RECOMMENDATIONS[use_case][size]
            print(f"   • {size_name}: {strategy.value}")
    
    print("\n\n📊 STRATEGY DETAILS:")
    print("-" * 30)
    
    smart_sampler = SmartSampler()
    for strategy in SamplingStrategy:
        info = smart_sampler.get_strategy_info(strategy)
        print(f"\n🔹 {info['name'].upper()}")
        print(f"   Description: {info['description']}")
        print(f"   Best for: {', '.join(info['best_for'])}")
        print(f"   Advantages: {', '.join(info['advantages'])}")
        print(f"   Disadvantages: {', '.join(info['disadvantages'])}")


def demonstrate_smart_sampling():
    """Demonstrate smart sampling for different use cases."""
    print("\n\n🤖 SMART SAMPLING DEMONSTRATION:")
    print("-" * 40)
    
    smart_sampler = SmartSampler(random_seed=42)
    
    # Test different scenarios
    scenarios = [
        (UseCase.PROTOTYPING, 50, "Quick prototype with 50 samples"),
        (UseCase.TRAINING, 5000, "Training dataset with 5000 samples"),
        (UseCase.REGIME_ANALYSIS, 300, "Regime analysis with 300 samples"),
        (UseCase.DEBUGGING, 20, "Debug run with 20 samples")
    ]
    
    for use_case, n_samples, description in scenarios:
        print(f"\n📝 Scenario: {description}")
        
        # Get recommendation
        recommended_strategy = smart_sampler.recommend_strategy(use_case, n_samples)
        print(f"   Recommended strategy: {recommended_strategy.value}")
        
        # Generate samples
        samples = smart_sampler.generate_samples(use_case, n_samples)
        
        # Analyze samples
        alphas = [p.alpha for p in samples]
        nus = [p.nu for p in samples]
        
        print(f"   Generated {len(samples)} samples")
        print(f"   Alpha range: [{min(alphas):.3f}, {max(alphas):.3f}]")
        print(f"   Nu range: [{min(nus):.3f}, {max(nus):.3f}]")
        
        # Check regime distribution for stratified sampling
        if recommended_strategy == SamplingStrategy.STRATIFIED:
            low_vol = sum(1 for a, n in zip(alphas, nus) if a <= 0.2 and n <= 0.3)
            med_vol = sum(1 for a, n in zip(alphas, nus) if 0.2 < a <= 0.4 and 0.3 < n <= 0.6)
            high_vol = sum(1 for a, n in zip(alphas, nus) if a > 0.4 and n > 0.6)
            print(f"   Regime distribution: Low={low_vol}, Med={med_vol}, High={high_vol}")


if __name__ == "__main__":
    print_sampling_guide()
    demonstrate_smart_sampling()