"""
Random seed management utilities for reproducibility.
Ensures consistent results across different runs and libraries.
"""

import random
import numpy as np
import os
from typing import Optional
import logging

# Try to import TensorFlow/Keras if available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Try to import PyTorch if available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_global_seed(seed: int = 42) -> None:
    """
    Set global random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Setting global random seed to {seed}")
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set environment variable for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set TensorFlow seed if available
    if TF_AVAILABLE:
        tf.random.set_seed(seed)
        logger.debug("TensorFlow random seed set")
    
    # Set PyTorch seed if available
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        logger.debug("PyTorch random seed set")


def configure_deterministic_behavior(enable: bool = True) -> None:
    """
    Configure deterministic behavior for deep learning frameworks.
    
    Args:
        enable: Whether to enable deterministic behavior
    """
    logger = logging.getLogger(__name__)
    
    if enable:
        logger.info("Enabling deterministic behavior")
        
        # TensorFlow deterministic operations
        if TF_AVAILABLE:
            # Enable deterministic operations
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
            
            # Configure TensorFlow for reproducibility
            tf.config.experimental.enable_op_determinism()
            logger.debug("TensorFlow deterministic operations enabled")
        
        # PyTorch deterministic operations
        if TORCH_AVAILABLE:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            logger.debug("PyTorch deterministic operations enabled")
    
    else:
        logger.info("Disabling deterministic behavior for better performance")
        
        if TF_AVAILABLE:
            os.environ.pop('TF_DETERMINISTIC_OPS', None)
            os.environ.pop('TF_CUDNN_DETERMINISTIC', None)
        
        if TORCH_AVAILABLE:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            torch.use_deterministic_algorithms(False)


class SeedManager:
    """
    Context manager for temporary seed changes.
    Useful for generating reproducible random data while maintaining
    different seeds for different components.
    """
    
    def __init__(self, seed: int):
        """
        Initialize seed manager.
        
        Args:
            seed: Temporary seed to use
        """
        self.seed = seed
        self.original_states = {}
    
    def __enter__(self):
        """Save current random states and set new seed."""
        # Save current states
        self.original_states['python'] = random.getstate()
        self.original_states['numpy'] = np.random.get_state()
        
        if TF_AVAILABLE:
            # TensorFlow doesn't have a direct way to get/set state
            # We'll just set the seed
            pass
        
        if TORCH_AVAILABLE:
            self.original_states['torch'] = torch.get_rng_state()
            if torch.cuda.is_available():
                self.original_states['torch_cuda'] = torch.cuda.get_rng_state()
        
        # Set new seed
        set_global_seed(self.seed)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original random states."""
        # Restore states
        random.setstate(self.original_states['python'])
        np.random.set_state(self.original_states['numpy'])
        
        if TORCH_AVAILABLE and 'torch' in self.original_states:
            torch.set_rng_state(self.original_states['torch'])
            if torch.cuda.is_available() and 'torch_cuda' in self.original_states:
                torch.cuda.set_rng_state(self.original_states['torch_cuda'])


def generate_experiment_seeds(base_seed: int = 42, n_seeds: int = 10) -> list:
    """
    Generate a list of seeds for multiple experiment runs.
    
    Args:
        base_seed: Base seed for generation
        n_seeds: Number of seeds to generate
        
    Returns:
        List of seeds for experiments
    """
    with SeedManager(base_seed):
        seeds = [random.randint(0, 2**32 - 1) for _ in range(n_seeds)]
    
    return seeds


def create_reproducible_split(
    data_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> tuple:
    """
    Create reproducible train/validation/test splits.
    
    Args:
        data_size: Total size of dataset
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for splitting
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    with SeedManager(seed):
        indices = np.arange(data_size)
        np.random.shuffle(indices)
        
        train_size = int(data_size * train_ratio)
        val_size = int(data_size * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices


def seed_worker(worker_id: int, base_seed: int = 42) -> None:
    """
    Seed function for DataLoader workers to ensure reproducibility.
    
    Args:
        worker_id: Worker ID
        base_seed: Base seed for generation
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ReproducibilityManager:
    """
    Manager class for handling all aspects of reproducibility.
    """
    
    def __init__(self, seed: int = 42, deterministic: bool = True):
        """
        Initialize reproducibility manager.
        
        Args:
            seed: Global random seed
            deterministic: Whether to enable deterministic behavior
        """
        self.seed = seed
        self.deterministic = deterministic
        self.logger = logging.getLogger(__name__)
    
    def setup(self) -> None:
        """Set up reproducible environment."""
        self.logger.info(f"Setting up reproducible environment with seed {self.seed}")
        
        # Set global seed
        set_global_seed(self.seed)
        
        # Configure deterministic behavior
        configure_deterministic_behavior(self.deterministic)
        
        # Log environment information
        self._log_environment_info()
    
    def _log_environment_info(self) -> None:
        """Log information about the current environment."""
        self.logger.info(f"Python random seed: {self.seed}")
        self.logger.info(f"NumPy version: {np.__version__}")
        
        if TF_AVAILABLE:
            self.logger.info(f"TensorFlow version: {tf.__version__}")
            self.logger.info(f"TensorFlow deterministic ops: {os.environ.get('TF_DETERMINISTIC_OPS', 'Not set')}")
        
        if TORCH_AVAILABLE:
            self.logger.info(f"PyTorch version: {torch.__version__}")
            self.logger.info(f"PyTorch deterministic: {torch.backends.cudnn.deterministic}")
    
    def create_data_splits(
        self,
        data_size: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> tuple:
        """Create reproducible data splits."""
        return create_reproducible_split(
            data_size, train_ratio, val_ratio, test_ratio, self.seed
        )
    
    def get_experiment_seeds(self, n_experiments: int = 5) -> list:
        """Get seeds for multiple experiment runs."""
        return generate_experiment_seeds(self.seed, n_experiments)


# Global reproducibility manager instance
_global_manager: Optional[ReproducibilityManager] = None


def get_reproducibility_manager() -> ReproducibilityManager:
    """Get the global reproducibility manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = ReproducibilityManager()
    return _global_manager


def setup_reproducibility(seed: int = 42, deterministic: bool = True) -> None:
    """
    Convenience function to set up reproducible environment.
    
    Args:
        seed: Global random seed
        deterministic: Whether to enable deterministic behavior
    """
    global _global_manager
    _global_manager = ReproducibilityManager(seed, deterministic)
    _global_manager.setup()