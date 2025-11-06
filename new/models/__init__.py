"""
SABR Volatility Surface Models Package

This package contains the MDA-CNN architecture and related model components
for SABR volatility surface modeling.
"""

from .mda_cnn import (
    MDACNN,
    CNNBranch,
    MLPBranch,
    FusionHead,
    create_mda_cnn_model,
    build_model_with_inputs
)

from .model_utils import (
    get_model_summary_string,
    count_trainable_parameters,
    create_residual_block,
    ResidualBlock,
    create_mlp_block,
    validate_input_shapes,
    create_custom_metrics,
    ModelCheckpointCallback,
    create_learning_rate_scheduler,
    get_model_flops
)

from .loss_functions import (
    WeightedMSELoss,
    HuberLoss,
    RelativePercentageErrorLoss,
    CombinedLoss,
    create_wing_weighted_mse,
    create_robust_huber_loss,
    create_combined_mse_rpe_loss,
    relative_percentage_error_metric,
    rmse_metric,
    wing_region_mse,
    atm_region_mse
)

__all__ = [
    # Main model classes
    'MDACNN',
    'CNNBranch', 
    'MLPBranch',
    'FusionHead',
    
    # Model factory functions
    'create_mda_cnn_model',
    'build_model_with_inputs',
    
    # Utilities
    'get_model_summary_string',
    'count_trainable_parameters',
    'create_residual_block',
    'ResidualBlock',
    'create_mlp_block',
    'validate_input_shapes',
    'create_custom_metrics',
    'ModelCheckpointCallback',
    'create_learning_rate_scheduler',
    'get_model_flops',
    
    # Loss functions
    'WeightedMSELoss',
    'HuberLoss', 
    'RelativePercentageErrorLoss',
    'CombinedLoss',
    'create_wing_weighted_mse',
    'create_robust_huber_loss',
    'create_combined_mse_rpe_loss',
    
    # Metrics
    'relative_percentage_error_metric',
    'rmse_metric',
    'wing_region_mse',
    'atm_region_mse'
]