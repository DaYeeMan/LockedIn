"""
Multi-fidelity Data Aggregation CNN (MDA-CNN) for SABR volatility surface modeling.

This module implements the MDA-CNN architecture in PyTorch that combines:
1. CNN branch for processing local LF surface patches
2. MLP branch for processing point features (SABR parameters, strike, maturity)
3. Fusion layer to combine representations
4. Residual prediction head for D(ξ) = σ_MC(ξ) - σ_Hagan(ξ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class CNNBranch(nn.Module):
    """CNN branch for processing local LF surface patches."""
    
    def __init__(self, patch_size: int = 9, channels: List[int] = None, 
                 kernel_sizes: List[int] = None):
        super(CNNBranch, self).__init__()
        
        self.patch_size = patch_size
        channels = channels or [32, 64, 128]
        kernel_sizes = kernel_sizes or [3, 3, 3]
        
        # CNN layers for patch processing
        layers = []
        in_channels = 1  # Single channel input (volatility surface patch)
        
        for i, (out_channels, kernel_size) in enumerate(zip(channels, kernel_sizes)):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Dense layer for feature extraction
        self.dense = nn.Linear(channels[-1], 128)
        
    def forward(self, x):
        """Forward pass through CNN branch.
        
        Args:
            x: Tensor of shape (batch_size, 1, patch_height, patch_width)
            
        Returns:
            Tensor of shape (batch_size, 128) - CNN features
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch_size, 1, height, width)
        
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.dense(x))
        
        return x


class MLPBranch(nn.Module):
    """MLP branch for processing point features."""
    
    def __init__(self, n_point_features: int = 10, hidden_dims: List[int] = None):
        super(MLPBranch, self).__init__()
        
        self.n_point_features = n_point_features
        hidden_dims = hidden_dims or [64, 64]
        
        # MLP layers for point feature processing
        layers = []
        in_dim = n_point_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim)
            ])
            in_dim = hidden_dim
        
        self.mlp_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass through MLP branch.
        
        Args:
            x: Tensor of shape (batch_size, n_point_features)
            
        Returns:
            Tensor of shape (batch_size, hidden_dims[-1]) - MLP features
        """
        return self.mlp_layers(x)


class FusionHead(nn.Module):
    """Fusion layer to combine CNN and MLP representations."""
    
    def __init__(self, cnn_features: int = 128, mlp_features: int = 64, 
                 fusion_dim: int = 128, dropout_rate: float = 0.2):
        super(FusionHead, self).__init__()
        
        # Fusion layers
        self.fusion_dense1 = nn.Linear(cnn_features + mlp_features, fusion_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fusion_dense2 = nn.Linear(fusion_dim, 64)
        
        # Residual prediction head with linear activation
        self.residual_head = nn.Linear(64, 1)
        
    def forward(self, cnn_features, mlp_features):
        """Forward pass through fusion head.
        
        Args:
            cnn_features: Tensor of shape (batch_size, cnn_features)
            mlp_features: Tensor of shape (batch_size, mlp_features)
            
        Returns:
            Tensor of shape (batch_size, 1) - Residual predictions
        """
        # Concatenate CNN and MLP features
        x = torch.cat([cnn_features, mlp_features], dim=1)
        
        # Fusion layers
        x = F.relu(self.fusion_dense1(x))
        x = self.dropout(x)
        x = F.relu(self.fusion_dense2(x))
        
        # Residual prediction (linear activation)
        residual = self.residual_head(x)
        
        return residual


class MDACNNModel(nn.Module):
    """
    Complete MDA-CNN model for SABR volatility surface residual prediction.
    
    Architecture:
    1. CNN branch processes local LF surface patches
    2. MLP branch processes point features (SABR params, strike, maturity, etc.)
    3. Fusion head combines both representations
    4. Outputs residual: D(ξ) = σ_MC(ξ) - σ_Hagan(ξ)
    """
    
    def __init__(self, patch_size: int = 9, point_features_dim: int = 10,
                 cnn_channels: List[int] = None, cnn_kernel_sizes: List[int] = None,
                 mlp_hidden_dims: List[int] = None, fusion_dim: int = 128,
                 dropout_rate: float = 0.2):
        super(MDACNNModel, self).__init__()
        
        self.patch_size = patch_size
        self.point_features_dim = point_features_dim
        
        # Initialize branches
        self.cnn_branch = CNNBranch(
            patch_size=patch_size,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes
        )
        
        self.mlp_branch = MLPBranch(
            n_point_features=point_features_dim,
            hidden_dims=mlp_hidden_dims
        )
        
        # Get feature dimensions
        cnn_features = 128  # Fixed output from CNN branch
        mlp_features = (mlp_hidden_dims or [64, 64])[-1]
        
        self.fusion_head = FusionHead(
            cnn_features=cnn_features,
            mlp_features=mlp_features,
            fusion_dim=fusion_dim,
            dropout_rate=dropout_rate
        )
        
    def forward(self, patches, point_features):
        """Forward pass through complete MDA-CNN model.
        
        Args:
            patches: Tensor of shape (batch_size, patch_height, patch_width) or
                    (batch_size, 1, patch_height, patch_width)
            point_features: Tensor of shape (batch_size, point_features_dim)
            
        Returns:
            Tensor of shape (batch_size, 1) - Residual predictions
        """
        # Process patches through CNN branch
        cnn_features = self.cnn_branch(patches)
        
        # Process point features through MLP branch
        mlp_features = self.mlp_branch(point_features)
        
        # Combine features and predict residual
        residual = self.fusion_head(cnn_features, mlp_features)
        
        return residual
    
    def get_feature_representations(self, patches, point_features):
        """Get intermediate feature representations for analysis.
        
        Args:
            patches: Tensor of shape (batch_size, patch_height, patch_width)
            point_features: Tensor of shape (batch_size, point_features_dim)
            
        Returns:
            Dictionary with CNN features, MLP features, and fused features
        """
        with torch.no_grad():
            cnn_features = self.cnn_branch(patches)
            mlp_features = self.mlp_branch(point_features)
            
            # Get fused features (before final residual head)
            fused_input = torch.cat([cnn_features, mlp_features], dim=1)
            fused_features = F.relu(self.fusion_head.fusion_dense1(fused_input))
            
            return {
                'cnn_features': cnn_features,
                'mlp_features': mlp_features,
                'fused_features': fused_features
            }
    
    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """Get model architecture information."""
        total_params = self.count_parameters()
        cnn_params = sum(p.numel() for p in self.cnn_branch.parameters() if p.requires_grad)
        mlp_params = sum(p.numel() for p in self.mlp_branch.parameters() if p.requires_grad)
        fusion_params = sum(p.numel() for p in self.fusion_head.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'cnn_parameters': cnn_params,
            'mlp_parameters': mlp_params,
            'fusion_parameters': fusion_params,
            'patch_size': self.patch_size,
            'point_features_dim': self.point_features_dim
        }


def create_mdacnn_model(config) -> MDACNNModel:
    """
    Create MDA-CNN model from configuration.
    
    Args:
        config: Training configuration object
        
    Returns:
        MDACNNModel instance
    """
    return MDACNNModel(
        patch_size=config.patch_size,
        point_features_dim=config.point_features_dim,
        cnn_channels=config.cnn_channels,
        cnn_kernel_sizes=config.cnn_kernel_sizes,
        mlp_hidden_dims=config.mlp_hidden_dims,
        fusion_dim=config.fusion_dim,
        dropout_rate=config.dropout_rate
    )