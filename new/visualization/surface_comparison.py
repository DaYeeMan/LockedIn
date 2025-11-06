"""
Surface visualization and comparison tools for SABR volatility surface modeling.

This module provides comprehensive visualization capabilities for comparing
MC, Hagan, baseline (Funahashi), and MDA-CNN surfaces with error analysis
and side-by-side comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Tuple, Optional, Dict, Any, List, Union
import warnings
from dataclasses import dataclass
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    # Create dummy classes for when plotly is not available
    class DummyFigure:
        def show(self): pass
        def write_html(self, *args, **kwargs): pass
    
    class go:
        Figure = DummyFigure
        Scatter = dict
        Surface = dict
        Heatmap = dict
        Histogram = dict
    
    class px:
        colors = type('colors', (), {'qualitative': type('qualitative', (), {'Set1': ['red', 'blue', 'green', 'orange']})})()
    
    def make_subplots(*args, **kwargs):
        return DummyFigure()

import pandas as pd
from pathlib import Path

from data_generation.sabr_params import SABRParams, GridConfig
from data_generation.sabr_mc_generator import MCResult
from data_generation.hagan_surface_generator import HaganResult


@dataclass
class SurfaceComparisonConfig:
    """
    Configuration for surface comparison visualization.
    
    Attributes:
        figure_size: Default figure size for matplotlib plots
        dpi: Resolution for saved figures
        color_scheme: Color scheme for plots
        save_format: Format for saved figures
        interactive: Use interactive plotly plots when possible
        show_plots: Display plots immediately
        save_plots: Save plots to files
        output_dir: Directory for saved plots
        surface_alpha: Transparency for 3D surfaces
        error_colormap: Colormap for error visualization
    """
    figure_size: Tuple[int, int] = (15, 10)
    dpi: int = 300
    color_scheme: str = 'viridis'
    save_format: str = 'png'
    interactive: bool = True
    show_plots: bool = True
    save_plots: bool = True
    output_dir: str = 'new/results/plots'
    surface_alpha: float = 0.8
    error_colormap: str = 'RdBu_r'


@dataclass
class ModelPredictions:
    """Container for model predictions on volatility surfaces."""
    strikes: np.ndarray
    maturities: np.ndarray
    funahashi_predictions: np.ndarray  # Funahashi baseline predictions
    mdacnn_predictions: np.ndarray     # MDA-CNN predictions
    hagan_surface: np.ndarray          # LF Hagan surface
    mc_surface: np.ndarray             # HF MC surface (ground truth)
    sabr_params: SABRParams


class SurfaceComparator:
    """
    Comprehensive surface visualization and comparison tools.
    
    Provides methods for visualizing and comparing MC, Hagan, Funahashi baseline,
    and MDA-CNN surfaces with detailed error analysis.
    """
    
    def __init__(self, config: SurfaceComparisonConfig = None):
        """
        Initialize surface comparator.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or SurfaceComparisonConfig()
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette(self.config.color_scheme)
        
        # Create output directory if needed
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def plot_all_surfaces_3d(self, predictions: ModelPredictions, 
                           maturity_idx: int = 0,
                           title_suffix: str = "") -> go.Figure:
        """
        Create 3D surface plots comparing MC, Hagan, Funahashi, and MDA-CNN surfaces.
        
        Args:
            predictions: Model predictions container
            maturity_idx: Index of maturity to plot
            title_suffix: Additional text for plot title
            
        Returns:
            Plotly figure object with 2x2 subplot layout
        """
        if maturity_idx >= len(predictions.maturities):
            raise ValueError(f"Maturity index {maturity_idx} out of range")
        
        maturity = predictions.maturities[maturity_idx]
        
        # Extract data for the specified maturity
        strikes = predictions.strikes[maturity_idx]
        mc_vols = predictions.mc_surface[maturity_idx]
        hagan_vols = predictions.hagan_surface[maturity_idx]
        funahashi_vols = predictions.funahashi_predictions[maturity_idx]
        mdacnn_vols = predictions.mdacnn_predictions[maturity_idx]
        
        # Filter valid data
        valid_mask = (~np.isnan(strikes) & ~np.isnan(mc_vols) & 
                     ~np.isnan(hagan_vols) & ~np.isnan(funahashi_vols) & 
                     ~np.isnan(mdacnn_vols))
        
        if np.sum(valid_mask) == 0:
            raise ValueError("No valid data points found")
        
        # Create 2x2 subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Monte Carlo (Ground Truth)',
                'Hagan Analytical (Low Fidelity)',
                'Funahashi Baseline Prediction',
                'MDA-CNN Prediction'
            ],
            specs=[[{'type': 'surface'}, {'type': 'surface'}],
                   [{'type': 'surface'}, {'type': 'surface'}]]
        )
        
        # Prepare meshgrid for 3D plotting
        strikes_valid = strikes[valid_mask]
        mc_vols_valid = mc_vols[valid_mask]
        hagan_vols_valid = hagan_vols[valid_mask]
        funahashi_vols_valid = funahashi_vols[valid_mask]
        mdacnn_vols_valid = mdacnn_vols[valid_mask]
        
        # Create grid for interpolation
        n_points = 50
        strike_grid = np.linspace(np.min(strikes_valid), np.max(strikes_valid), n_points)
        maturity_grid = np.array([maturity])
        
        K_mesh, T_mesh = np.meshgrid(strike_grid, maturity_grid)
        
        # Interpolate surfaces
        mc_surface = np.tile(np.interp(strike_grid, strikes_valid, mc_vols_valid), (1, 1))
        hagan_surface = np.tile(np.interp(strike_grid, strikes_valid, hagan_vols_valid), (1, 1))
        funahashi_surface = np.tile(np.interp(strike_grid, strikes_valid, funahashi_vols_valid), (1, 1))
        mdacnn_surface = np.tile(np.interp(strike_grid, strikes_valid, mdacnn_vols_valid), (1, 1))
        
        # Add surfaces to subplots
        surfaces = [
            (mc_surface, 'Blues', 1, 1),
            (hagan_surface, 'Reds', 1, 2),
            (funahashi_surface, 'Greens', 2, 1),
            (mdacnn_surface, 'Purples', 2, 2)
        ]
        
        for surface, colorscale, row, col in surfaces:
            fig.add_trace(
                go.Surface(
                    x=K_mesh,
                    y=T_mesh,
                    z=surface,
                    colorscale=colorscale,
                    showscale=False,
                    opacity=self.config.surface_alpha
                ),
                row=row, col=col
            )
        
        # Update layout
        title = f"SABR Volatility Surfaces Comparison (T={maturity:.2f}){title_suffix}"
        fig.update_layout(
            title=title,
            height=800,
            scene=dict(
                xaxis_title='Strike',
                yaxis_title='Maturity',
                zaxis_title='Implied Volatility'
            ),
            scene2=dict(
                xaxis_title='Strike',
                yaxis_title='Maturity',
                zaxis_title='Implied Volatility'
            ),
            scene3=dict(
                xaxis_title='Strike',
                yaxis_title='Maturity',
                zaxis_title='Implied Volatility'
            ),
            scene4=dict(
                xaxis_title='Strike',
                yaxis_title='Maturity',
                zaxis_title='Implied Volatility'
            )
        )
        
        if self.config.show_plots:
            fig.show()
        
        if self.config.save_plots:
            filename = f"all_surfaces_3d_T{maturity:.2f}.html"
            fig.write_html(f"{self.config.output_dir}/{filename}")
        
        return fig
    
    def plot_volatility_smiles_comparison(self, predictions: ModelPredictions,
                                        maturity_indices: List[int] = None,
                                        title_suffix: str = "") -> go.Figure:
        """
        Create volatility smile plots comparing all four approaches.
        
        Args:
            predictions: Model predictions container
            maturity_indices: List of maturity indices to plot (default: first 3)
            title_suffix: Additional text for plot title
            
        Returns:
            Plotly figure object
        """
        if maturity_indices is None:
            maturity_indices = list(range(min(3, len(predictions.maturities))))
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        line_styles = ['solid', 'dash', 'dot', 'dashdot']
        
        for i, mat_idx in enumerate(maturity_indices):
            if mat_idx >= len(predictions.maturities):
                continue
                
            maturity = predictions.maturities[mat_idx]
            color = colors[i % len(colors)]
            
            # Extract data for this maturity
            strikes = predictions.strikes[mat_idx]
            mc_vols = predictions.mc_surface[mat_idx]
            hagan_vols = predictions.hagan_surface[mat_idx]
            funahashi_vols = predictions.funahashi_predictions[mat_idx]
            mdacnn_vols = predictions.mdacnn_predictions[mat_idx]
            
            # Filter valid data
            valid_mask = (~np.isnan(strikes) & ~np.isnan(mc_vols) & 
                         ~np.isnan(hagan_vols) & ~np.isnan(funahashi_vols) & 
                         ~np.isnan(mdacnn_vols))
            
            if np.sum(valid_mask) == 0:
                continue
            
            # Calculate moneyness
            moneyness = strikes[valid_mask] / predictions.sabr_params.F0
            
            # Sort by moneyness for smooth lines
            sort_idx = np.argsort(moneyness)
            moneyness_sorted = moneyness[sort_idx]
            
            # Add traces for each surface type
            surfaces = [
                (mc_vols[valid_mask][sort_idx], 'MC', 'circle', 'solid'),
                (hagan_vols[valid_mask][sort_idx], 'Hagan', 'square', 'dash'),
                (funahashi_vols[valid_mask][sort_idx], 'Funahashi', 'diamond', 'dot'),
                (mdacnn_vols[valid_mask][sort_idx], 'MDA-CNN', 'triangle-up', 'dashdot')
            ]
            
            for vols, name, symbol, line_style in surfaces:
                fig.add_trace(go.Scatter(
                    x=moneyness_sorted,
                    y=vols,
                    mode='lines+markers',
                    name=f'{name} T={maturity:.2f}',
                    line=dict(color=color, dash=line_style),
                    marker=dict(symbol=symbol, size=6),
                    legendgroup=f'T{mat_idx}'
                ))
        
        # Add vertical line at ATM
        fig.add_vline(x=1.0, line_dash="dot", line_color="gray", 
                     annotation_text="ATM", annotation_position="top")
        
        # Update layout
        title = f"SABR Volatility Smiles Comparison{title_suffix}"
        fig.update_layout(
            title=title,
            xaxis_title='Moneyness (K/F)',
            yaxis_title='Implied Volatility',
            width=1000,
            height=600,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        
        if self.config.show_plots:
            fig.show()
        
        if self.config.save_plots:
            filename = f"volatility_smiles_comparison.html"
            fig.write_html(f"{self.config.output_dir}/{filename}")
        
        return fig