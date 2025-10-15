"""
Data visualization tools for SABR volatility surface generation.

This module provides comprehensive visualization capabilities for MC and Hagan
surface comparison, residual analysis, and parameter space exploration during
the data generation phase.
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
        colors = type('colors', (), {'qualitative': type('qualitative', (), {'Set1': ['red', 'blue', 'green']})})()
    
    def make_subplots(*args, **kwargs):
        return DummyFigure()
import pandas as pd

from data_generation.sabr_params import SABRParams, GridConfig
from data_generation.sabr_mc_generator import MCResult
from data_generation.hagan_surface_generator import HaganResult


@dataclass
class VisualizationConfig:
    """
    Configuration for visualization settings.
    
    Attributes:
        figure_size: Default figure size for matplotlib plots
        dpi: Resolution for saved figures
        color_scheme: Color scheme for plots ('viridis', 'plasma', 'coolwarm', etc.)
        save_format: Format for saved figures ('png', 'pdf', 'svg')
        interactive: Use interactive plotly plots when possible
        show_plots: Display plots immediately
        save_plots: Save plots to files
        output_dir: Directory for saved plots
    """
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    color_scheme: str = 'viridis'
    save_format: str = 'png'
    interactive: bool = True
    show_plots: bool = True
    save_plots: bool = True
    output_dir: str = 'new/results/plots'


class DataVisualizer:
    """
    Comprehensive visualization tools for SABR surface data generation.
    
    Provides methods for visualizing MC and Hagan surfaces, residual analysis,
    parameter space exploration, and statistical distributions.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        """
        Initialize data visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette(self.config.color_scheme)
        
        # Create output directory if needed
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def plot_surface_comparison_3d(self, mc_result: MCResult, hagan_result: HaganResult,
                                 sabr_params: SABRParams, maturity_idx: int = 0,
                                 title_suffix: str = "") -> go.Figure:
        """
        Create 3D surface comparison plot showing MC vs Hagan surfaces.
        
        Args:
            mc_result: Monte Carlo simulation result
            hagan_result: Hagan analytical result
            sabr_params: SABR parameters used
            maturity_idx: Index of maturity to plot
            title_suffix: Additional text for plot title
            
        Returns:
            Plotly figure object
        """
        if maturity_idx >= len(mc_result.maturities):
            raise ValueError(f"Maturity index {maturity_idx} out of range")
        
        maturity = mc_result.maturities[maturity_idx]
        
        # Extract data for the specified maturity
        mc_strikes = mc_result.strikes[maturity_idx]
        mc_vols = mc_result.volatility_surface[maturity_idx]
        hagan_strikes = hagan_result.strikes[maturity_idx]
        hagan_vols = hagan_result.volatility_surface[maturity_idx]
        
        # Filter valid data
        mc_valid = ~np.isnan(mc_strikes) & ~np.isnan(mc_vols)
        hagan_valid = ~np.isnan(hagan_strikes) & ~np.isnan(hagan_vols)
        
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Monte Carlo Surface', 'Hagan Analytical Surface'],
            specs=[[{'type': 'surface'}, {'type': 'surface'}]]
        )
        
        # Prepare meshgrid for 3D plotting
        if np.sum(mc_valid) > 0 and np.sum(hagan_valid) > 0:
            # Use common strike range
            min_strike = min(np.min(mc_strikes[mc_valid]), np.min(hagan_strikes[hagan_valid]))
            max_strike = max(np.max(mc_strikes[mc_valid]), np.max(hagan_strikes[hagan_valid]))
            
            # Create grid for interpolation
            strike_grid = np.linspace(min_strike, max_strike, 50)
            maturity_grid = np.array([maturity])
            
            K_mesh, T_mesh = np.meshgrid(strike_grid, maturity_grid)
            
            # Interpolate MC surface
            mc_vol_interp = np.interp(strike_grid, mc_strikes[mc_valid], mc_vols[mc_valid])
            mc_surface = np.tile(mc_vol_interp, (1, 1))
            
            # Interpolate Hagan surface  
            hagan_vol_interp = np.interp(strike_grid, hagan_strikes[hagan_valid], hagan_vols[hagan_valid])
            hagan_surface = np.tile(hagan_vol_interp, (1, 1))
            
            # Add MC surface
            fig.add_trace(
                go.Surface(
                    x=K_mesh,
                    y=T_mesh,
                    z=mc_surface,
                    colorscale='Blues',
                    name='Monte Carlo',
                    showscale=False
                ),
                row=1, col=1
            )
            
            # Add Hagan surface
            fig.add_trace(
                go.Surface(
                    x=K_mesh,
                    y=T_mesh,
                    z=hagan_surface,
                    colorscale='Reds',
                    name='Hagan',
                    showscale=False
                ),
                row=1, col=2
            )
        
        # Update layout
        title = f"SABR Volatility Surfaces Comparison (T={maturity:.2f}){title_suffix}"
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Strike',
                yaxis_title='Maturity',
                zaxis_title='Implied Volatility'
            ),
            scene2=dict(
                xaxis_title='Strike',
                yaxis_title='Maturity', 
                zaxis_title='Implied Volatility'
            )
        )
        
        if self.config.show_plots:
            fig.show()
        
        if self.config.save_plots:
            filename = f"surface_comparison_3d_T{maturity:.2f}.html"
            fig.write_html(f"{self.config.output_dir}/{filename}")
        
        return fig
    
    def plot_residual_heatmap(self, mc_result: MCResult, hagan_result: HaganResult,
                            sabr_params: SABRParams, title_suffix: str = "") -> go.Figure:
        """
        Create residual heatmap showing D(ξ) = σ_MC(ξ) - σ_Hagan(ξ).
        
        Args:
            mc_result: Monte Carlo simulation result
            hagan_result: Hagan analytical result
            sabr_params: SABR parameters used
            title_suffix: Additional text for plot title
            
        Returns:
            Plotly figure object
        """
        # Calculate residuals
        residuals = self._calculate_residuals(mc_result, hagan_result)
        
        # Create meshgrid for heatmap
        n_maturities = len(mc_result.maturities)
        max_strikes = mc_result.strikes.shape[1]
        
        # Prepare data for heatmap
        residual_matrix = np.full((n_maturities, max_strikes), np.nan)
        strike_matrix = np.full((n_maturities, max_strikes), np.nan)
        
        for i in range(n_maturities):
            valid_mask = ~np.isnan(residuals[i])
            n_valid = np.sum(valid_mask)
            if n_valid > 0:
                residual_matrix[i, :n_valid] = residuals[i][valid_mask]
                strike_matrix[i, :n_valid] = mc_result.strikes[i][valid_mask]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=residual_matrix,
            x=np.arange(max_strikes),
            y=mc_result.maturities,
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title='Residual (MC - Hagan)'),
            hoverongaps=False
        ))
        
        # Update layout
        title = f"SABR Residual Heatmap: σ_MC - σ_Hagan{title_suffix}"
        fig.update_layout(
            title=title,
            xaxis_title='Strike Index',
            yaxis_title='Maturity (years)',
            width=800,
            height=600
        )
        
        if self.config.show_plots:
            fig.show()
        
        if self.config.save_plots:
            filename = f"residual_heatmap.html"
            fig.write_html(f"{self.config.output_dir}/{filename}")
        
        return fig
    
    def plot_volatility_smiles(self, mc_result: MCResult, hagan_result: HaganResult,
                             sabr_params: SABRParams, maturity_indices: List[int] = None,
                             title_suffix: str = "") -> go.Figure:
        """
        Create volatility smile plots for individual parameter sets.
        
        Args:
            mc_result: Monte Carlo simulation result
            hagan_result: Hagan analytical result
            sabr_params: SABR parameters used
            maturity_indices: List of maturity indices to plot (default: first 3)
            title_suffix: Additional text for plot title
            
        Returns:
            Plotly figure object
        """
        if maturity_indices is None:
            maturity_indices = list(range(min(3, len(mc_result.maturities))))
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, mat_idx in enumerate(maturity_indices):
            if mat_idx >= len(mc_result.maturities):
                continue
                
            maturity = mc_result.maturities[mat_idx]
            color = colors[i % len(colors)]
            
            # MC data
            mc_strikes = mc_result.strikes[mat_idx]
            mc_vols = mc_result.volatility_surface[mat_idx]
            mc_valid = ~np.isnan(mc_strikes) & ~np.isnan(mc_vols)
            
            if np.sum(mc_valid) > 0:
                # Calculate moneyness
                mc_moneyness = mc_strikes[mc_valid] / sabr_params.F0
                
                fig.add_trace(go.Scatter(
                    x=mc_moneyness,
                    y=mc_vols[mc_valid],
                    mode='markers',
                    name=f'MC T={maturity:.2f}',
                    marker=dict(color=color, symbol='circle', size=8),
                    legendgroup=f'T{mat_idx}'
                ))
            
            # Hagan data
            hagan_strikes = hagan_result.strikes[mat_idx]
            hagan_vols = hagan_result.volatility_surface[mat_idx]
            hagan_valid = ~np.isnan(hagan_strikes) & ~np.isnan(hagan_vols)
            
            if np.sum(hagan_valid) > 0:
                # Calculate moneyness
                hagan_moneyness = hagan_strikes[hagan_valid] / sabr_params.F0
                
                # Sort for smooth line
                sort_idx = np.argsort(hagan_moneyness)
                
                fig.add_trace(go.Scatter(
                    x=hagan_moneyness[sort_idx],
                    y=hagan_vols[hagan_valid][sort_idx],
                    mode='lines',
                    name=f'Hagan T={maturity:.2f}',
                    line=dict(color=color, dash='dash'),
                    legendgroup=f'T{mat_idx}'
                ))
        
        # Add vertical line at ATM
        fig.add_vline(x=1.0, line_dash="dot", line_color="gray", 
                     annotation_text="ATM", annotation_position="top")
        
        # Update layout
        title = f"SABR Volatility Smiles{title_suffix}"
        fig.update_layout(
            title=title,
            xaxis_title='Moneyness (K/F)',
            yaxis_title='Implied Volatility',
            width=900,
            height=600,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        
        if self.config.show_plots:
            fig.show()
        
        if self.config.save_plots:
            filename = f"volatility_smiles.html"
            fig.write_html(f"{self.config.output_dir}/{filename}")
        
        return fig
    
    def plot_residual_distribution(self, mc_result: MCResult, hagan_result: HaganResult,
                                 sabr_params: SABRParams, identify_wings: bool = True,
                                 title_suffix: str = "") -> go.Figure:
        """
        Create statistical distribution plots of residuals to identify wing regions.
        
        Args:
            mc_result: Monte Carlo simulation result
            hagan_result: Hagan analytical result
            sabr_params: SABR parameters used
            identify_wings: Highlight wing regions where residuals are largest
            title_suffix: Additional text for plot title
            
        Returns:
            Plotly figure object
        """
        # Calculate residuals
        residuals = self._calculate_residuals(mc_result, hagan_result)
        
        # Flatten residuals and calculate moneyness
        all_residuals = []
        all_moneyness = []
        all_maturities = []
        
        for i in range(len(mc_result.maturities)):
            valid_mask = ~np.isnan(residuals[i])
            if np.sum(valid_mask) > 0:
                strikes = mc_result.strikes[i][valid_mask]
                moneyness = strikes / sabr_params.F0
                
                all_residuals.extend(residuals[i][valid_mask])
                all_moneyness.extend(moneyness)
                all_maturities.extend([mc_result.maturities[i]] * np.sum(valid_mask))
        
        all_residuals = np.array(all_residuals)
        all_moneyness = np.array(all_moneyness)
        all_maturities = np.array(all_maturities)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Residual Distribution (Histogram)',
                'Residuals vs Moneyness',
                'Residuals vs Maturity',
                'Wing Region Analysis'
            ],
            specs=[[{'type': 'xy'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'xy'}]]
        )
        
        # 1. Histogram of residuals
        fig.add_trace(
            go.Histogram(
                x=all_residuals,
                nbinsx=50,
                name='Residual Distribution',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # 2. Residuals vs Moneyness
        fig.add_trace(
            go.Scatter(
                x=all_moneyness,
                y=all_residuals,
                mode='markers',
                name='Residuals',
                marker=dict(
                    color=all_maturities,
                    colorscale='viridis',
                    colorbar=dict(title='Maturity', x=0.45),
                    size=4,
                    opacity=0.6
                )
            ),
            row=1, col=2
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        # 3. Residuals vs Maturity
        fig.add_trace(
            go.Scatter(
                x=all_maturities,
                y=all_residuals,
                mode='markers',
                name='Residuals vs T',
                marker=dict(
                    color=all_moneyness,
                    colorscale='plasma',
                    colorbar=dict(title='Moneyness', x=0.45, y=0.3),
                    size=4,
                    opacity=0.6
                )
            ),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        # 4. Wing region analysis
        if identify_wings:
            # Define wing regions (moneyness < 0.7 or > 1.3)
            wing_mask = (all_moneyness < 0.7) | (all_moneyness > 1.3)
            atm_mask = (all_moneyness >= 0.9) & (all_moneyness <= 1.1)
            
            if np.sum(wing_mask) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=all_moneyness[wing_mask],
                        y=all_residuals[wing_mask],
                        mode='markers',
                        name='Wing Regions',
                        marker=dict(color='red', size=6, symbol='diamond')
                    ),
                    row=2, col=2
                )
            
            if np.sum(atm_mask) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=all_moneyness[atm_mask],
                        y=all_residuals[atm_mask],
                        mode='markers',
                        name='ATM Region',
                        marker=dict(color='green', size=4, symbol='circle')
                    ),
                    row=2, col=2
                )
            
            # Add wing region boundaries
            fig.add_vline(x=0.7, line_dash="dot", line_color="orange", row=2, col=2)
            fig.add_vline(x=1.3, line_dash="dot", line_color="orange", row=2, col=2)
        
        # Add zero line to wing analysis
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
        
        # Update layout
        title = f"SABR Residual Statistical Analysis{title_suffix}"
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Residual Value", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        
        fig.update_xaxes(title_text="Moneyness (K/F)", row=1, col=2)
        fig.update_yaxes(title_text="Residual", row=1, col=2)
        
        fig.update_xaxes(title_text="Maturity (years)", row=2, col=1)
        fig.update_yaxes(title_text="Residual", row=2, col=1)
        
        fig.update_xaxes(title_text="Moneyness (K/F)", row=2, col=2)
        fig.update_yaxes(title_text="Residual", row=2, col=2)
        
        if self.config.show_plots:
            fig.show()
        
        if self.config.save_plots:
            filename = f"residual_distribution.html"
            fig.write_html(f"{self.config.output_dir}/{filename}")
        
        return fig
    
    def plot_parameter_space_exploration(self, param_sets: List[SABRParams],
                                       mc_results: List[MCResult],
                                       hagan_results: List[HaganResult],
                                       title_suffix: str = "") -> go.Figure:
        """
        Create interactive plots for exploring parameter space effects.
        
        Args:
            param_sets: List of SABR parameter sets
            mc_results: List of MC results for each parameter set
            hagan_results: List of Hagan results for each parameter set
            title_suffix: Additional text for plot title
            
        Returns:
            Plotly figure object
        """
        if len(param_sets) != len(mc_results) or len(param_sets) != len(hagan_results):
            raise ValueError("Parameter sets and results must have same length")
        
        # Calculate summary statistics for each parameter set
        summary_data = []
        
        for i, (params, mc_res, hagan_res) in enumerate(zip(param_sets, mc_results, hagan_results)):
            residuals = self._calculate_residuals(mc_res, hagan_res)
            
            # Calculate statistics
            all_residuals = residuals[~np.isnan(residuals)]
            
            if len(all_residuals) > 0:
                summary_data.append({
                    'param_idx': i,
                    'alpha': params.alpha,
                    'beta': params.beta,
                    'nu': params.nu,
                    'rho': params.rho,
                    'mean_residual': np.mean(all_residuals),
                    'std_residual': np.std(all_residuals),
                    'max_abs_residual': np.max(np.abs(all_residuals)),
                    'rmse': np.sqrt(np.mean(all_residuals**2))
                })
        
        if not summary_data:
            raise ValueError("No valid residual data found")
        
        df = pd.DataFrame(summary_data)
        
        # Create subplots for parameter space exploration
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'RMSE vs Alpha-Beta',
                'Max Absolute Residual vs Nu-Rho',
                'Mean Residual vs Parameters',
                'Parameter Correlation Matrix'
            ],
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'heatmap'}]]
        )
        
        # 1. RMSE vs Alpha-Beta
        fig.add_trace(
            go.Scatter(
                x=df['alpha'],
                y=df['beta'],
                mode='markers',
                marker=dict(
                    color=df['rmse'],
                    colorscale='viridis',
                    size=10,
                    colorbar=dict(title='RMSE', x=0.45, y=0.85),
                    showscale=True
                ),
                text=[f"α={a:.3f}, β={b:.3f}<br>RMSE={r:.4f}" 
                      for a, b, r in zip(df['alpha'], df['beta'], df['rmse'])],
                hovertemplate='%{text}<extra></extra>',
                name='RMSE'
            ),
            row=1, col=1
        )
        
        # 2. Max Absolute Residual vs Nu-Rho
        fig.add_trace(
            go.Scatter(
                x=df['nu'],
                y=df['rho'],
                mode='markers',
                marker=dict(
                    color=df['max_abs_residual'],
                    colorscale='plasma',
                    size=10,
                    colorbar=dict(title='Max |Residual|', x=0.45, y=0.35),
                    showscale=True
                ),
                text=[f"ν={n:.3f}, ρ={r:.3f}<br>Max|Res|={m:.4f}" 
                      for n, r, m in zip(df['nu'], df['rho'], df['max_abs_residual'])],
                hovertemplate='%{text}<extra></extra>',
                name='Max Residual'
            ),
            row=1, col=2
        )
        
        # 3. Mean Residual vs Parameters (parallel coordinates style)
        param_cols = ['alpha', 'beta', 'nu', 'rho']
        
        # Normalize parameters for better visualization
        df_norm = df.copy()
        for col in param_cols:
            df_norm[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        # Create traces for each parameter
        colors = ['red', 'blue', 'green', 'orange']
        for i, (param, color) in enumerate(zip(param_cols, colors)):
            fig.add_trace(
                go.Scatter(
                    x=df_norm[f'{param}_norm'],
                    y=df['mean_residual'],
                    mode='markers',
                    marker=dict(color=color, size=6),
                    name=param,
                    legendgroup='params'
                ),
                row=2, col=1
            )
        
        # 4. Parameter correlation matrix
        corr_matrix = df[param_cols + ['rmse', 'max_abs_residual']].corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu_r',
                zmid=0,
                colorbar=dict(title='Correlation', x=1.0),
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ),
            row=2, col=2
        )
        
        # Update layout
        title = f"SABR Parameter Space Exploration{title_suffix}"
        fig.update_layout(
            title=title,
            height=900,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Alpha", row=1, col=1)
        fig.update_yaxes(title_text="Beta", row=1, col=1)
        
        fig.update_xaxes(title_text="Nu", row=1, col=2)
        fig.update_yaxes(title_text="Rho", row=1, col=2)
        
        fig.update_xaxes(title_text="Normalized Parameter Value", row=2, col=1)
        fig.update_yaxes(title_text="Mean Residual", row=2, col=1)
        
        if self.config.show_plots:
            fig.show()
        
        if self.config.save_plots:
            filename = f"parameter_space_exploration.html"
            fig.write_html(f"{self.config.output_dir}/{filename}")
        
        return fig
    
    def create_comprehensive_report(self, param_sets: List[SABRParams],
                                  mc_results: List[MCResult],
                                  hagan_results: List[HaganResult],
                                  save_individual: bool = True) -> Dict[str, go.Figure]:
        """
        Create comprehensive visualization report for data generation phase.
        
        Args:
            param_sets: List of SABR parameter sets
            mc_results: List of MC results
            hagan_results: List of Hagan results
            save_individual: Save individual plots as well as combined report
            
        Returns:
            Dictionary of figure objects
        """
        figures = {}
        
        print("Generating comprehensive visualization report...")
        
        # 1. Surface comparisons for first few parameter sets
        for i in range(min(3, len(param_sets))):
            print(f"Creating surface comparison {i+1}/3...")
            fig = self.plot_surface_comparison_3d(
                mc_results[i], hagan_results[i], param_sets[i],
                title_suffix=f" (Param Set {i+1})"
            )
            figures[f'surface_comparison_{i}'] = fig
        
        # 2. Volatility smiles for first parameter set
        if len(param_sets) > 0:
            print("Creating volatility smiles...")
            fig = self.plot_volatility_smiles(
                mc_results[0], hagan_results[0], param_sets[0],
                title_suffix=" (Representative Set)"
            )
            figures['volatility_smiles'] = fig
        
        # 3. Residual analysis for first parameter set
        if len(param_sets) > 0:
            print("Creating residual distribution analysis...")
            fig = self.plot_residual_distribution(
                mc_results[0], hagan_results[0], param_sets[0],
                title_suffix=" (Representative Set)"
            )
            figures['residual_distribution'] = fig
        
        # 4. Parameter space exploration (if multiple parameter sets)
        if len(param_sets) > 1:
            print("Creating parameter space exploration...")
            fig = self.plot_parameter_space_exploration(
                param_sets, mc_results, hagan_results
            )
            figures['parameter_exploration'] = fig
        
        # 5. Residual heatmaps for first parameter set
        if len(param_sets) > 0:
            print("Creating residual heatmap...")
            fig = self.plot_residual_heatmap(
                mc_results[0], hagan_results[0], param_sets[0],
                title_suffix=" (Representative Set)"
            )
            figures['residual_heatmap'] = fig
        
        print(f"Generated {len(figures)} visualization plots.")
        
        return figures
    
    def _calculate_residuals(self, mc_result: MCResult, hagan_result: HaganResult) -> np.ndarray:
        """
        Calculate residuals D(ξ) = σ_MC(ξ) - σ_Hagan(ξ).
        
        Args:
            mc_result: Monte Carlo result
            hagan_result: Hagan result
            
        Returns:
            Array of residuals with same shape as volatility surfaces
        """
        # Ensure both surfaces have same shape
        mc_shape = mc_result.volatility_surface.shape
        hagan_shape = hagan_result.volatility_surface.shape
        
        if mc_shape != hagan_shape:
            warnings.warn(f"Surface shapes differ: MC {mc_shape} vs Hagan {hagan_shape}")
            # Use minimum dimensions
            min_rows = min(mc_shape[0], hagan_shape[0])
            min_cols = min(mc_shape[1], hagan_shape[1])
            
            mc_surface = mc_result.volatility_surface[:min_rows, :min_cols]
            hagan_surface = hagan_result.volatility_surface[:min_rows, :min_cols]
        else:
            mc_surface = mc_result.volatility_surface
            hagan_surface = hagan_result.volatility_surface
        
        # Calculate residuals
        residuals = mc_surface - hagan_surface
        
        return residuals
    
    def save_summary_statistics(self, param_sets: List[SABRParams],
                              mc_results: List[MCResult],
                              hagan_results: List[HaganResult],
                              filename: str = "data_generation_summary.csv") -> pd.DataFrame:
        """
        Save summary statistics of data generation to CSV file.
        
        Args:
            param_sets: List of SABR parameter sets
            mc_results: List of MC results
            hagan_results: List of Hagan results
            filename: Output filename
            
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for i, (params, mc_res, hagan_res) in enumerate(zip(param_sets, mc_results, hagan_results)):
            residuals = self._calculate_residuals(mc_res, hagan_res)
            all_residuals = residuals[~np.isnan(residuals)]
            
            # Calculate wing region statistics
            # Approximate moneyness calculation (assuming first maturity)
            if len(mc_res.strikes) > 0:
                strikes = mc_res.strikes[0]
                valid_strikes = strikes[~np.isnan(strikes)]
                if len(valid_strikes) > 0:
                    moneyness = valid_strikes / params.F0
                    wing_mask = (moneyness < 0.7) | (moneyness > 1.3)
                    
                    if len(all_residuals) > 0 and np.sum(wing_mask) > 0:
                        wing_residuals = residuals[0][~np.isnan(residuals[0])][:len(moneyness)][wing_mask]
                        wing_rmse = np.sqrt(np.mean(wing_residuals**2)) if len(wing_residuals) > 0 else np.nan
                    else:
                        wing_rmse = np.nan
                else:
                    wing_rmse = np.nan
            else:
                wing_rmse = np.nan
            
            summary_data.append({
                'param_set_id': i,
                'alpha': params.alpha,
                'beta': params.beta,
                'nu': params.nu,
                'rho': params.rho,
                'F0': params.F0,
                'n_valid_points': len(all_residuals),
                'mean_residual': np.mean(all_residuals) if len(all_residuals) > 0 else np.nan,
                'std_residual': np.std(all_residuals) if len(all_residuals) > 0 else np.nan,
                'rmse': np.sqrt(np.mean(all_residuals**2)) if len(all_residuals) > 0 else np.nan,
                'max_abs_residual': np.max(np.abs(all_residuals)) if len(all_residuals) > 0 else np.nan,
                'wing_rmse': wing_rmse,
                'mc_computation_time': mc_res.computation_time,
                'hagan_computation_time': hagan_res.computation_time
            })
        
        df = pd.DataFrame(summary_data)
        
        # Save to CSV
        output_path = f"{self.config.output_dir}/{filename}"
        df.to_csv(output_path, index=False)
        print(f"Summary statistics saved to {output_path}")
        
        return df


# Utility functions for matplotlib-based plots (for cases where plotly is not available)

def plot_surface_matplotlib(strikes: np.ndarray, maturities: np.ndarray, 
                           volatilities: np.ndarray, title: str = "Volatility Surface",
                           save_path: str = None) -> plt.Figure:
    """
    Create 3D surface plot using matplotlib.
    
    Args:
        strikes: Strike prices (2D array)
        maturities: Maturities (2D array)
        volatilities: Volatility surface (2D array)
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(strikes, maturities, volatilities, 
                          cmap='viridis', alpha=0.8, edgecolor='none')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Implied Volatility')
    
    # Labels and title
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(title)
    
    # Improve viewing angle
    ax.view_init(elev=20, azim=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_smile_matplotlib(strikes: np.ndarray, volatilities: np.ndarray,
                         forward: float, maturity: float, title: str = "Volatility Smile",
                         save_path: str = None) -> plt.Figure:
    """
    Create volatility smile plot using matplotlib.
    
    Args:
        strikes: Strike prices
        volatilities: Implied volatilities
        forward: Forward price
        maturity: Time to maturity
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate moneyness
    moneyness = strikes / forward
    
    # Plot smile
    ax.plot(moneyness, volatilities, 'b-', linewidth=2, label=f'T={maturity:.2f}')
    ax.scatter(moneyness, volatilities, c='red', s=30, zorder=5)
    
    # Add ATM line
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='ATM')
    
    # Labels and formatting
    ax.set_xlabel('Moneyness (K/F)')
    ax.set_ylabel('Implied Volatility')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig