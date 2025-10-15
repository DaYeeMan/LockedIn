"""
Patch extraction module for MDA-CNN volatility surface modeling.

This module implements the PatchExtractor class that extracts local surface patches
around high-fidelity points for CNN input. It handles grid alignment, boundary
conditions, and different patch sizes.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union
from dataclasses import dataclass
import warnings

from data_generation.sabr_params import SABRParams, GridConfig


@dataclass
class PatchConfig:
    """
    Configuration for patch extraction.
    
    Attributes:
        patch_size: (height, width) of patches in grid points
        boundary_mode: How to handle boundaries ('pad', 'reflect', 'wrap', 'constant')
        pad_value: Value to use for 'constant' boundary mode
        normalize_patches: Whether to normalize patches locally
        center_on_hf_point: Whether to center patches exactly on HF points
    """
    patch_size: Tuple[int, int] = (9, 9)
    boundary_mode: str = 'reflect'
    pad_value: float = 0.0
    normalize_patches: bool = True
    center_on_hf_point: bool = True
    
    def __post_init__(self):
        """Validate patch configuration."""
        if self.patch_size[0] <= 0 or self.patch_size[1] <= 0:
            raise ValueError("Patch size must be positive")
        
        if self.patch_size[0] % 2 == 0 or self.patch_size[1] % 2 == 0:
            warnings.warn("Even patch sizes may not center properly on HF points")
        
        valid_modes = ['pad', 'reflect', 'wrap', 'constant']
        if self.boundary_mode not in valid_modes:
            raise ValueError(f"boundary_mode must be one of {valid_modes}")


@dataclass
class HFPoint:
    """
    High-fidelity point specification.
    
    Attributes:
        strike: Strike price
        maturity: Time to maturity
        volatility: High-fidelity volatility value
        grid_coords: (i, j) coordinates in the LF grid (if aligned)
        patch_coords: Patch boundaries in grid coordinates
    """
    strike: float
    maturity: float
    volatility: float
    grid_coords: Optional[Tuple[int, int]] = None
    patch_coords: Optional[Tuple[slice, slice]] = None


@dataclass
class ExtractedPatch:
    """
    Extracted patch data.
    
    Attributes:
        patch: 2D array containing the patch data
        hf_point: Associated high-fidelity point
        center_coords: Center coordinates in original grid
        boundary_flags: Flags indicating which boundaries were padded
        normalization_stats: Statistics used for normalization (if applied)
    """
    patch: np.ndarray
    hf_point: HFPoint
    center_coords: Tuple[int, int]
    boundary_flags: Dict[str, bool]
    normalization_stats: Optional[Dict[str, float]] = None


class PatchExtractor:
    """
    Extract local surface patches around high-fidelity points.
    
    This class handles the extraction of local patches from low-fidelity surfaces
    around high-fidelity points for CNN input. It includes grid alignment logic,
    boundary handling, and patch normalization.
    """
    
    def __init__(self, config: PatchConfig = None):
        """
        Initialize patch extractor.
        
        Args:
            config: Patch extraction configuration
        """
        self.config = config or PatchConfig()
        self.grid_info = None
        
    def align_hf_to_grid(self, hf_points: List[HFPoint], 
                        strikes_grid: np.ndarray, 
                        maturities_grid: np.ndarray) -> List[HFPoint]:
        """
        Align high-fidelity points to low-fidelity grid coordinates.
        
        This method finds the closest grid points to each HF point and updates
        the HF point objects with grid coordinates.
        
        Args:
            hf_points: List of high-fidelity points
            strikes_grid: 2D array of strike values (shape: n_maturities x n_strikes)
            maturities_grid: 1D array of maturity values
            
        Returns:
            List of HF points with updated grid coordinates
        """
        aligned_points = []
        
        for hf_point in hf_points:
            # Find closest maturity
            maturity_idx = np.argmin(np.abs(maturities_grid - hf_point.maturity))
            
            # Get strikes for this maturity (handle variable strike grids)
            strikes_for_maturity = strikes_grid[maturity_idx]
            valid_strikes = strikes_for_maturity[~np.isnan(strikes_for_maturity)]
            
            if len(valid_strikes) == 0:
                warnings.warn(f"No valid strikes for maturity {hf_point.maturity}")
                continue
            
            # Find closest strike
            strike_idx = np.argmin(np.abs(valid_strikes - hf_point.strike))
            
            # Create aligned HF point
            aligned_point = HFPoint(
                strike=hf_point.strike,
                maturity=hf_point.maturity,
                volatility=hf_point.volatility,
                grid_coords=(maturity_idx, strike_idx)
            )
            
            aligned_points.append(aligned_point)
        
        return aligned_points
    
    def extract_patch(self, surface: np.ndarray, center_coords: Tuple[int, int]) -> ExtractedPatch:
        """
        Extract a single patch from the surface around given coordinates.
        
        Args:
            surface: 2D surface array (n_maturities x n_strikes)
            center_coords: (maturity_idx, strike_idx) center coordinates
            
        Returns:
            ExtractedPatch object containing patch data and metadata
        """
        patch_height, patch_width = self.config.patch_size
        center_i, center_j = center_coords
        
        # Calculate patch boundaries
        half_height = patch_height // 2
        half_width = patch_width // 2
        
        # Desired patch boundaries (may extend outside surface)
        i_start = center_i - half_height
        i_end = center_i + half_height + 1
        j_start = center_j - half_width
        j_end = center_j + half_width + 1
        
        # Surface dimensions
        surface_height, surface_width = surface.shape
        
        # Track boundary conditions
        boundary_flags = {
            'top': i_start < 0,
            'bottom': i_end > surface_height,
            'left': j_start < 0,
            'right': j_end > surface_width
        }
        
        # Extract patch with boundary handling
        patch = self._extract_with_boundaries(
            surface, i_start, i_end, j_start, j_end, boundary_flags
        )
        
        # Apply normalization if requested
        normalization_stats = None
        if self.config.normalize_patches:
            patch, normalization_stats = self._normalize_patch(patch)
        
        # Create dummy HF point for this extraction
        dummy_hf_point = HFPoint(
            strike=0.0,  # Will be updated by caller
            maturity=0.0,  # Will be updated by caller
            volatility=0.0,  # Will be updated by caller
            grid_coords=center_coords
        )
        
        return ExtractedPatch(
            patch=patch,
            hf_point=dummy_hf_point,
            center_coords=center_coords,
            boundary_flags=boundary_flags,
            normalization_stats=normalization_stats
        )
    
    def extract_patches_for_hf_points(self, surface: np.ndarray,
                                    hf_points: List[HFPoint],
                                    strikes_grid: np.ndarray,
                                    maturities_grid: np.ndarray) -> List[ExtractedPatch]:
        """
        Extract patches for all high-fidelity points.
        
        Args:
            surface: 2D LF surface array
            hf_points: List of high-fidelity points
            strikes_grid: 2D grid of strikes
            maturities_grid: 1D array of maturities
            
        Returns:
            List of extracted patches
        """
        # First align HF points to grid
        aligned_points = self.align_hf_to_grid(hf_points, strikes_grid, maturities_grid)
        
        extracted_patches = []
        
        for hf_point in aligned_points:
            if hf_point.grid_coords is None:
                warnings.warn(f"Skipping HF point without grid coordinates: {hf_point}")
                continue
            
            # Extract patch
            patch_result = self.extract_patch(surface, hf_point.grid_coords)
            
            # Update with correct HF point information
            patch_result.hf_point = hf_point
            
            extracted_patches.append(patch_result)
        
        return extracted_patches
    
    def _extract_with_boundaries(self, surface: np.ndarray, 
                               i_start: int, i_end: int,
                               j_start: int, j_end: int,
                               boundary_flags: Dict[str, bool]) -> np.ndarray:
        """
        Extract patch with boundary handling.
        
        Args:
            surface: Source surface
            i_start, i_end: Row boundaries (may be outside surface)
            j_start, j_end: Column boundaries (may be outside surface)
            boundary_flags: Which boundaries are outside surface
            
        Returns:
            Extracted patch with boundary handling applied
        """
        surface_height, surface_width = surface.shape
        patch_height = i_end - i_start
        patch_width = j_end - j_start
        
        if self.config.boundary_mode == 'constant':
            # Initialize patch with constant value
            patch = np.full((patch_height, patch_width), self.config.pad_value)
            
            # Calculate overlap region
            overlap_i_start = max(0, i_start)
            overlap_i_end = min(surface_height, i_end)
            overlap_j_start = max(0, j_start)
            overlap_j_end = min(surface_width, j_end)
            
            # Calculate patch indices for overlap
            patch_i_start = overlap_i_start - i_start
            patch_i_end = patch_i_start + (overlap_i_end - overlap_i_start)
            patch_j_start = overlap_j_start - j_start
            patch_j_end = patch_j_start + (overlap_j_end - overlap_j_start)
            
            # Copy overlapping region
            if overlap_i_end > overlap_i_start and overlap_j_end > overlap_j_start:
                patch[patch_i_start:patch_i_end, patch_j_start:patch_j_end] = \
                    surface[overlap_i_start:overlap_i_end, overlap_j_start:overlap_j_end]
        
        elif self.config.boundary_mode == 'reflect':
            # Use numpy's pad function with reflect mode
            # First extract the valid region
            valid_i_start = max(0, i_start)
            valid_i_end = min(surface_height, i_end)
            valid_j_start = max(0, j_start)
            valid_j_end = min(surface_width, j_end)
            
            # Calculate padding needed
            pad_top = max(0, -i_start)
            pad_bottom = max(0, i_end - surface_height)
            pad_left = max(0, -j_start)
            pad_right = max(0, j_end - surface_width)
            
            # Extract valid region
            if valid_i_end > valid_i_start and valid_j_end > valid_j_start:
                valid_patch = surface[valid_i_start:valid_i_end, valid_j_start:valid_j_end]
                
                # Apply padding
                if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                    patch = np.pad(valid_patch, 
                                 ((pad_top, pad_bottom), (pad_left, pad_right)),
                                 mode='reflect')
                else:
                    patch = valid_patch
            else:
                # Fallback to constant padding if no valid region
                patch = np.full((patch_height, patch_width), self.config.pad_value)
        
        elif self.config.boundary_mode == 'wrap':
            # Periodic boundary conditions
            patch = np.zeros((patch_height, patch_width))
            
            for i in range(patch_height):
                for j in range(patch_width):
                    surface_i = (i_start + i) % surface_height
                    surface_j = (j_start + j) % surface_width
                    patch[i, j] = surface[surface_i, surface_j]
        
        else:  # 'pad' mode - extend edge values
            # Similar to constant but use edge values
            patch = np.zeros((patch_height, patch_width))
            
            for i in range(patch_height):
                for j in range(patch_width):
                    surface_i = np.clip(i_start + i, 0, surface_height - 1)
                    surface_j = np.clip(j_start + j, 0, surface_width - 1)
                    patch[i, j] = surface[surface_i, surface_j]
        
        return patch
    
    def _normalize_patch(self, patch: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Normalize patch locally.
        
        Args:
            patch: Input patch
            
        Returns:
            Tuple of (normalized_patch, normalization_stats)
        """
        # Calculate statistics excluding NaN values
        valid_values = patch[~np.isnan(patch)]
        
        if len(valid_values) == 0:
            # All NaN patch
            return patch, {'mean': 0.0, 'std': 1.0, 'valid_count': 0}
        
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)
        
        # Avoid division by zero
        if std_val < 1e-12:
            std_val = 1.0
        
        # Normalize
        normalized_patch = (patch - mean_val) / std_val
        
        stats = {
            'mean': float(mean_val),
            'std': float(std_val),
            'valid_count': len(valid_values)
        }
        
        return normalized_patch, stats
    
    def batch_extract_patches(self, surfaces: List[np.ndarray],
                            hf_points_list: List[List[HFPoint]],
                            strikes_grids: List[np.ndarray],
                            maturities_grids: List[np.ndarray]) -> List[List[ExtractedPatch]]:
        """
        Extract patches for multiple surfaces in batch.
        
        Args:
            surfaces: List of 2D surface arrays
            hf_points_list: List of HF points for each surface
            strikes_grids: List of strike grids for each surface
            maturities_grids: List of maturity grids for each surface
            
        Returns:
            List of lists of extracted patches (one list per surface)
        """
        if not (len(surfaces) == len(hf_points_list) == len(strikes_grids) == len(maturities_grids)):
            raise ValueError("All input lists must have the same length")
        
        all_patches = []
        
        for surface, hf_points, strikes_grid, maturities_grid in zip(
            surfaces, hf_points_list, strikes_grids, maturities_grids
        ):
            patches = self.extract_patches_for_hf_points(
                surface, hf_points, strikes_grid, maturities_grid
            )
            all_patches.append(patches)
        
        return all_patches
    
    def validate_patch_extraction(self, patches: List[ExtractedPatch]) -> Dict[str, Any]:
        """
        Validate extracted patches for quality and consistency.
        
        Args:
            patches: List of extracted patches
            
        Returns:
            Dictionary with validation results
        """
        if not patches:
            return {'valid': False, 'error': 'No patches provided'}
        
        validation_results = {
            'valid': True,
            'patch_count': len(patches),
            'size_consistency': True,
            'boundary_statistics': {},
            'normalization_statistics': {},
            'warnings': []
        }
        
        # Check size consistency
        expected_size = self.config.patch_size
        for i, patch in enumerate(patches):
            if patch.patch.shape != expected_size:
                validation_results['size_consistency'] = False
                validation_results['warnings'].append(
                    f"Patch {i} has size {patch.patch.shape}, expected {expected_size}"
                )
        
        # Boundary statistics
        boundary_counts = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        for patch in patches:
            for boundary, flag in patch.boundary_flags.items():
                if flag:
                    boundary_counts[boundary] += 1
        
        validation_results['boundary_statistics'] = boundary_counts
        
        # Normalization statistics (if applicable)
        if self.config.normalize_patches:
            means = []
            stds = []
            valid_counts = []
            
            for patch in patches:
                if patch.normalization_stats:
                    means.append(patch.normalization_stats['mean'])
                    stds.append(patch.normalization_stats['std'])
                    valid_counts.append(patch.normalization_stats['valid_count'])
            
            if means:
                validation_results['normalization_statistics'] = {
                    'mean_range': (min(means), max(means)),
                    'std_range': (min(stds), max(stds)),
                    'avg_valid_count': np.mean(valid_counts),
                    'min_valid_count': min(valid_counts)
                }
                
                # Check for patches with too few valid points
                min_valid_threshold = expected_size[0] * expected_size[1] * 0.5
                low_valid_patches = sum(1 for count in valid_counts if count < min_valid_threshold)
                
                if low_valid_patches > 0:
                    validation_results['warnings'].append(
                        f"{low_valid_patches} patches have fewer than 50% valid points"
                    )
        
        # Check for NaN patches
        nan_patches = 0
        for patch in patches:
            if np.all(np.isnan(patch.patch)):
                nan_patches += 1
        
        if nan_patches > 0:
            validation_results['warnings'].append(f"{nan_patches} patches are entirely NaN")
            if nan_patches == len(patches):
                validation_results['valid'] = False
        
        return validation_results
    
    def get_patch_statistics(self, patches: List[ExtractedPatch]) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for extracted patches.
        
        Args:
            patches: List of extracted patches
            
        Returns:
            Dictionary with patch statistics
        """
        if not patches:
            return {'error': 'No patches provided'}
        
        # Collect all patch data
        all_patch_data = []
        boundary_usage = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        
        for patch in patches:
            valid_data = patch.patch[~np.isnan(patch.patch)]
            if len(valid_data) > 0:
                all_patch_data.extend(valid_data)
            
            # Count boundary usage
            for boundary, used in patch.boundary_flags.items():
                if used:
                    boundary_usage[boundary] += 1
        
        if not all_patch_data:
            return {'error': 'No valid patch data found'}
        
        all_patch_data = np.array(all_patch_data)
        
        statistics = {
            'total_patches': len(patches),
            'total_valid_points': len(all_patch_data),
            'value_statistics': {
                'mean': float(np.mean(all_patch_data)),
                'std': float(np.std(all_patch_data)),
                'min': float(np.min(all_patch_data)),
                'max': float(np.max(all_patch_data)),
                'median': float(np.median(all_patch_data)),
                'percentiles': {
                    '5th': float(np.percentile(all_patch_data, 5)),
                    '25th': float(np.percentile(all_patch_data, 25)),
                    '75th': float(np.percentile(all_patch_data, 75)),
                    '95th': float(np.percentile(all_patch_data, 95))
                }
            },
            'boundary_usage': boundary_usage,
            'boundary_usage_percentage': {
                k: (v / len(patches)) * 100 for k, v in boundary_usage.items()
            }
        }
        
        return statistics