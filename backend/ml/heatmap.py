"""
Heatmap Generator for CrowdGuard.

This module generates 10x10 density heatmap overlays for crowd visualization.
It counts persons in grid cells, applies color mapping, and computes density metrics.

**Validates: Requirement 4.1**
"""

import numpy as np
import cv2
from typing import Dict, Tuple
from dataclasses import dataclass
from backend.ml.tracker import TrackedPerson
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Heatmap:
    """
    Represents a density heatmap with grid and overlay.
    
    Attributes:
        grid: 10x10 array of person counts per cell
        overlay: Color-mapped overlay image (H, W, 3)
        density: Ratio of occupied cells to total cells (0.0-1.0)
        density_zone: Classification of density level (LOW, MEDIUM, HIGH)
    """
    grid: np.ndarray  # (10, 10) array of person counts
    overlay: np.ndarray  # (H, W, 3) color-mapped overlay
    density: float  # Ratio of occupied cells (0.0-1.0)
    density_zone: str  # LOW, MEDIUM, HIGH


class HeatmapGenerator:
    """
    Generates 10x10 density heatmap overlays for crowd visualization.
    
    Divides video frames into a grid, counts persons in each cell based on
    centroids, applies color mapping (blue=empty, red=dense), and computes
    density metrics.
    
    **Validates: Requirement 4.1**
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (10, 10)):
        """
        Initialize heatmap generator.
        
        Args:
            grid_size: Tuple of (rows, cols) for grid dimensions (default: 10x10)
            
        **Validates: Requirement 4.1**
        """
        self.grid_size = grid_size
        self.grid_rows, self.grid_cols = grid_size
        self.total_cells = self.grid_rows * self.grid_cols
        
        logger.info(
            f"HeatmapGenerator initialized: grid_size={grid_size}, "
            f"total_cells={self.total_cells}"
        )
    
    def generate(self, 
                 tracked_persons: Dict[int, TrackedPerson],
                 frame_shape: Tuple[int, int]) -> Tuple[np.ndarray, float]:
        """
        Generate heatmap from tracked persons.
        
        Creates a 10x10 grid overlay showing crowd density distribution.
        Counts persons in each cell based on centroid positions, applies
        OpenCV COLORMAP_JET color mapping (blue=0, red=max), and computes
        density as the ratio of occupied cells.
        
        Args:
            tracked_persons: Dictionary mapping track_id to TrackedPerson
            frame_shape: Tuple of (height, width) of video frame
            
        Returns:
            Tuple of (heatmap_overlay, density_value)
            - heatmap_overlay: Color-mapped overlay image (H, W, 3) in BGR format
            - density_value: Ratio of occupied cells (0.0-1.0)
            
        **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
        """
        frame_height, frame_width = frame_shape
        
        # Initialize grid with zeros
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        # Calculate cell dimensions
        cell_height = frame_height / self.grid_rows
        cell_width = frame_width / self.grid_cols
        
        # Count persons in each grid cell
        for tracked_person in tracked_persons.values():
            cx, cy = tracked_person.centroid
            
            # Determine which cell the centroid falls into
            col = int(cx / cell_width)
            row = int(cy / cell_height)
            
            # Clamp to grid boundaries (handle edge cases)
            col = max(0, min(col, self.grid_cols - 1))
            row = max(0, min(row, self.grid_rows - 1))
            
            # Increment count for this cell
            grid[row, col] += 1
        
        # Compute density (ratio of occupied cells to total cells)
        occupied_cells = np.count_nonzero(grid)
        density = occupied_cells / self.total_cells
        
        # Create heatmap overlay
        heatmap_overlay = self._create_overlay(grid, frame_height, frame_width)
        
        logger.debug(
            f"Heatmap generated: density={density:.3f}, "
            f"occupied_cells={occupied_cells}/{self.total_cells}, "
            f"total_persons={len(tracked_persons)}"
        )
        
        return heatmap_overlay, density
    
    def _create_overlay(self, 
                       grid: np.ndarray, 
                       frame_height: int, 
                       frame_width: int) -> np.ndarray:
        """
        Create color-mapped heatmap overlay from grid.
        
        Applies OpenCV COLORMAP_JET to the grid where blue represents empty
        zones (0 persons) and red represents maximum density.
        
        Args:
            grid: 10x10 array of person counts
            frame_height: Height of video frame
            frame_width: Width of video frame
            
        Returns:
            Color-mapped overlay image (H, W, 3) in BGR format
            
        **Validates: Requirement 4.3**
        """
        # Normalize grid to 0-255 range for color mapping
        if grid.max() > 0:
            normalized_grid = (grid / grid.max() * 255).astype(np.uint8)
        else:
            normalized_grid = grid.astype(np.uint8)
        
        # Apply COLORMAP_JET (blue=0, green/yellow=medium, red=max)
        colored_grid = cv2.applyColorMap(normalized_grid, cv2.COLORMAP_JET)
        
        # Resize grid to match frame dimensions
        heatmap_overlay = cv2.resize(
            colored_grid, 
            (frame_width, frame_height), 
            interpolation=cv2.INTER_LINEAR
        )
        
        return heatmap_overlay
    
    def classify_density_zone(self, density: float) -> str:
        """
        Classify density into zones (LOW, MEDIUM, HIGH).
        
        Args:
            density: Density value (0.0-1.0)
            
        Returns:
            Density zone classification: "LOW", "MEDIUM", or "HIGH"
        """
        if density < 0.3:
            return "LOW"
        elif density <= 0.6:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def create_heatmap_object(self,
                             tracked_persons: Dict[int, TrackedPerson],
                             frame_shape: Tuple[int, int]) -> Heatmap:
        """
        Generate complete Heatmap object with all metadata.
        
        Creates a Heatmap dataclass containing the grid, overlay, density,
        and density zone classification.
        
        Args:
            tracked_persons: Dictionary mapping track_id to TrackedPerson
            frame_shape: Tuple of (height, width) of video frame
            
        Returns:
            Heatmap object with grid, overlay, density, and zone classification
            
        **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
        """
        frame_height, frame_width = frame_shape
        
        # Initialize grid
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        # Calculate cell dimensions
        cell_height = frame_height / self.grid_rows
        cell_width = frame_width / self.grid_cols
        
        # Count persons in each grid cell
        for tracked_person in tracked_persons.values():
            cx, cy = tracked_person.centroid
            
            col = int(cx / cell_width)
            row = int(cy / cell_height)
            
            col = max(0, min(col, self.grid_cols - 1))
            row = max(0, min(row, self.grid_rows - 1))
            
            grid[row, col] += 1
        
        # Compute density
        occupied_cells = np.count_nonzero(grid)
        density = occupied_cells / self.total_cells
        
        # Create overlay
        overlay = self._create_overlay(grid, frame_height, frame_width)
        
        # Classify density zone
        density_zone = self.classify_density_zone(density)
        
        heatmap = Heatmap(
            grid=grid,
            overlay=overlay,
            density=density,
            density_zone=density_zone
        )
        
        logger.debug(
            f"Heatmap object created: density={density:.3f}, zone={density_zone}"
        )
        
        return heatmap
    
    def get_grid_info(self) -> dict:
        """
        Get information about the grid configuration.
        
        Returns:
            Dictionary containing grid dimensions and total cells
        """
        return {
            "grid_size": self.grid_size,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "total_cells": self.total_cells
        }
