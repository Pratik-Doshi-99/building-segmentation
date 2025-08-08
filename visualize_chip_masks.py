#!/usr/bin/env python3
"""
Visualization script for building segmentation chip masks.
Creates 3 masked overlay visualizations from multi-channel output chips.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.plot import show
import warnings

# Suppress rasterio warnings for cleaner output
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChipMaskVisualizer:
    """Visualizes building segmentation masks overlaid on Sentinel-2 imagery."""
    
    def __init__(self):
        self.colors = ['red', 'green', 'blue']  # Colors for the 3 channels
        self.channel_names = ['Building Fractional Count', 'Building Height', 'Building Presence']
        
    def load_chip(self, chip_path: str) -> Tuple[np.ndarray, dict]:
        """
        Load a chip (either soft label or Sentinel-2).
        
        Args:
            chip_path: Path to the chip file
            
        Returns:
            Tuple of (data_array, metadata_dict)
        """
        logger.info(f"Loading chip: {chip_path}")
        
        with rasterio.open(chip_path) as src:
            data = src.read()  # Shape: (bands, height, width)
            metadata = {
                'shape': src.shape,
                'count': src.count,
                'dtype': src.dtypes[0],
                'bounds': src.bounds,
                'crs': src.crs,
                'transform': src.transform
            }
        
        logger.info(f"Loaded chip: {data.shape} shape, {src.count} bands")
        return data, metadata
    
    def normalize_sentinel_for_display(self, sentinel_data: np.ndarray) -> np.ndarray:
        """
        Normalize Sentinel-2 data for RGB display.
        
        Args:
            sentinel_data: Array of shape (bands, height, width)
            
        Returns:
            RGB array of shape (height, width, 3) normalized to [0, 1]
        """
        # Use first 3 bands for RGB (assuming bands 1-3 are B, G, R or similar)
        rgb_bands = sentinel_data[:3]  # Shape: (3, height, width)
        
        # Transpose to (height, width, 3) for matplotlib
        rgb = np.transpose(rgb_bands, (1, 2, 0))
        
        # Normalize using percentile stretch (common for satellite imagery)
        # This handles the wide dynamic range of Sentinel-2 data
        rgb_normalized = np.zeros_like(rgb, dtype=np.float32)
        
        for i in range(3):
            band = rgb[:, :, i]
            # Remove zeros (likely NoData) from percentile calculation
            valid_pixels = band[band > 0]
            
            if len(valid_pixels) > 0:
                p2, p98 = np.percentile(valid_pixels, (2, 98))
                # Stretch to 0-1 range
                band_stretched = np.clip((band - p2) / (p98 - p2), 0, 1)
                rgb_normalized[:, :, i] = band_stretched
            else:
                rgb_normalized[:, :, i] = 0
        
        return rgb_normalized
    
    def create_colored_mask(self, mask_channel: np.ndarray, color: str, alpha: float = 0.6) -> np.ndarray:
        """
        Create a colored mask from a single channel.
        
        Args:
            mask_channel: 2D array representing mask values
            color: Color name ('red', 'green', 'blue')
            alpha: Transparency level
            
        Returns:
            RGBA array of shape (height, width, 4)
        """
        h, w = mask_channel.shape
        colored_mask = np.zeros((h, w, 4), dtype=np.float32)
        
        # Normalize mask values to 0-1 range
        mask_normalized = (mask_channel - mask_channel.min()) / (mask_channel.max() - mask_channel.min() + 1e-8)
        
        # Set color channel
        if color == 'red':
            colored_mask[:, :, 0] = mask_normalized
        elif color == 'green':
            colored_mask[:, :, 1] = mask_normalized
        elif color == 'blue':
            colored_mask[:, :, 2] = mask_normalized
        
        # Set alpha channel (transparency based on mask intensity)
        colored_mask[:, :, 3] = mask_normalized * alpha
        
        return colored_mask
    
    def create_overlay_visualization(self, 
                                   sentinel_rgb: np.ndarray, 
                                   mask_data: np.ndarray,
                                   output_path: str,
                                   title_prefix: str = "Building Segmentation") -> None:
        """
        Create and save overlay visualization with 6 subplots in 2x3 grid.
        Top row: 3 building channels overlaid on Sentinel-2
        Bottom row: 3 heatmaps of the channels only (concentration maps)
        
        Args:
            sentinel_rgb: RGB Sentinel-2 data, shape (height, width, 3)
            mask_data: Mask data, shape (3, height, width) for 3 channels
            output_path: Path to save the output image
            title_prefix: Prefix for the plot title
        """
        logger.info("Creating overlay and heatmap visualization")
        
        # Create figure with 2x3 subplots for high resolution display
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{title_prefix} - Overlays and Concentration Heatmaps', 
                     fontsize=20, fontweight='bold', y=0.95)
        
        # Define colormaps for heatmaps
        heatmap_cmaps = ['Reds', 'Greens', 'Blues']
        
        for i in range(3):
            # Top row: Overlays on Sentinel-2
            ax_overlay = axes[0, i]
            
            # Display base Sentinel-2 image
            ax_overlay.imshow(sentinel_rgb)
            
            # Create colored mask for this channel
            mask_channel = mask_data[i]  # Shape: (height, width)
            colored_mask = self.create_colored_mask(mask_channel, self.colors[i], alpha=0.7)
            
            # Overlay the colored mask
            ax_overlay.imshow(colored_mask, alpha=0.7)
            
            # Set title and formatting for overlay
            ax_overlay.set_title(f'{self.channel_names[i]}\n({self.colors[i].capitalize()} Overlay)', 
                               fontweight='bold', color=self.colors[i], fontsize=14)
            ax_overlay.set_xlabel('X (pixels)', fontsize=12)
            ax_overlay.set_ylabel('Y (pixels)', fontsize=12)
            ax_overlay.grid(True, alpha=0.3)
            
            # Bottom row: Pure heatmaps
            ax_heatmap = axes[1, i]
            
            # Create heatmap with appropriate colormap
            im = ax_heatmap.imshow(mask_channel, cmap=heatmap_cmaps[i], 
                                  vmin=np.min(mask_channel), vmax=np.max(mask_channel))
            
            # Add colorbar for heatmap
            cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8, aspect=20)
            cbar.set_label('Intensity', fontsize=12)
            
            # Set title and formatting for heatmap
            ax_heatmap.set_title(f'{self.channel_names[i]}\nConcentration Heatmap', 
                               fontweight='bold', fontsize=14)
            ax_heatmap.set_xlabel('X (pixels)', fontsize=12)
            ax_heatmap.set_ylabel('Y (pixels)', fontsize=12)
            
            # Add comprehensive statistics for both plots
            channel_min = np.min(mask_channel)
            channel_max = np.max(mask_channel)
            channel_mean = np.mean(mask_channel)
            channel_std = np.std(mask_channel)
            channel_nonzero = np.count_nonzero(mask_channel)
            channel_total = mask_channel.size
            coverage_pct = (channel_nonzero / channel_total) * 100
            
            # Stats for overlay plot
            overlay_stats_text = (f'Min: {channel_min:.3f}\n'
                                 f'Max: {channel_max:.3f}\n'
                                 f'Mean: {channel_mean:.3f}\n'
                                 f'Coverage: {coverage_pct:.1f}%')
            
            ax_overlay.text(0.02, 0.98, overlay_stats_text, transform=ax_overlay.transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                          fontsize=10, fontfamily='monospace')
            
            # Stats for heatmap
            heatmap_stats_text = (f'μ: {channel_mean:.3f}\n'
                                 f'σ: {channel_std:.3f}\n'
                                 f'Range: [{channel_min:.3f}, {channel_max:.3f}]\n'
                                 f'Non-zero: {channel_nonzero:,}')
            
            ax_heatmap.text(0.02, 0.98, heatmap_stats_text, transform=ax_heatmap.transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                          fontsize=10, fontfamily='monospace')
        
        # Adjust layout for better spacing
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        # Save with very high resolution
        plt.savefig(output_path, dpi=450, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png', pil_kwargs={'optimize': True})
        logger.info(f"Saved high-resolution visualization to: {output_path}")
        plt.close()
    
    def process_chip_pair(self, 
                         soft_label_path: str, 
                         sentinel_path: str, 
                         output_path: str) -> None:
        """
        Process a pair of soft label and Sentinel-2 chips to create overlay visualization.
        
        Args:
            soft_label_path: Path to soft label chip
            sentinel_path: Path to Sentinel-2 chip
            output_path: Path to save the output visualization
        """
        # Load soft label chip (this contains the 3-channel mask)
        mask_data, mask_metadata = self.load_chip(soft_label_path)
        
        if mask_data.shape[0] != 3:
            raise ValueError(f"Expected 3 channels in soft label, got {mask_data.shape[0]}")
        
        # Load Sentinel-2 chip
        sentinel_data, sentinel_metadata = self.load_chip(sentinel_path)
        
        if sentinel_data.shape[0] < 3:
            raise ValueError(f"Expected at least 3 bands in Sentinel-2, got {sentinel_data.shape[0]}")
        
        # Check spatial alignment
        if mask_data.shape[1:] != sentinel_data.shape[1:]:
            logger.warning(f"Spatial dimension mismatch: mask {mask_data.shape[1:]} vs sentinel {sentinel_data.shape[1:]}")
            # You could add resampling here if needed
        
        # Normalize Sentinel-2 for display
        sentinel_rgb = self.normalize_sentinel_for_display(sentinel_data)
        
        # Extract chip identifier from filename
        chip_id = Path(soft_label_path).stem.split('_')[-1]
        title_prefix = f"Chip {chip_id}"
        
        # Create overlay visualization
        self.create_overlay_visualization(
            sentinel_rgb=sentinel_rgb,
            mask_data=mask_data,
            output_path=output_path,
            title_prefix=title_prefix
        )
    
    def auto_find_chip_pair(self, chip_id: str, base_dir: str = ".") -> Tuple[Optional[str], Optional[str]]:
        """
        Automatically find corresponding soft label and Sentinel-2 chips.
        
        Args:
            chip_id: Chip identifier (e.g., 'tile_YV3xUniiUOc_0001')
            base_dir: Base directory to search in
            
        Returns:
            Tuple of (soft_label_path, sentinel_path) or (None, None) if not found
        """
        base_path = Path(base_dir)
        
        # Look for soft label chip
        soft_label_pattern = f"chip_downsampled_{chip_id}.tif"
        soft_label_paths = list(base_path.rglob(soft_label_pattern))
        
        # Look for Sentinel-2 chip
        sentinel_pattern = f"sentinel2_chip_downsampled_{chip_id}.tif"
        sentinel_paths = list(base_path.rglob(sentinel_pattern))
        
        soft_label_path = str(soft_label_paths[0]) if soft_label_paths else None
        sentinel_path = str(sentinel_paths[0]) if sentinel_paths else None
        
        if soft_label_path:
            logger.info(f"Found soft label chip: {soft_label_path}")
        else:
            logger.warning(f"Soft label chip not found for: {chip_id}")
            
        if sentinel_path:
            logger.info(f"Found Sentinel-2 chip: {sentinel_path}")
        else:
            logger.warning(f"Sentinel-2 chip not found for: {chip_id}")
        
        return soft_label_path, sentinel_path

def main():
    parser = argparse.ArgumentParser(description="Visualize building segmentation chip masks")
    parser.add_argument('--chip-id', required=True, help='Chip identifier (e.g., tile_YV3xUniiUOc_0001)')
    parser.add_argument('--soft-label-path', help='Path to soft label chip (auto-detected if not provided)')
    parser.add_argument('--sentinel-path', help='Path to Sentinel-2 chip (auto-detected if not provided)')
    parser.add_argument('--output', help='Output image path (default: chip_visualization_{chip_id}.png)')
    parser.add_argument('--base-dir', default='.', help='Base directory for auto-detection (default: current directory)')
    
    args = parser.parse_args()
    
    visualizer = ChipMaskVisualizer()
    
    # Auto-detect paths if not provided
    if not args.soft_label_path or not args.sentinel_path:
        logger.info(f"Auto-detecting chip paths for: {args.chip_id}")
        auto_soft, auto_sentinel = visualizer.auto_find_chip_pair(args.chip_id, args.base_dir)
        
        soft_label_path = args.soft_label_path or auto_soft
        sentinel_path = args.sentinel_path or auto_sentinel
    else:
        soft_label_path = args.soft_label_path
        sentinel_path = args.sentinel_path
    
    # Check that both paths are found
    if not soft_label_path or not sentinel_path:
        logger.error("Could not find both soft label and Sentinel-2 chips")
        return 1
    
    # Set output path
    output_path = args.output or f"chip_visualization_{args.chip_id}.png"
    
    try:
        # Process the chip pair
        visualizer.process_chip_pair(
            soft_label_path=soft_label_path,
            sentinel_path=sentinel_path,
            output_path=output_path
        )
        
        logger.info("Visualization completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error processing chip pair: {e}")
        return 1

if __name__ == "__main__":
    exit(main())