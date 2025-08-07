#!/usr/bin/env python3
"""
Script 2: Process soft label tiles and download corresponding Sentinel-2 imagery
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
from datetime import datetime, timedelta

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS
import ee
from pyproj import Transformer
import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TileInfo:
    """Information about a tile"""
    filepath: Path
    bounds: rasterio.coords.BoundingBox
    crs: CRS
    shape: Tuple[int, int]
    transform: rasterio.Affine
    date: Optional[datetime] = None


@dataclass
class ChipInfo:
    """Information about a chip extracted from a tile"""
    tile_path: Path
    chip_idx: int
    bounds: rasterio.coords.BoundingBox
    crs: CRS
    shape: Tuple[int, int] = (256, 256)
    gsd: float = 10.0  # meters


class Sentinel2Downloader:
    """Handle Sentinel-2 data downloads from Google Earth Engine"""
    
    def __init__(self):
        """Initialize Earth Engine"""
        try:
            ee.Initialize()
            logger.info("Earth Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Earth Engine: {e}")
            logger.error("Run 'earthengine authenticate' first")
            sys.exit(1)
    
    def get_sentinel2_image(
        self, 
        bounds: Dict, 
        date: datetime,
        date_range_days: int = 30,
        cloud_threshold: int = 20
    ) -> Optional[ee.Image]:
        """
        Get Sentinel-2 image for given bounds and date range.
        
        Args:
            bounds: Dictionary with 'coordinates' key containing polygon coords
            date: Target date
            date_range_days: Days to search before/after target date
            cloud_threshold: Maximum cloud percentage
        
        Returns:
            Earth Engine Image or None
        """
        # Create date range
        start_date = (date - timedelta(days=date_range_days)).strftime('%Y-%m-%d')
        end_date = (date + timedelta(days=date_range_days)).strftime('%Y-%m-%d')
        
        # Create geometry
        geometry = ee.Geometry.Polygon(bounds['coordinates'])
        
        # Get Sentinel-2 collection
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(geometry)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
                     .select(['B2', 'B3', 'B4', 'B8']))  # Blue, Green, Red, NIR
        
        # Get median composite
        image = collection.median().clip(geometry)
        
        return image
    
    def download_chip(
        self,
        chip: ChipInfo,
        output_dir: Path,
        date: Optional[datetime] = None
    ) -> Optional[Path]:
        """
        Download Sentinel-2 data for a specific chip.
        
        Args:
            chip: Chip information
            output_dir: Output directory
            date: Date for temporal matching
        
        Returns:
            Path to downloaded file or None
        """
        # Convert bounds to lat/lon if needed
        if chip.crs != CRS.from_epsg(4326):
            # Transform bounds to WGS84
            
            transformer = Transformer.from_crs(chip.crs, CRS.from_epsg(4326), always_xy=True)
            
            # Get corner coordinates
            corners = [
                (chip.bounds.left, chip.bounds.bottom),
                (chip.bounds.right, chip.bounds.bottom),
                (chip.bounds.right, chip.bounds.top),
                (chip.bounds.left, chip.bounds.top),
                (chip.bounds.left, chip.bounds.bottom)  # Close polygon
            ]
            
            # Transform to lat/lon
            transformed_corners = [transformer.transform(x, y) for x, y in corners]
            bounds = {
                'type': 'Polygon',
                'coordinates': [transformed_corners]
            }
        else:
            bounds = {
                'type': 'Polygon',
                'coordinates': [[
                    [chip.bounds.left, chip.bounds.bottom],
                    [chip.bounds.right, chip.bounds.bottom],
                    [chip.bounds.right, chip.bounds.top],
                    [chip.bounds.left, chip.bounds.top],
                    [chip.bounds.left, chip.bounds.bottom]
                ]]
            }
        
        # Use current date if not specified
        if date is None:
            date = datetime.now()
        
        # Get Sentinel-2 image
        image = self.get_sentinel2_image(bounds, date)
        
        if image is None:
            logger.warning(f"No Sentinel-2 data found for chip {chip.chip_idx}")
            return None
        
        # Define export parameters
        output_filename = f"sentinel2_chip_{chip.tile_path.stem}_{chip.chip_idx:04d}.tif"
        output_path = output_dir / output_filename
        
        # Skip if exists
        if output_path.exists():
            logger.info(f"Chip already exists: {output_filename}")
            return output_path
        
        try:
            # Export to Drive first (EE limitation), then download
            # For production, consider using ee.batch.Export.image.toCloudStorage
            
            # Get URL for download
            logger.debug('Downloading...')
            url = image.getDownloadUrl({
                'scale': chip.gsd,
                'crs': f'EPSG:{chip.crs.to_epsg()}',
                'region': bounds,
                'format': 'GEO_TIFF'
            })
            logger.debug(f'Download URL: {url}')
            response = requests.get(url)
            response.raise_for_status()
            
            # Save file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded Sentinel-2 chip: {output_filename}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to download chip {chip.chip_idx}: {e}")
            return None


def extract_date_from_filename(filepath: Path) -> Optional[datetime]:
    """
    Extract date from filename like 'tile_YV3xUniiUOc.tif' or '3068c_2023_06_30'.
    """
    # Try to extract date from parent directory first
    parent_name = filepath.parent.name
    parts = parent_name.split('_')
    if len(parts) >= 4:
        try:
            year, month, day = parts[-3], parts[-2], parts[-1]
            return datetime(int(year), int(month), int(day))
        except (ValueError, IndexError):
            pass
    
    # Default to None if no date found
    return None


def downsample_tile(
    input_path: Path,
    output_path: Path,
    target_gsd: float = 10.0,
    original_gsd: float = 0.5
) -> TileInfo:
    """
    Downsample a tile from 0.5m to 10m resolution.
    
    Args:
        input_path: Path to input tile
        output_path: Path to save downsampled tile
        target_gsd: Target ground sampling distance (meters)
        original_gsd: Original ground sampling distance (meters)
    
    Returns:
        TileInfo for the downsampled tile
    """
    scale_factor = int(target_gsd / original_gsd)  # 20
    
    with rasterio.open(input_path) as src:
        # Calculate new dimensions
        new_height = src.height // scale_factor
        new_width = src.width // scale_factor
        
        # Update transform for new resolution
        transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )
        
        # Read and downsample data
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.average
        )
        
        # Update metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'height': new_height,
            'width': new_width,
            'transform': transform
        })
        
        # Write downsampled tile
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            dst.write(data)
        
        # Calculate bounds for the downsampled tile
        bounds = rasterio.coords.BoundingBox(
            left=transform.c,
            bottom=transform.f + transform.e * new_height,
            right=transform.c + transform.a * new_width,
            top=transform.f
        )
        
        return TileInfo(
            filepath=output_path,
            bounds=bounds,
            crs=src.crs,
            shape=(new_height, new_width),
            transform=transform,
            date=extract_date_from_filename(input_path)
        )


def create_chips_from_tile(
    tile_info: TileInfo,
    chip_size: int = 256,
    overlap: int = 0
) -> List[ChipInfo]:
    """
    Create chip definitions from a downsampled tile.
    
    Args:
        tile_info: Information about the tile
        chip_size: Size of each chip in pixels
        overlap: Overlap between chips in pixels
    
    Returns:
        List of ChipInfo objects
    """
    chips = []
    height, width = tile_info.shape
    
    # Calculate stride
    stride = chip_size - overlap
    
    # Calculate number of chips (with proper handling of remainders)
    n_rows = (height - overlap) // stride
    n_cols = (width - overlap) // stride
    
    chip_idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            # Calculate pixel coordinates
            row_start = row * stride
            col_start = col * stride
            
            # Ensure we don't exceed bounds
            row_end = min(row_start + chip_size, height)
            col_end = min(col_start + chip_size, width)
            
            # Skip if chip is too small (edge case)
            if (row_end - row_start) < chip_size * 0.9 or \
               (col_end - col_start) < chip_size * 0.9:
                continue
            
            # Convert pixel coordinates to geographic coordinates
            x_min, y_max = tile_info.transform * (col_start, row_start)
            x_max, y_min = tile_info.transform * (col_end, row_end)
            
            chip_bounds = rasterio.coords.BoundingBox(
                left=x_min,
                bottom=y_min,
                right=x_max,
                top=y_max
            )
            
            chips.append(ChipInfo(
                tile_path=tile_info.filepath,
                chip_idx=chip_idx,
                bounds=chip_bounds,
                crs=tile_info.crs,
                shape=(chip_size, chip_size),
                gsd=10.0
            ))
            
            chip_idx += 1
    
    return chips


def save_chip_from_tile(
    tile_path: Path,
    chip: ChipInfo,
    output_dir: Path
) -> Path:
    """
    Extract and save a chip from a downsampled tile.
    
    Args:
        tile_path: Path to the downsampled tile
        chip: Chip information
        output_dir: Output directory
    
    Returns:
        Path to saved chip
    """
    output_path = output_dir / f"chip_{tile_path.stem}_{chip.chip_idx:04d}.tif"
    
    with rasterio.open(tile_path) as src:
        # Calculate window from bounds
        window = rasterio.windows.from_bounds(
            chip.bounds.left,
            chip.bounds.bottom,
            chip.bounds.right,
            chip.bounds.top,
            transform=src.transform
        )
        
        # Read the window
        data = src.read(window=window)
        
        # Update metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'height': window.height,
            'width': window.width,
            'transform': rasterio.windows.transform(window, src.transform)
        })
        
        # Write chip
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            dst.write(data)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Process soft labels and download Sentinel-2 imagery'
    )
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        default='soft_labels',
        help='Input directory containing soft label tiles (default: soft_labels/)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='sentinel2',
        help='Output directory for Sentinel-2 imagery (default: sentinel2/)'
    )
    parser.add_argument(
        '--chip-size',
        type=int,
        default=256,
        help='Chip size in pixels (default: 256)'
    )
    parser.add_argument(
        '--use-250',
        action='store_true',
        help='Use 250x250 chips for perfect division (recommended)'
    )
    parser.add_argument(
        '--download-sentinel',
        action='store_true',
        help='Download Sentinel-2 imagery from GEE'
    )
    parser.add_argument(
        '--save-downsampled',
        action='store_true',
        help='Save downsampled tiles'
    )
    parser.add_argument(
        '--save-chips',
        action='store_true',
        help='Save extracted chips'
    )
    
    args = parser.parse_args()
    
    # Adjust chip size if requested
    if args.use_250:
        args.chip_size = 250
        logger.info("Using 250x250 chips for perfect division")
    
    # Setup directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    downsampled_dir = output_dir / 'downsampled'
    chips_dir = output_dir / 'chips'
    sentinel_dir = output_dir / 'sentinel2'
    
    if args.save_downsampled:
        downsampled_dir.mkdir(exist_ok=True)
    if args.save_chips:
        chips_dir.mkdir(exist_ok=True)
    if args.download_sentinel:
        sentinel_dir.mkdir(exist_ok=True)
    
    # Find all TIF files
    tif_files = list(input_dir.glob('*.tif')) + list(input_dir.glob('*.tiff'))
    logger.info(f"Found {len(tif_files)} TIF files to process")
    
    # Initialize Sentinel-2 downloader if needed
    if args.download_sentinel:
        s2_downloader = Sentinel2Downloader()
    
    # Process each tile
    all_chips = []
    for tif_path in tif_files:
        logger.info(f"Processing {tif_path.name}")
        
        # Step 1: Downsample to 10m resolution
        downsampled_path = downsampled_dir / f"downsampled_{tif_path.name}"
        
        if args.save_downsampled:
            tile_info = downsample_tile(tif_path, downsampled_path)
            logger.info(f"  Downsampled from {25000}x{25000} to {tile_info.shape}")
        else:
            # Create tile info without saving
            with rasterio.open(tif_path) as src:
                # Calculate downsampled dimensions
                new_height = src.height // 20
                new_width = src.width // 20
                transform = src.transform * src.transform.scale(20, 20)
                
                tile_info = TileInfo(
                    filepath=tif_path,
                    bounds=src.bounds,
                    crs=src.crs,
                    shape=(new_height, new_width),
                    transform=transform,
                    date=extract_date_from_filename(tif_path)
                )
        
        # Step 2: Create chips
        chips = create_chips_from_tile(tile_info, chip_size=args.chip_size)
        logger.info(f"  Created {len(chips)} chips of size {args.chip_size}x{args.chip_size}")
        
        # Step 3: Save chips if requested
        if args.save_chips and args.save_downsampled:
            for chip in chips:
                save_chip_from_tile(downsampled_path, chip, chips_dir)
        
        # Step 4: Download Sentinel-2 data if requested
        if args.download_sentinel:
            for chip in chips:
                s2_downloader.download_chip(chip, sentinel_dir, date=tile_info.date)
        
        all_chips.extend(chips)
    
    # Save metadata about all chips
    metadata_file = output_dir / 'chips_metadata.json'
    metadata = {
        'total_tiles': len(tif_files),
        'total_chips': len(all_chips),
        'chip_size': args.chip_size,
        'target_gsd': 10.0,
        'chips': [
            {
                'tile': str(chip.tile_path),
                'chip_idx': chip.chip_idx,
                'bounds': {
                    'left': chip.bounds.left,
                    'bottom': chip.bounds.bottom,
                    'right': chip.bounds.right,
                    'top': chip.bounds.top
                },
                'crs': str(chip.crs)
            }
            for chip in all_chips
        ]
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_file}")
    logger.info(f"Processing complete: {len(all_chips)} chips from {len(tif_files)} tiles")


if __name__ == '__main__':
    sys.exit(main())