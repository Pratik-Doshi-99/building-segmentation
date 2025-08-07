#!/usr/bin/env python3
"""
Pipeline verification script for building segmentation workflow.
Verifies all stages: original tile -> downsampling -> chipping -> Sentinel-2 download.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from pyproj import Transformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineVerifier:
    """Verifies the entire building segmentation pipeline."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.soft_labels_dir = self.base_dir / "soft_labels"
        self.sentinel_dir = self.base_dir / "sentinel2"
        self.downsampled_dir = self.sentinel_dir / "downsampled"
        self.chips_dir = self.sentinel_dir / "chips"
        self.sentinel_chips_dir = self.sentinel_dir / "sentinel2"  # Sentinel-2 chips location
        self.metadata_file = self.sentinel_dir / "chips_metadata.json"
        
    def verify_tile_pipeline(self, tile_name: str) -> Dict:
        """
        Verify the complete pipeline for a single tile.
        
        Args:
            tile_name: Name of the tile (e.g., 'tile_YV3xUniiUOc')
        
        Returns:
            Dictionary containing verification results
        """
        logger.info(f"Verifying pipeline for {tile_name}")
        
        results = {
            'tile_name': tile_name,
            'original_tile': {},
            'downsampled_tile': {},
            'chips': {},
            'sentinel_chips': {},
            'overall_status': 'unknown'
        }
        
        try:
            # Step 1: Verify original tile
            results['original_tile'] = self._verify_original_tile(tile_name)
            
            # Step 2: Verify downsampled tile
            results['downsampled_tile'] = self._verify_downsampled_tile(tile_name, results['original_tile'])
            
            # Step 3: Verify chips
            results['chips'] = self._verify_chips(tile_name, results['downsampled_tile'])
            
            # Step 4: Verify Sentinel-2 downloads
            results['sentinel_chips'] = self._verify_sentinel_chips(tile_name)
            
            # Overall status
            all_stages_ok = all([
                results['original_tile'].get('status') == 'ok',
                results['downsampled_tile'].get('status') == 'ok',
                results['chips'].get('status') == 'ok',
                results['sentinel_chips'].get('status') in ['ok', 'partial']
            ])
            results['overall_status'] = 'ok' if all_stages_ok else 'failed'
            
        except Exception as e:
            logger.error(f"Pipeline verification failed for {tile_name}: {e}")
            results['overall_status'] = 'error'
            results['error'] = str(e)
            
        return results
    
    def _verify_original_tile(self, tile_name: str) -> Dict:
        """Verify the original soft label tile."""
        logger.info(f"Verifying original tile: {tile_name}")
        
        tile_path = self.soft_labels_dir / f"{tile_name}.tif"
        if not tile_path.exists():
            return {'status': 'missing', 'path': str(tile_path)}
        
        try:
            with rasterio.open(tile_path) as src:
                result = {
                    'status': 'ok',
                    'path': str(tile_path),
                    'shape': src.shape,
                    'bands': src.count,
                    'dtype': str(src.dtypes[0]),
                    'crs': str(src.crs),
                    'bounds': {
                        'left': src.bounds.left,
                        'bottom': src.bounds.bottom,
                        'right': src.bounds.right,
                        'top': src.bounds.top
                    },
                    'transform': list(src.transform),
                    'gsd': abs(src.transform[0])  # Ground Sample Distance
                }
                
                # Verify expected properties
                if src.shape != (25000, 25000):
                    result['warnings'] = result.get('warnings', [])
                    result['warnings'].append(f"Unexpected shape: {src.shape}, expected (25000, 25000)")
                
                logger.info(f"Original tile verified: {src.shape} pixels, {src.count} bands, GSD: {result['gsd']:.1f}m")
                return result
                
        except Exception as e:
            return {'status': 'error', 'path': str(tile_path), 'error': str(e)}
    
    def _verify_downsampled_tile(self, tile_name: str, original_info: Dict) -> Dict:
        """Verify the downsampled tile."""
        logger.info(f"Verifying downsampled tile: {tile_name}")
        
        downsampled_path = self.downsampled_dir / f"downsampled_{tile_name}.tif"
        if not downsampled_path.exists():
            return {'status': 'missing', 'path': str(downsampled_path)}
        
        try:
            with rasterio.open(downsampled_path) as src:
                result = {
                    'status': 'ok',
                    'path': str(downsampled_path),
                    'shape': src.shape,
                    'bands': src.count,
                    'dtype': str(src.dtypes[0]),
                    'crs': str(src.crs),
                    'bounds': {
                        'left': src.bounds.left,
                        'bottom': src.bounds.bottom,
                        'right': src.bounds.right,
                        'top': src.bounds.top
                    },
                    'transform': list(src.transform),
                    'gsd': abs(src.transform[0])
                }
                
                # Verify downsampling ratio
                if original_info.get('status') == 'ok':
                    original_shape = original_info['shape']
                    expected_shape = (original_shape[0] // 20, original_shape[1] // 20)
                    
                    if src.shape != expected_shape:
                        result['warnings'] = result.get('warnings', [])
                        result['warnings'].append(f"Unexpected downsampled shape: {src.shape}, expected {expected_shape}")
                    
                    # Verify bounds are approximately the same
                    orig_bounds = original_info['bounds']
                    bounds_match = all([
                        abs(src.bounds.left - orig_bounds['left']) < 1.0,
                        abs(src.bounds.bottom - orig_bounds['bottom']) < 1.0,
                        abs(src.bounds.right - orig_bounds['right']) < 1.0,
                        abs(src.bounds.top - orig_bounds['top']) < 1.0
                    ])
                    
                    if not bounds_match:
                        result['warnings'] = result.get('warnings', [])
                        result['warnings'].append("Bounds do not match original tile")
                
                logger.info(f"Downsampled tile verified: {src.shape} pixels, GSD: {result['gsd']:.1f}m")
                return result
                
        except Exception as e:
            return {'status': 'error', 'path': str(downsampled_path), 'error': str(e)}
    
    def _verify_chips(self, tile_name: str, downsampled_info: Dict) -> Dict:
        """Verify the chip splitting process."""
        logger.info(f"Verifying chips for: {tile_name}")
        
        result = {
            'status': 'unknown',
            'expected_chips': 16,  # 4x4 grid
            'found_chips': 0,
            'chips_info': [],
            'coverage_verification': {}
        }
        
        # Find all chips for this tile
        chip_pattern = f"chip_downsampled_{tile_name}_*.tif"
        chip_files = list(self.chips_dir.glob(chip_pattern))
        result['found_chips'] = len(chip_files)
        
        if result['found_chips'] == 0:
            result['status'] = 'missing'
            return result
        
        chip_infos = []
        for chip_file in sorted(chip_files):
            try:
                with rasterio.open(chip_file) as src:
                    chip_info = {
                        'filename': chip_file.name,
                        'shape': src.shape,
                        'bounds': {
                            'left': src.bounds.left,
                            'bottom': src.bounds.bottom,
                            'right': src.bounds.right,
                            'top': src.bounds.top
                        },
                        'crs': str(src.crs),
                        'gsd': abs(src.transform[0])
                    }
                    chip_infos.append(chip_info)
            except Exception as e:
                chip_infos.append({
                    'filename': chip_file.name,
                    'error': str(e)
                })
        
        result['chips_info'] = chip_infos
        
        # Verify chip coverage if we have downsampled tile info
        if downsampled_info.get('status') == 'ok' and chip_infos:
            result['coverage_verification'] = self._verify_chip_coverage(
                downsampled_info['bounds'], chip_infos
            )
        
        # Status determination
        valid_chips = [c for c in chip_infos if 'error' not in c]
        if len(valid_chips) == result['expected_chips']:
            result['status'] = 'ok'
        elif len(valid_chips) > 0:
            result['status'] = 'partial'
        else:
            result['status'] = 'failed'
        
        logger.info(f"Chips verified: {len(valid_chips)}/{result['expected_chips']} valid chips")
        return result
    
    def _verify_chip_coverage(self, parent_bounds: Dict, chip_infos: List[Dict]) -> Dict:
        """Verify that chips cover the parent tile completely."""
        valid_chips = [c for c in chip_infos if 'bounds' in c]
        
        if not valid_chips:
            return {'status': 'no_valid_chips'}
        
        # Calculate combined bounds of all chips
        all_lefts = [c['bounds']['left'] for c in valid_chips]
        all_rights = [c['bounds']['right'] for c in valid_chips]
        all_bottoms = [c['bounds']['bottom'] for c in valid_chips]
        all_tops = [c['bounds']['top'] for c in valid_chips]
        
        combined_bounds = {
            'left': min(all_lefts),
            'right': max(all_rights),
            'bottom': min(all_bottoms),
            'top': max(all_tops)
        }
        
        # Check if combined bounds match parent bounds (within tolerance)
        tolerance = 1.0  # 1 meter tolerance
        bounds_match = all([
            abs(combined_bounds['left'] - parent_bounds['left']) < tolerance,
            abs(combined_bounds['right'] - parent_bounds['right']) < tolerance,
            abs(combined_bounds['bottom'] - parent_bounds['bottom']) < tolerance,
            abs(combined_bounds['top'] - parent_bounds['top']) < tolerance
        ])
        
        return {
            'status': 'ok' if bounds_match else 'mismatch',
            'parent_bounds': parent_bounds,
            'combined_chip_bounds': combined_bounds,
            'tolerance_meters': tolerance
        }
    
    def _verify_sentinel_chips(self, tile_name: str) -> Dict:
        """Verify Sentinel-2 chip downloads."""
        logger.info(f"Verifying Sentinel-2 chips for: {tile_name}")
        
        result = {
            'status': 'unknown',
            'expected_chips': 16,
            'found_chips': 0,
            'valid_chips': 0,
            'chips_info': []
        }
        
        # Find Sentinel-2 chips
        sentinel_pattern = f"sentinel2_chip_downsampled_{tile_name}_*.tif"
        sentinel_files = list(self.sentinel_chips_dir.glob(sentinel_pattern))
        result['found_chips'] = len(sentinel_files)
        
        if result['found_chips'] == 0:
            result['status'] = 'missing'
            return result
        
        valid_count = 0
        for sentinel_file in sorted(sentinel_files):
            try:
                with rasterio.open(sentinel_file) as src:
                    chip_info = {
                        'filename': sentinel_file.name,
                        'shape': src.shape,
                        'bands': src.count,
                        'dtype': [str(dtype) for dtype in src.dtypes],
                        'bounds': {
                            'left': src.bounds.left,
                            'bottom': src.bounds.bottom,
                            'right': src.bounds.right,
                            'top': src.bounds.top
                        },
                        'crs': str(src.crs),
                        'gsd': abs(src.transform[0]) if src.transform else None,
                        'data_range': {}
                    }
                    
                    # Check data ranges for each band
                    for i in range(src.count):
                        band_data = src.read(i + 1)
                        chip_info['data_range'][f'band_{i+1}'] = {
                            'min': float(np.min(band_data)),
                            'max': float(np.max(band_data)),
                            'mean': float(np.mean(band_data))
                        }
                    
                    result['chips_info'].append(chip_info)
                    valid_count += 1
                    
            except Exception as e:
                result['chips_info'].append({
                    'filename': sentinel_file.name,
                    'error': str(e)
                })
        
        result['valid_chips'] = valid_count
        
        # Status determination
        if valid_count == result['expected_chips']:
            result['status'] = 'ok'
        elif valid_count > 0:
            result['status'] = 'partial'
        else:
            result['status'] = 'failed'
        
        logger.info(f"Sentinel-2 chips verified: {valid_count}/{result['expected_chips']} valid chips")
        return result
    
    def verify_all_tiles(self) -> Dict[str, Dict]:
        """Verify pipeline for all available tiles."""
        logger.info("Verifying pipeline for all tiles")
        
        # Find all soft label tiles
        soft_label_files = list(self.soft_labels_dir.glob("tile_*.tif"))
        tile_names = [f.stem for f in soft_label_files]
        
        results = {}
        for tile_name in tile_names:
            results[tile_name] = self.verify_tile_pipeline(tile_name)
        
        return results
    
    def generate_report(self, results: Dict[str, Dict]) -> str:
        """Generate a human-readable report."""
        report_lines = [
            "=" * 60,
            "PIPELINE VERIFICATION REPORT",
            "=" * 60,
            ""
        ]
        
        # Summary
        total_tiles = len(results)
        ok_tiles = sum(1 for r in results.values() if r.get('overall_status') == 'ok')
        partial_tiles = sum(1 for r in results.values() if r.get('overall_status') in ['partial', 'ok'] and 
                           r.get('sentinel_chips', {}).get('status') == 'partial')
        
        report_lines.extend([
            f"Total tiles processed: {total_tiles}",
            f"Fully successful: {ok_tiles}",
            f"Partially successful: {partial_tiles}",
            f"Failed: {total_tiles - ok_tiles}",
            ""
        ])
        
        # Detailed results
        for tile_name, result in results.items():
            report_lines.extend([
                f"TILE: {tile_name}",
                "-" * 40
            ])
            
            # Original tile
            orig = result.get('original_tile', {})
            report_lines.append(f"Original tile: {orig.get('status', 'unknown')}")
            if orig.get('status') == 'ok':
                report_lines.append(f"  Shape: {orig['shape']}, GSD: {orig['gsd']:.1f}m")
            
            # Downsampled tile
            down = result.get('downsampled_tile', {})
            report_lines.append(f"Downsampled tile: {down.get('status', 'unknown')}")
            if down.get('status') == 'ok':
                report_lines.append(f"  Shape: {down['shape']}, GSD: {down['gsd']:.1f}m")
            
            # Chips
            chips = result.get('chips', {})
            report_lines.append(f"Chips: {chips.get('status', 'unknown')} ({chips.get('found_chips', 0)}/16)")
            
            # Sentinel-2
            sentinel = result.get('sentinel_chips', {})
            report_lines.append(f"Sentinel-2: {sentinel.get('status', 'unknown')} ({sentinel.get('valid_chips', 0)}/16)")
            
            report_lines.extend(["", ""])
        
        return "\n".join(report_lines)

def main():
    parser = argparse.ArgumentParser(description="Verify building segmentation pipeline")
    parser.add_argument('--tile', help='Verify specific tile (e.g., tile_YV3xUniiUOc)')
    parser.add_argument('--all', action='store_true', help='Verify all tiles')
    parser.add_argument('--output', help='Save report to file')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    
    args = parser.parse_args()
    
    verifier = PipelineVerifier()
    
    if args.tile:
        results = {args.tile: verifier.verify_tile_pipeline(args.tile)}
    elif args.all:
        results = verifier.verify_all_tiles()
    else:
        # Default: verify all tiles
        results = verifier.verify_all_tiles()
    
    if args.json:
        import json
        output = json.dumps(results, indent=2)
    else:
        output = verifier.generate_report(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Report saved to {args.output}")
    else:
        print(output)

if __name__ == "__main__":
    main()