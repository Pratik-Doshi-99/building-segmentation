#!/usr/bin/env python3
"""
Script 1: Download Google Open Buildings temporal data tiles
"""

import argparse
import os
import subprocess
import concurrent.futures
from pathlib import Path
import sys
import time
from typing import List, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_tile(url: str, output_dir: Path, retry_count: int = 3) -> Optional[str]:
    """
    Download a single tile using gsutil.
    
    Args:
        url: GCS URL of the tile
        output_dir: Directory to save the file
        retry_count: Number of retries on failure
    
    Returns:
        Path to downloaded file or None if failed
    """
    # Extract filename from URL
    filename = url.split('/')[-1]
    output_path = output_dir / filename
    
    # Skip if already exists
    if output_path.exists():
        logger.info(f"File already exists: {filename}")
        return str(output_path)
    
    for attempt in range(retry_count):
        try:
            # Use gsutil to download
            cmd = ['gsutil', 'cp', url, str(output_path)]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            logger.info(f"Successfully downloaded: {filename}")
            return str(output_path)
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Attempt {attempt + 1} failed for {filename}: {e.stderr}")
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to download {filename} after {retry_count} attempts")
                return None
    
    return None


def download_tiles_concurrent(
    urls: List[str], 
    output_dir: Path, 
    max_workers: int = 4
) -> List[str]:
    """
    Download multiple tiles concurrently.
    
    Args:
        urls: List of GCS URLs
        output_dir: Directory to save files
        max_workers: Number of concurrent downloads
    
    Returns:
        List of successfully downloaded file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_url = {
            executor.submit(download_tile, url, output_dir): url 
            for url in urls
        }
        
        # Process completed downloads
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                if result:
                    downloaded_files.append(result)
            except Exception as e:
                logger.error(f"Exception downloading {url}: {e}")
    
    return downloaded_files


def main():
    parser = argparse.ArgumentParser(
        description='Download Google Open Buildings temporal data tiles'
    )
    parser.add_argument(
        'url_file',
        help='Text file containing URLs (one per line)'
    )
    parser.add_argument(
        '-n', '--num-links',
        type=int,
        default=None,
        help='Number of links to download (default: all)'
    )
    parser.add_argument(
        '-c', '--concurrent',
        type=int,
        default=4,
        help='Number of concurrent downloads (default: 4)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='soft_labels',
        help='Output directory (default: soft_labels/)'
    )
    parser.add_argument(
        '--check-gsutil',
        action='store_true',
        help='Check if gsutil is installed'
    )
    
    args = parser.parse_args()
    
    # Check gsutil availability
    if args.check_gsutil or True:  # Always check
        try:
            subprocess.run(
                ['gsutil', 'version'], 
                capture_output=True, 
                check=True
            )
            logger.info("gsutil is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("gsutil is not installed or not in PATH")
            logger.error("Install with: pip install gsutil")
            sys.exit(1)
    
    # Read URLs from file
    url_file = Path(args.url_file)
    if not url_file.exists():
        logger.error(f"URL file not found: {url_file}")
        sys.exit(1)
    
    with open(url_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Found {len(urls)} URLs in {url_file}")
    
    # Limit number of downloads if specified
    if args.num_links:
        urls = urls[:args.num_links]
        logger.info(f"Limiting to {len(urls)} downloads")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download tiles
    logger.info(f"Starting download with {args.concurrent} concurrent workers")
    start_time = time.time()
    
    downloaded = download_tiles_concurrent(
        urls, 
        output_dir, 
        max_workers=args.concurrent
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Downloaded {len(downloaded)}/{len(urls)} files in {elapsed:.2f} seconds")
    
    # Save list of downloaded files
    manifest_file = output_dir / 'download_manifest.txt'
    with open(manifest_file, 'w') as f:
        for filepath in downloaded:
            f.write(f"{filepath}\n")
    logger.info(f"Saved download manifest to {manifest_file}")
    
    return 0 if len(downloaded) == len(urls) else 1


if __name__ == '__main__':
    sys.exit(main())