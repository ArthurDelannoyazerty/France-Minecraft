import json
import logging
from pathlib import Path
from typing import Dict, Iterable

import pyproj
import requests
from shapely.geometry import Polygon, shape
from shapely.ops import transform
from tqdm.auto import tqdm

import src.config as config

logger = logging.getLogger(__name__)


def init_folders():
    """Creates all necessary folders for the project if they don't exist."""
    logger.info("Initializing project folders...")
    
    # List all directory paths from the config file
    folders_to_create = [
        config.DATA_DIR, config.GRID_DIR, config.RAW_POINT_CLOUD_DIR,
        config.LOG_DIR, config.SCHEMATICS_DIR, config.MCFUNCTIONS_DIR,
        config.LIDAR_TILES_DIR, config.MNT_TILES_DIR
    ]
    
    for folder in folders_to_create:
        folder.mkdir(parents=True, exist_ok=True)
    logger.info("Folder structure verified.")


def _stream_download(url: str, output_filepath: Path, desc: str):
    """Downloads a file with a progress bar, handling large files efficiently."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Will raise an exception for 4xx/5xx status codes
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_filepath, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        # Clean up partially downloaded file
        if output_filepath.exists():
            output_filepath.unlink()


def _download_paginated_geojson(wfs_url: str, output_filepath: Path, force_download: bool):
    """
    Downloads a complete GeoJSON from a paginated WFS source.
    Handles the IGN API's `startIndex` mechanism.
    """
    if output_filepath.exists() and not force_download:
        logger.info(f"Catalog file {output_filepath.name} already exists. Skipping download.")
        return

    logger.info(f"Downloading full catalog from {wfs_url}...")
    
    try:
        # First request to get total feature count
        response = requests.get(wfs_url)
        response.raise_for_status()
        geojson_data = response.json()
        
        total_features = geojson_data.get('totalFeatures', 0)
        if total_features == 0:
            logger.warning("API returned 0 total features. Catalog might be empty.")
            return

        start_index = geojson_data.get('numberReturned', 0)
        
        with tqdm(total=total_features, initial=start_index, desc=f"Downloading {output_filepath.stem}") as pbar:
            while start_index < total_features:
                indexed_url = f'{wfs_url}&startIndex={start_index}'
                response = requests.get(indexed_url)
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch data at index {start_index}. Status: {response.status_code}. Stopping.")
                    break
                
                batch_data = response.json()
                new_features = batch_data.get('features', [])
                num_returned = len(new_features)

                if num_returned == 0:
                    logger.warning(f"Received 0 features at index {start_index} but expected more. Stopping.")
                    break
                    
                geojson_data['features'].extend(new_features)
                start_index += num_returned
                pbar.update(num_returned)

        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f)
        logger.info(f"Successfully saved catalog to {output_filepath}")

    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.error(f"Failed to download or parse catalog: {e}")


def download_ign_catalogs():
    """
    Downloads the master catalogs for both LIDAR and MNT tiles from IGN.
    """
    # Define the base URLs for the IGN WFS services
    lidar_wfs_url = "https://data.geopf.fr/private/wfs/?service=WFS&version=2.0.0&apikey=interface_catalogue&request=GetFeature&typeNames=IGNF_LIDAR-HD_TA:nuage-dalle&outputFormat=application/json"
    mnt_wfs_url = "https://data.geopf.fr/private/wfs/?service=WFS&version=2.0.0&apikey=interface_catalogue&request=GetFeature&typeNames=IGNF_LIDAR-HD_TA:mnt-dalle&outputFormat=application/json"
    
    _download_paginated_geojson(lidar_wfs_url, config.LIDAR_CATALOG_FILE, config.FORCE_DOWNLOAD_CATALOG)
    _download_paginated_geojson(mnt_wfs_url, config.MNT_CATALOG_FILE, config.FORCE_DOWNLOAD_CATALOG)


def find_intersecting_tiles(zone_geojson: dict) -> Dict[str, Dict]:
    """
    Finds pairs of MNT and LIDAR tiles that intersect with the given zone.
    
    Returns:
        A dictionary where keys are BBOX strings and values contain filepaths 
        and metadata for compatible LIDAR and MNT tiles.
    """
    logger.info("Finding intersecting tiles for the specified zone...")

    # 1. Load the zone polygon and transform it to the correct CRS (Lambert-93)
    try:
        zone_coords = zone_geojson['features'][0]['geometry']['coordinates'][0]
        zone_poly_wgs84 = Polygon(zone_coords)
        
        # Define transformers for CRS conversion
        wgs84_to_lambert93 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True).transform
        zone_poly_lambert93 = transform(wgs84_to_lambert93, zone_poly_wgs84)
    except (KeyError, IndexError) as e:
        logger.error(f"Invalid GeoJSON structure for the zone file: {e}")
        return {}

    # 2. Load the downloaded catalogs
    try:
        with open(config.LIDAR_CATALOG_FILE, 'r') as f:
            lidar_catalog = json.load(f)
        with open(config.MNT_CATALOG_FILE, 'r') as f:
            mnt_catalog = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Could not read catalog files. Please run download_ign_catalogs() first. Error: {e}")
        return {}
        
    # 3. Find intersecting features and pair them by BBOX
    compatible_tiles = {}

    # Process LIDAR tiles
    for feature in tqdm(lidar_catalog.get('features', []), desc="Filtering LIDAR tiles"):
        tile_poly = shape(feature['geometry'])
        if tile_poly.intersects(zone_poly_lambert93):
            bbox = [int(coord) for coord in tile_poly.bounds]
            bbox_str = '-'.join(map(str, bbox))
            if bbox_str not in compatible_tiles:
                compatible_tiles[bbox_str] = {}
            compatible_tiles[bbox_str]['lidar'] = {
                'filepath': config.LIDAR_TILES_DIR / feature['properties']['name'],
                'url': feature['properties']['url'],
                'bbox': bbox
            }
            
    # Process MNT tiles
    for feature in tqdm(mnt_catalog.get('features', []), desc="Filtering MNT tiles"):
        tile_poly = shape(feature['geometry'])
        if tile_poly.intersects(zone_poly_lambert93):
            bbox = [int(coord) for coord in tile_poly.bounds]
            bbox_str = '-'.join(map(str, bbox))
            if bbox_str in compatible_tiles: # Only add if a matching LIDAR tile was found
                compatible_tiles[bbox_str]['mnt'] = {
                    'filepath': config.MNT_TILES_DIR / feature['properties']['name'],
                    'url': feature['properties']['url'],
                    'bbox': bbox
                }
    
    # 4. Filter for pairs that have both MNT and LIDAR data
    final_tiles = {
        bbox_str: data for bbox_str, data in compatible_tiles.items()
        if 'lidar' in data and 'mnt' in data
    }
    
    logger.info(f"Found {len(final_tiles)} compatible MNT/LIDAR tile pairs.")
    return final_tiles


def download_tiles(tiles_to_download: Iterable[Dict]):
    """
    Downloads all LIDAR and MNT files for the given tiles if they don't already exist.
    """
    logger.info("Checking for and downloading required tile files...")
    all_files_to_check = []
    for tile_data in tiles_to_download:
        if 'lidar' in tile_data:
            all_files_to_check.append(tile_data['lidar'])
        if 'mnt' in tile_data:
            all_files_to_check.append(tile_data['mnt'])

    for file_info in tqdm(all_files_to_check, desc="Downloading tiles"):
        filepath = file_info['filepath']
        if not filepath.exists():
            logger.info(f"Downloading {filepath.name}...")
            _stream_download(file_info['url'], filepath, desc=filepath.name)
        else:
            logger.debug(f"File {filepath.name} already exists. Skipping.")
    logger.info("All required tiles are available locally.")