import rasterio
import laspy
import wget
import logging
import os
import requests
import json
import numpy as np
import pyvista as pv
import geopandas as gpd
import mcschematic
import pyproj

from scipy.interpolate import LinearNDInterpolator
# NEW: Import warnings to suppress potential division-by-zero in IDW if needed
import warnings
from collections import defaultdict


from shapely.geometry import mapping, Polygon
from shapely.ops import unary_union
from shapely.vectorized import contains
from pathlib import Path
from utils.logger import setup_logging
from script import find_occupied_voxels_vectorized
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


logger = logging.getLogger(__name__)


crs_leaflet = 'EPSG:4326'
crs_ign =     'EPSG:2154'


def init_folders():
    logger.info('Create folders for the project')
    Path('data/data_grille'    ).mkdir(parents=True, exist_ok=True)
    Path('data/orders'         ).mkdir(parents=True, exist_ok=True)
    Path('data/raw_point_cloud').mkdir(parents=True, exist_ok=True)
    Path('data/logs'           ).mkdir(parents=True, exist_ok=True)
    Path('data/myschems'       ).mkdir(parents=True, exist_ok=True)
    Path('data/mcfunctions'    ).mkdir(parents=True, exist_ok=True)


def geodataframe_from_leaflet_to_ign(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.set_crs(crs_leaflet)
    gdf_transformed  = gdf.to_crs(crs_ign)
    return gdf_transformed


def geodataframe_from_ign_to_leaflet(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.set_crs(crs_ign)
    gdf_transformed  = gdf.to_crs(crs_leaflet)
    return gdf_transformed


def download_ign_available_tiles(output_filepath:str, data_type:str, force_download:bool=False):
    # If file already exists and no force, then do not download file
    if os.path.isfile(output_filepath) and not force_download:
        logger.info(f"File {output_filepath} already exists. Skipping download.")
        return

    if data_type=='point_cloud':
        wfs_url = "https://data.geopf.fr/private/wfs/?service=WFS&version=2.0.0&apikey=interface_catalogue&request=GetFeature&typeNames=IGNF_LIDAR-HD_TA:nuage-dalle&outputFormat=application/json"
    elif data_type=='mnt':
        wfs_url = "https://data.geopf.fr/private/wfs/?service=WFS&version=2.0.0&apikey=interface_catalogue&request=GetFeature&typeNames=IGNF_LIDAR-HD_TA:mnt-dalle&outputFormat=application/json"


    # First request to initialize the geojson and know the total number of features
    logger.info('First download for all features available')
    response = requests.get(wfs_url)
    if response.status_code != 200:
        logger.error(f"Failed to retrieve data for url : {wfs_url}. HTTP Status code: {response.status_code}")
        exit(1)

    try:
        geojson = response.json()
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from {wfs_url}. Response text: {response.text[:500]}...")
        exit(1)

    # Check if necessary keys exist
    if 'totalFeatures' not in geojson or 'numberReturned' not in geojson or 'features' not in geojson:
         logger.error(f"Unexpected JSON structure from {wfs_url}. Missing keys. Response: {geojson}")
         exit(1)


    total_features = geojson['totalFeatures']
    number_returned = geojson['numberReturned']
    logger.info(f'First download finished. Total Feature : {total_features}  | Number Returned : {number_returned}')

    start_index = number_returned
    pbar = tqdm(total=total_features, initial=start_index, desc="Downloading tile metadata")
    while start_index < total_features:
        # logger.info(f'Downloading features from index {start_index} / {total_features}')
        wfs_url_indexed = f'{wfs_url}&startIndex={start_index}'
        response = requests.get(wfs_url_indexed)
        if response.status_code != 200:
            logger.warning(f"Failed to retrieve data for url : {wfs_url_indexed}. HTTP Status code: {response.status_code}. Trying to continue...")
            # Decide how to handle errors: break, continue, retry? For now, log and break.
            break
            # If continuing: start_index += number_returned # Need a default increment or better logic
            # pbar.update(number_returned) # Assuming number_returned is the expected batch size
            # continue

        try:
            response_data = response.json()
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON response from {wfs_url_indexed}. Response text: {response.text[:500]}... Skipping this batch.")
            # Need logic to advance start_index appropriately if skipping
            break # Simplest for now

        current_features = response_data.get('features', [])
        number_returned_this_batch = response_data.get('numberReturned', len(current_features)) # Use actual number returned

        if not current_features and number_returned_this_batch > 0:
             logger.warning(f"API reported {number_returned_this_batch} returned, but 'features' array is empty or missing. Index: {start_index}")
             # Maybe retry or break

        geojson['features'].extend(current_features)
        start_index += number_returned_this_batch
        pbar.update(number_returned_this_batch)

        # Safety break if number_returned_this_batch is 0 to avoid infinite loop
        if number_returned_this_batch == 0 and start_index < total_features:
            logger.warning(f"Received 0 features at index {start_index} but expected more. Stopping download.")
            break
    pbar.close()

    logger.info(f"Finished downloading metadata. Total features in GeoJSON: {len(geojson['features'])}")
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved all available tiles metadata to {output_filepath}")





def merge_all_geojson_features(geojson_filepath:str, merged_geojson_filepath:str, force:bool=False):
    # If file already exists and no force, then do not download file
    if os.path.isfile(merged_geojson_filepath) and not force: return
    logger.info(f'Merging all tiles from geojson : {geojson_filepath}')
    gdf = gpd.read_file(geojson_filepath)
    merged_gdf = unary_union(gdf.geometry)
    merged_gdf_dict = mapping(merged_gdf)
    with open(merged_geojson_filepath, 'w') as f:
        f.write(json.dumps(merged_gdf_dict))
    logger.info(f'Merged geojson saved at {merged_geojson_filepath}')


def decimate_array(array:np.ndarray, percentage_to_remove):
    """
    Remove a given percentage of rows from a NumPy array randomly.

    Parameters:
    - array (np.ndarray): Input array of shape (n, 3).
    - percentage (float): Percentage of rows to remove (between 0 and 100).
    - seed (int, optional): Seed for the random number generator for reproducibility.

    Returns:
    - np.ndarray: Array with the specified percentage of rows removed.
    """
    logger.info(f'Decimating array of shape {array.shape} by {percentage_to_remove}%')
    if not 0 <= percentage_to_remove <= 100:
        raise ValueError("Percentage must be between 0 and 100.")
    
    if percentage_to_remove==0: return array

    mask = np.random.rand(array.shape[0]) > (percentage_to_remove / 100.0)
    decimated_array = array[mask]

    logger.info(f'Decimation done. Number of points : {array.shape[0]} (before) | {decimated_array.shape[0]} (after)')
    return decimated_array


def display_point_cloud(points):
    """
    Display a 3D point cloud using PyVista.
    
    Parameters:
    - points (numpy.ndarray): A Nx3 array of 3D points (x, y, z).
    """
    logger.info('Display point cloud')
    point_cloud = pv.PolyData(points)
    plotter = pv.Plotter()
    plotter.add_points(point_cloud, cmap="viridis", point_size=1)
    plotter.set_background("white")
    plotter.show()


def get_intersecting_tiles_from_order(geojson_order_filepath:str, geojson_all_tiles_available_filepath:str) -> gpd.GeoDataFrame:
    logger.info('Executing intersection of tiles for the order')
    logger.info('Loading geojson of all tiles available')
    available_tiles_gdf = gpd.read_file(geojson_all_tiles_available_filepath)
    
    logger.info('Loading geojson orders')
    order_gdf = gpd.read_file(geojson_order_filepath)
    logger.info(f'Order geojson head : {order_gdf.head()}')

    logger.info('Filtering the intersecting tiles')
    intersect_gdf = available_tiles_gdf[available_tiles_gdf.intersects(order_gdf.geometry.iloc[0])]
    logger.info(f'Intersect GeoDataFrame head : {intersect_gdf.head()}')
    return intersect_gdf


def download_tiles_from_gdf(gdf:gpd.GeoDataFrame, laz_folderpath:Path):
    for index, row in gdf.iterrows():
        filename = row['name']
        url = row['url']
        filepath = laz_folderpath / filename
        if not os.path.isfile(filepath):
            logger.info(f'Downloading file {filename} into {filepath}')
            wget.download(url, out=str(filepath))


def filter_points_by_polygon(xyz, polygon):
    # Use shapely.vectorized.contains for efficient point-in-polygon testing
    logger.info(f'Filtering {xyz.shape[0]} points with the polygon {polygon}')
    inside_mask = contains(polygon, xyz[:, 0], xyz[:, 1])
    filtered_points = xyz[inside_mask]
    logger.info(f'Points filtered. {xyz.shape[0]} --> {filtered_points.shape[0]} (Keeping {filtered_points.shape[0]/xyz.shape[0]} %)')
    return filtered_points


def download_available_tile(geojson_all_tiles_available_filepath:Path, laz_folderpath:Path):
    
    logger.info('Loading geojson of all tiles available')
    available_tiles_gdf = gpd.read_file(geojson_all_tiles_available_filepath)
    logger.info('Loading complete')
    
    tile = available_tiles_gdf.iloc[0]
    print(tile)

    filename = tile.name
    url_tile = tile.url
    filepath = laz_folderpath / str(filename) 
    print(filepath)
    if not os.path.isfile(filepath):
        logger.info(f'Downloading file {filename} into {filepath}')
        wget.download(url_tile, out=str(filepath))




def interpolate_point_cloud(points, grid_cell_size):
    """
    Processes a 3D point cloud by:
      1. Creating a 2D grid that covers the full XY extent of the point cloud.
      2. Flattening the point cloud onto that grid and flagging the empty cells.
      3. For each empty cell, adding 4 new points located at evenly-spaced positions inside the cell.
         Their z-values are obtained via a linear interpolation using the original points.
    
    Parameters:
      points (np.ndarray): Nx3 numpy array of points (x, y, z).
      grid_cell_size (float): grid cell size (default 0.5 meters).
    
    Returns:
      np.ndarray: The augmented array of points (original + the new points).
    """
    logger.info("Applying ground hole filling...")

    # Determine grid bounds (using the points' x-y projection)
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)
    
    # Determine the number of cells required in each dimension (ceiling)
    n_cells_x = int(np.floor((x_max - x_min) / grid_cell_size))
    n_cells_y = int(np.floor((y_max - y_min) / grid_cell_size))
    
    # Create an occupancy grid that marks cells with any point present.
    # Compute cell indices (i for x, j for y) for each point.
    ix = ((points[:, 0] - x_min) / grid_cell_size).astype(np.int32)
    iy = ((points[:, 1] - y_min) / grid_cell_size).astype(np.int32)

    # remove the points on the edge of the grid by replacing them by the last cell
    ix[np.argwhere(ix==n_cells_x)] = n_cells_x - 1
    iy[np.argwhere(iy==n_cells_y)] = n_cells_y - 1
    
    # Initialize a boolean grid of False (empty)
    occupancy = np.zeros((n_cells_x, n_cells_y), dtype=bool)
    occupancy[ix, iy] = True  # mark cells that contain at least one point
    
    # Identify indices (i,j) of the empty cells
    empty_cells = np.argwhere(~occupancy)  # each row: [i, j]
    
    # Set new points for each empty cells

    offsets_list = [[x/10,y/10] for x in range(1, 10) for y in range(1, 10)]
    offsets = np.array(offsets_list) * grid_cell_size
    
    
    # Compute the lower left coordinate for each empty cell.
    empty_cell_origin = np.empty((empty_cells.shape[0], 2))
    empty_cell_origin[:, 0] = x_min + empty_cells[:, 0] * grid_cell_size
    empty_cell_origin[:, 1] = y_min + empty_cells[:, 1] * grid_cell_size
    
    # For each empty cell, compute the coordinates for the 4 new sample points.
    # We use broadcasting to add the offsets to each cell's origin.
    new_xy = (empty_cell_origin[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
    
    # Create a linear interpolator for the z values from the original points using their XY positions.
    interp_lin = LinearNDInterpolator(points[:, :2], points[:, 2])
    new_z = interp_lin(new_xy)
    
    # (Optional) In case some points fall outside the convex hull and are NaN, one may fill them using nearest neighbor:
    nan_mask = np.isnan(new_z)
    if np.any(nan_mask):
        from scipy.interpolate import NearestNDInterpolator
        interp_nearest = NearestNDInterpolator(points[:, :2], points[:, 2])
        new_z[nan_mask] = interp_nearest(new_xy[nan_mask])
    
    # Combine the new XY and Z coordinates.
    new_points = np.hstack([new_xy, new_z[:, None]])
    # display_point_cloud(new_points)
    
    # Merge the original points with the newly generated points.
    all_points = np.vstack([points, new_points])
    # display_point_cloud(all_points)
    return all_points

import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

Point = Tuple[float, float, float]
Class = str

def dominant_voxel_points(
    point_coordinates: Dict[Class, List[Point]]
) -> Tuple[Dict[Tuple[int,int,int], Class], Dict[Class, List[Point]]]:
    """
    Given a mapping from point-class → list of (x,y,z) coordinates (floats in 0.5 increments),
    treats each integer-floored (x,y,z) as a “voxel”.  Determines:

      1. dominant_per_voxel: for each voxel, which class has the most points in that voxel
         (ties broken by choosing the smaller class identifier).
      2. filtered_points: for each class, the subset of its input points that lie in a voxel
         where it is dominant.

    Returns
    -------
    dominant_per_voxel : dict[ (i,j,k) → class ]
    filtered_points    : dict[ class → list of (x,y,z) ]
    """
    # 1) count per-voxel, per-class
    voxel_counts: Dict[Tuple[int,int,int], Counter] = defaultdict(Counter)
    for cls, pts in point_coordinates.items():
        for x,y,z in pts:
            voxel = (math.floor(x), math.floor(y), math.floor(z))
            voxel_counts[voxel][cls] += 1

    # 2) pick dominant class in each voxel
    dominant_per_voxel: Dict[Tuple[int,int,int], Class] = {}
    for voxel, cnts in voxel_counts.items():
        # sort by (-count, class) so that highest count wins, tie → smaller class id
        dominant_cls, _ = sorted(
            cnts.items(),
            key=lambda item: (-item[1], item[0])
        )[0]
        dominant_per_voxel[voxel] = dominant_cls

    # 3) filter original points
    filtered_points: Dict[Class, List[Point]] = {cls: [] for cls in point_coordinates}
    for cls, pts in point_coordinates.items():
        for x,y,z in pts:
            voxel = (math.floor(x), math.floor(y), math.floor(z))
            if dominant_per_voxel[voxel] == cls:
                filtered_points[cls].append((x,y,z))

    return dominant_per_voxel, filtered_points



# --- Main Execution ---
if __name__=='__main__':
    log_name = Path(__file__).stem
    setup_logging(log_name)
    logger = logging.getLogger(__name__)

    init_folders()

    # ---------------------------------------------------------------------------- #
    #                                 Configuration                                #
    # ---------------------------------------------------------------------------- #
    # Download and file paths
    FORCE_DOWNLOAD_LIDAR_CATALOG = False
    lidar_tiles_available_filepath = Path('data/grid/lidar_public_tiles_available.geojson')
    mnt_tiles_available_filepath   = Path('data/grid/mnt_public_tiles_available.geojson')

    zone_geojson_filepath = Path('data/zone_test.geojson')

    lidar_folderpath      = Path('data/tiles/lidar/')
    mnt_folderpath        = Path('data/tiles/mnt')
    schematic_folderpath  = Path('data/myschems')
    mcfunction_folderpath = Path('data/mcfunctions')

    SEARCH_FOR_TILE_IN_ZONE = False


    # Minecraft parameters
    LOWEST_MINECRAFT_POINT = 0  #2031          # If normal minecraft : -60
    HIGHEST_MINECRAFT_POINT = 2025          # If normal minecraft : 319
    GROUND_CLASS = 2
    GROUND_BLOCK_TOP = "minecraft:grass_block"
    GROUND_BLOCK_BELOW = "minecraft:dirt"
    GROUND_THICKNESS = 10

    # Processing parameters
    PERCENTAGE_TO_REMOVE_NON_GROUND = 0     # Decimation for non-ground features
    VOXEL_SIDE = 0.5
    MIN_POINTS_PER_VOXEL_NON_GROUND = 3     # Filtering for non-ground features
    BATCH_PER_PRODUCT_SIDE = 4              # Must be divisible by 2. Split 1 tile into BATCH_PER_PRODUCT_SIDE * BATCH_PER_PRODUCT_SIDE batches

    # Ground filling parameters
    INTERPOLATION_GRID_CELL_SIZE = 1.0      # Grid resolution for point cloud interpolation

    # Block mapping
    points_classes_name = {
        1 : "No Class", 
        2 : "Ground", 
        3 : "Small Vegetation", 
        4 : "Medium Vegetation",
        5 : "High Vegetation", 
        6 : "Building", 
        9 : "Water", 
        17: "Bridge",
        64: "Perennial Soil", 
        66: "Virtual Points", 
        67: "Miscellaneous"
    }
    choosen_template_point_classes = {
        1 : ["minecraft:stone"],
        2 : ["minecraft:grass_block"],
        3 : ["minecraft:short_grass"], 
        4 : ["minecraft:moss_block"],
        5 : ["minecraft:oak_leaves"],
        6 : ["minecraft:stone_bricks"],
        9 : ["minecraft:blue_stained_glass"],
        17: ["minecraft:polished_blackstone"],
        64: ["minecraft:dirt_path"],
        66: ["minecraft:diorite"],
        67: ["minecraft:basalt"],
    }
    

    # -------------------------- Optional: Download step ------------------------- #
    download_ign_available_tiles(lidar_tiles_available_filepath, 'point_cloud', FORCE_DOWNLOAD_LIDAR_CATALOG)
    download_ign_available_tiles(mnt_tiles_available_filepath,   'mnt',         FORCE_DOWNLOAD_LIDAR_CATALOG)

    # ----------------------------- load zone geojson ---------------------------- #
    zone_geojson = json.loads(zone_geojson_filepath.read_text())
    zone_coords = zone_geojson['features'][0]['geometry']['coordinates'][0]
    zone_shape_wgs84 = Polygon(zone_coords)

    from shapely.ops import transform
    
    wgs84 = pyproj.CRS('EPSG:4326')
    lambert93 = pyproj.CRS('EPSG:2154')

    project = pyproj.Transformer.from_crs(wgs84, lambert93, always_xy=True).transform
    zone_shape_lambert93 = transform(project, zone_shape_wgs84)

    # ----------- Scan MNT and cloud points catalog for compatible zone ---------- #

    lidar_tiles_catalog = json.loads(lidar_tiles_available_filepath.read_text())
    mnt_tiles_catalog   = json.loads(mnt_tiles_available_filepath.read_text())    

    if SEARCH_FOR_TILE_IN_ZONE:
        logger.info('Filtering the lidar tiles for the selected zone...')
        lidar_intersecting_feature = []
        for feature in tqdm(lidar_tiles_catalog['features'], desc='Filtering lidar tiles'):
            feature_shape = Polygon(feature['geometry']['coordinates'][0])
            if feature_shape.intersects(zone_shape_lambert93):
                lidar_intersecting_feature.append(feature)
        logger.info(f'Lidar tile filtering done. Found : {len(lidar_intersecting_feature)}')

        logger.info('Filtering the mnt tiles for the selected zone...')
        mnt_intersecting_feature = []
        for feature in tqdm(mnt_tiles_catalog['features'], desc='Filtering mnt tiles'):
            feature_shape = Polygon(feature['geometry']['coordinates'][0])
            if feature_shape.intersects(zone_shape_lambert93):
                mnt_intersecting_feature.append(feature)
        logger.info(f'MNT tile filtering done. Found : {len(mnt_intersecting_feature)}')

    else:
        logger.info('Using test tiles')
        lidar_test_feature = {
            'type': 'Feature', 
            'id': 'nuage-dalle.278426', 
            'geometry': {'type': 'Polygon', 'coordinates': [...]}, 
            'geometry_name': 'geom', 
            'properties': {
                'fid': 278426, 
                'name': 'LHD_FXX_1016_6293_PTS_C_LAMB93_IGN69.copc.laz', 
                'url': 'https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/RQ/LHD_FXX_1016_6293_PTS_C_LAMB93_IGN69.copc.laz'
            }, 
            'bbox': [1016000, 6292000, 1017000, 6293000]
        }
        lidar_intersecting_feature = [lidar_test_feature]

        mnt_test_feature = {
            'type': 'Feature', 
            'id': 'mnt-dalle.168575', 
            'geometry': {'type': 'Polygon', 'coordinates': [...]}, 
            'geometry_name': 'geom', 
            'properties': {
                'fid': 168575, 
                'name': 'LHD_FXX_1016_6293_MNT_O_0M50_LAMB93_IGN69.tif', 
                'url': 'https://data.geopf.fr/wms-r/LHD_FXX_1016_6293_MNT_O_0M50_LAMB93_IGN69.tif?SERVICE=WMS&VERSION=1.3.0&EXCEPTIONS=text/xml&REQUEST=GetMap&LAYERS=IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93&FORMAT=image/geotiff&STYLES=&CRS=EPSG:2154&BBOX=1015999.75,6292000.25,1016999.75,6293000.25&WIDTH=2000&HEIGHT=2000&FILENAME=LHD_FXX_1016_6293_MNT_O_0M50_LAMB93_IGN69.tif', 
                'srs': 2154
            }, 
            'bbox': [1016000, 6292000, 1017000, 6293000]
        }
        mnt_intersecting_feature = [mnt_test_feature]


    # ----------------------- Download the tiles if needed ----------------------- #

    def stream_download(url:str, output_filepath:Path, desc:str='Item Download'):
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        with open(output_filepath, 'wb') as f, tqdm(desc=desc,total=total,unit='iB',unit_scale=True,unit_divisor=1024,) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)

    for lidar_feature in tqdm(lidar_intersecting_feature, desc='Downloading lidar tiles'):
        tile_filename:str = lidar_feature['properties']['name']
        tile_url:str      = lidar_feature['properties']['url']
        tile_filepath = lidar_folderpath / tile_filename
        if not tile_filepath.exists():
            stream_download(tile_url, tile_filepath)

    for mnt_feature in tqdm(mnt_intersecting_feature, desc='Downloading mnt tiles'):
        tile_filename:str = mnt_feature['properties']['name']
        tile_url:str      = mnt_feature['properties']['url']
        tile_filepath = mnt_folderpath / tile_filename
        if not tile_filepath.exists():
            stream_download(tile_url, tile_filepath)
    

    


    # --------------------- Create compatible mnt-lidar tiles -------------------- #

    # We assume lidar and MNT have the same bbox 
    tiles = {}
    for lidar_feature in lidar_intersecting_feature:
        lidar_bbox_str  = '-'.join(map(str, lidar_feature['bbox']))
        if lidar_bbox_str not in tiles:
            tiles[lidar_bbox_str] = {}
        tiles[lidar_bbox_str]['lidar'] = {
            'filepath': lidar_folderpath / lidar_feature['properties']['name'],
            'bbox': lidar_feature['bbox']
        }

    for mnt_feature in mnt_intersecting_feature:    
        mnt_bbox_str = '-'.join(map(str, mnt_feature['bbox']))
        if mnt_bbox_str not in tiles:
            tiles[mnt_bbox_str] = {}
        tiles[mnt_bbox_str]['mnt'] = {
            'filepath': mnt_folderpath / mnt_feature['properties']['name'],
            'bbox': mnt_feature['bbox']
        }

    
    # ---------------------------------------------------------------------------- #
    #                           Loop for each tile found                           #
    # ---------------------------------------------------------------------------- #

    for tile_bbox, tile_data in tiles.items():
        logger.info(f"Processing tile with bbox: {tile_bbox}")

        # Check if both lidar and mnt are available for this tile
        if 'lidar' not in tile_data or 'mnt' not in tile_data:
            logger.warning(f"Skipping tile {tile_bbox} as it does not have both lidar and mnt data.")
            continue

        lidar_tile_filepath = Path(tile_data['lidar']['filepath'])
        mnt_tile_filepath   = Path(tile_data['mnt']['filepath'])

        # --------------------------------- Load MNT --------------------------------- #
        logger.info(f"Loading MNT file: {mnt_tile_filepath} ...")
        mnt = rasterio.open(mnt_tile_filepath)
        logger.info(f"MNT loaded")

        # --------------------------------- clean MNT -------------------------------- #
        logger.info(f"Cleaning MNT data...")
        mnt_array:np.ndarray = mnt.read(1)
        mnt_array[0] = mnt_array[1]                     # Replace first row with second row (to avoid NaN/-9999.0 issues)      
        
        
        # Pooling the MNT array to transform a mnt resolution of 0.5m to 1m
        M, N = mnt_array.shape
        K, L = 2, 2
        MK, NL = M//K, N//L
        pooled_mnt_array:np.ndarray = mnt_array.reshape(MK, K, NL, L).mean(axis=(1, 3))  # Average pooling
        mnt_array = pooled_mnt_array.astype(np.int32)          # round the values, int16 ok but strange errors in mcschematic so int32 instead
        lowest_coordinate = mnt_array.min()
        
        logger.info(f"MNT data cleaned")

        # ------------------------------ Load Lidar Data ----------------------------- #
        logger.info(f"Loading lidar file: {lidar_tile_filepath} ...")
        lidar = laspy.read(lidar_tile_filepath)
        logger.info(f"Lidar loaded | containing {len(lidar.points)} points.")


        # ------------------------- Calculate Global Z Offset ------------------------ #
        z_axis_translate:int = LOWEST_MINECRAFT_POINT - lowest_coordinate
        logger.info(f"Calculated Z translation: {z_axis_translate:.2f} (Real min Z: {lowest_coordinate:.2f} -> MC Y: {LOWEST_MINECRAFT_POINT})")


        # -------------------------- Initialize mc function -------------------------- #

        mcfunction_filepath = mcfunction_folderpath / (tile_bbox + '.mcfunction')
        text_mcfunction = '# Auto-generated MCFunction for placing lidar data\n'
        text_mcfunction += '/gamerule doDaylightCycle false\n'
        text_mcfunction += '/time set day\n'
        text_mcfunction += '/gamerule randomTickSpeed 0\n'
        text_mcfunction += '/gamemode spectator @s\n'
        text_mcfunction += '/say Starting lidar placement...\n'

        # ----------------------------- Batch Calculation ---------------------------- #
        tile_edge_size = mnt_array.shape[0] 
        batch_size = tile_edge_size // BATCH_PER_PRODUCT_SIDE

        tile_x_origin, tile_y_origin = tile_data['mnt']['bbox'][0], tile_data['mnt']['bbox'][1]

        # Transform the coordinate for the minecraft world
        mnt_array = np.rot90(mnt_array, k=1)    # Rotate the MNT array counter-clockwise by 90 degrees (a quarter turn)
        mnt_array = np.flip(mnt_array, axis=0)  # Flip the MNT array to match Minecraft coordinates (Y down)


        with logging_redirect_tqdm():
            for batch_x in tqdm(range(BATCH_PER_PRODUCT_SIDE), desc='Processing batches X axis', position=0):
                for batch_y in tqdm(range(BATCH_PER_PRODUCT_SIDE), desc='Processing batches Y axis', leave=False, position=1):

                    schem = mcschematic.MCSchematic()

                    # ------------------------ Calculate batch coordinates ----------------------- #
                    xmin_relative = batch_size * batch_x
                    xmax_relative = batch_size * (batch_x + 1)
                    ymin_relative = batch_size * batch_y
                    ymax_relative = batch_size * (batch_y + 1)                    

                    xmin_absolute = tile_x_origin + xmin_relative
                    xmax_absolute = tile_x_origin + xmax_relative
                    ymin_absolute = tile_y_origin + ymin_relative
                    ymax_absolute = tile_y_origin + ymax_relative
                   
                    # ------------------------------ MNT batch data ------------------------------ #
                    mnt_batch_array:np.ndarray = mnt_array[xmin_relative:xmax_relative, ymin_relative:ymax_relative]
                    mnt_batch_array = mnt_batch_array + z_axis_translate


                    # ------------------------ Write MNT data to schematic ----------------------- #
                    for x in tqdm(range(mnt_batch_array.shape[0]), desc='Placing MNT X', position=2, leave=False):
                        for y in range(mnt_batch_array.shape[1]):
                            z = mnt_batch_array[x, y]

                            schem.setBlock((x, z, y), GROUND_BLOCK_TOP)
                            for i in range(1, GROUND_THICKNESS+1):
                                if z-i > LOWEST_MINECRAFT_POINT:
                                    schem.setBlock((x, z-i, y), GROUND_BLOCK_BELOW)

                    schem_batch_filename = f'xmin~{xmin_absolute}_ymin~{ymin_absolute}_size~{tile_edge_size}'
                    schem.save(str(schematic_folderpath), schem_batch_filename, mcschematic.Version.JE_1_21)

                    text_mcfunction += f'\n/say Placing Batch {batch_x*BATCH_PER_PRODUCT_SIDE + batch_y + 1}/{BATCH_PER_PRODUCT_SIDE**2} at X={xmin_absolute} Z={ymin_absolute}\n'
                    text_mcfunction += f'/tp @s {xmin_absolute} 0 {ymin_absolute}\n'
                    text_mcfunction += f'//schematic load {schem_batch_filename}\n'
                    text_mcfunction += f'//paste -a\n'

        # ---------------------------- Finalize MCFunction --------------------------- #
        text_mcfunction += '\nsay Lidar placement complete!\n'
        text_mcfunction += 'gamemode creative @s\n' # Set player back to creative

        with open(mcfunction_filepath, 'w') as f: f.write(text_mcfunction)
        logger.info(f"Generated MCFunction file: {mcfunction_filepath}")
        logger.info("--- Processing Finished ---")



    exit(0)





    # ------------------------------ Main Batch Loop ----------------------------- #
    total_batches = BATCH_PER_PRODUCT_SIDE * BATCH_PER_PRODUCT_SIDE
    for index_batch, (xmin, ymin, xmax, ymax) in enumerate(batch_limit_list):
        batch_num = index_batch + 1
        logger.info(f"\n--- Processing Batch {batch_num}/{total_batches} ---")
        logger.info(f"Bounds (X): {xmin:.2f} - {xmax:.2f}, (Y): {ymin:.2f} - {ymax:.2f}")

        schem = mcschematic.MCSchematic()
        batch_points:laspy.LasData = las[(las.x<=xmax) & (las.x>=xmin) & (las.y<=ymax) & (las.y>=ymin)]

        if len(batch_points) == 0:
            logger.warning(f"Batch {batch_num} contains no points. Skipping.")
            text_mcfunction += f'# Skipping empty batch {batch_num}\n'
            continue
        
        logger.info(f"Batch {batch_num}: Found {len(batch_points)} points.")

        # Define the origin for this batch (in Minecraft coordinates)
        # Use the minimum corner, flip Y
        batch_origin_mc_x = int(np.floor(xmin)) # Convert mm to m for MC coords
        batch_origin_mc_z = int(np.floor(-ymax)) # Convert mm to m, flip Y, use MAX Y as MIN Z
        
        voxel_origin_x_m = xmin
        voxel_origin_y_m_flipped = -ymax

         # --- Store voxels temporarily per class ---
        voxels_by_class = defaultdict(set) # {class_id: set((x,y,z)), ...} -> Stores RELATIVE integer coords
        
        # --- 1a. Process Ground (Interpolation & Initial Voxelization) ---
        DO_GROUND = True
        def do_ground():
            logger.info("Processing Ground Layer (Interpolation & Voxelization)...")
            ground_mask = batch_points.classification == GROUND_CLASS
            batch_ground_points = batch_points[ground_mask]
            set_ground_voxels = set() # Store relative integer coords

            if len(batch_ground_points) > 0:
                x = batch_ground_points.x
                y = batch_ground_points.y * -1                 # Flip Y
                z = batch_ground_points.z + z_axis_translate
                xyz_ground_transformed = np.vstack([x, y, z]).T

                filled_ground_points = interpolate_point_cloud(xyz_ground_transformed, grid_cell_size=INTERPOLATION_GRID_CELL_SIZE)

                # Voxelize relative to batch origin
                points_relative_to_voxel_origin = filled_ground_points - [voxel_origin_x_m, voxel_origin_y_m_flipped, 0]

                logger.info("Voxelizing filled ground points...")
                voxel_origins_ground_relative_m = find_occupied_voxels_vectorized(
                    points_relative_to_voxel_origin,
                    voxel_size=VOXEL_SIDE,
                    min_points_per_voxel=0
                )

                

                # Add ground blocks (Grass & Dirt)
                logger.info("Creating ground blocks...")
                voxel_origins_ground_relative_m_int = voxel_origins_ground_relative_m.astype(np.int32)
                for mc_x, mc_z, mc_y in tqdm(voxel_origins_ground_relative_m_int, desc='Creating Ground'):
                    set_ground_voxels.add((int(mc_x), int(mc_y), int(mc_z)))
                    try:
                        schem.setBlock((mc_x, mc_y,   mc_z), GROUND_BLOCK_TOP  )
                        schem.setBlock((mc_x, mc_y-1, mc_z), GROUND_BLOCK_BELOW)
                        schem.setBlock((mc_x, mc_y-2, mc_z), GROUND_BLOCK_BELOW)
                        schem.setBlock((mc_x, mc_y-3, mc_z), GROUND_BLOCK_BELOW)
                        schem.setBlock((mc_x, mc_y-4, mc_z), GROUND_BLOCK_BELOW)
                        schem.setBlock((mc_x, mc_y-5, mc_z), GROUND_BLOCK_BELOW)
                        schem.setBlock((mc_x, mc_y-6, mc_z), GROUND_BLOCK_BELOW)
                        schem.setBlock((mc_x, mc_y-7, mc_z), GROUND_BLOCK_BELOW)
                    except:
                        pass
                
                # Filter lone ground block 
                logger.info("Filtering lone ground blocks...")
                nb_grass_block, nb_lone_grass_block = 0, 0
                for x, y, z in tqdm(set_ground_voxels, desc='Filtering Lone Ground Blocks'):
                    current_block_state = schem.getBlockDataAt((x,y,z))
                    if current_block_state!=GROUND_BLOCK_TOP: continue
                    nb_grass_block += 1
                    block_state_north = schem.getBlockDataAt((x,   y, z-1))
                    block_state_south = schem.getBlockDataAt((x,   y, z+1))
                    block_state_east  = schem.getBlockDataAt((x+1, y, z  ))
                    block_state_west  = schem.getBlockDataAt((x-1, y, z  ))
                    # If no neighbors, remove current block and replace dirt block below by grass
                    if (block_state_north=='minecraft:air' and 
                        block_state_south=='minecraft:air' and 
                        block_state_east =='minecraft:air' and 
                        block_state_west =='minecraft:air'):
                        nb_lone_grass_block += 1
                        schem.setBlock((x, y,   z), 'minecraft:air')
                        schem.setBlock((x, y-1, z), GROUND_BLOCK_TOP)
                logger.info(f'Total grass block : {nb_grass_block}  |  Lone grass block removed : {nb_lone_grass_block}')

            else:
                logger.warning(f"Batch {batch_num} has no ground points.")

        if DO_GROUND: do_ground()

        # --- 2. Process other points class ---
        # For each voxel, what is the main class ?
        # For each voxel, ignore the others points and get the main point class
        # For some classes, check the 1/8 of the voxels to check for partial block placement

        logger.info("Processing othher point classes")
        def coords_no_ground_points():
            point_classes_no_ground =  [1, 3, 4, 5, 6, 9, 17, 64, 66, 67]

            # Voxelize the points for each class
            point_coordinates = {point_class:list() for point_class in point_classes_no_ground}     # {1: [(x1,y1,z1),...], ...}

            for point_class in tqdm(point_classes_no_ground, desc='Voxelize points no ground'):
                mask = batch_points.classification == point_class
                batch_ground_points_no_ground = batch_points[mask]

                x = batch_ground_points_no_ground.x
                y = batch_ground_points_no_ground.y * -1                 # Flip Y
                z = batch_ground_points_no_ground.z + z_axis_translate
                xyz_no_ground = np.vstack([x, y, z]).T

                points_relative_to_voxel_origin = xyz_no_ground - [voxel_origin_x_m, voxel_origin_y_m_flipped, 0]
                
                voxel_origins_relative_m = find_occupied_voxels_vectorized(
                    points_relative_to_voxel_origin,
                    voxel_size=VOXEL_SIDE,
                    min_points_per_voxel=0
                )
                point_coordinates[point_class] = voxel_origins_relative_m

            dominant_per_voxel, filtered_points = dominant_voxel_points(point_coordinates)
            return dominant_per_voxel, filtered_points

        _, filtered_points = coords_no_ground_points()


        # Do "No Class":
        def do_No_Class(coordinates):
            bloc_type = choosen_template_point_classes[1]
            for x,y,z in coordinates:
                current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
                if current_block_state=='minecraft:air':
                    schem.setBlock((int(x),int(z),int(y)), bloc_type[0])

        def do_Small_Vegetation(coordinates):
            bloc_type = choosen_template_point_classes[3]
            for x,y,z in coordinates:
                current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
                if current_block_state==GROUND_BLOCK_TOP:
                    above_block_state = schem.getBlockDataAt((int(x),int(z+1),int(y)))
                    if above_block_state==GROUND_BLOCK_TOP:
                        continue
                    else: 
                        z += 1
                schem.setBlock((int(x),int(z),int(y)), bloc_type[0])
        
        def do_Medium_Vegetation(coordinates):
            bloc_type = choosen_template_point_classes[4]
            for x,y,z in coordinates:
                current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
                if current_block_state=='minecraft:air' or current_block_state==choosen_template_point_classes[1]:
                    schem.setBlock((int(x),int(z),int(y)), bloc_type[0])
        
        def do_High_Vegetation(coordinates):
            bloc_type = choosen_template_point_classes[5]
            for x,y,z in coordinates:
                current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
                if current_block_state=='minecraft:air' or current_block_state==choosen_template_point_classes[1]:
                    schem.setBlock((int(x),int(z),int(y)), bloc_type[0])
        
        def do_Building(coordinates):
            bloc_type = choosen_template_point_classes[6]
            for x,y,z in coordinates:
                current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
                if current_block_state=='minecraft:air' or current_block_state==choosen_template_point_classes[1]:
                    schem.setBlock((int(x),int(z),int(y)), bloc_type[0])
                # Extend the building block to the ground
                for z_below in range(int(z), LOWEST_MINECRAFT_POINT, -1):
                    below_block_state = schem.getBlockDataAt((int(x),int(z_below),int(y)))
                    if below_block_state==GROUND_BLOCK_TOP:
                        break
                    schem.setBlock((int(x),int(z_below),int(y)), bloc_type[0])

        def do_Water(coordinates):
            bloc_type = choosen_template_point_classes[9]
            for x,y,z in coordinates:
                current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
                if current_block_state=='minecraft:air' or current_block_state==choosen_template_point_classes[1]:
                    schem.setBlock((int(x),int(z),int(y)), bloc_type[0])

        def do_Bridge(coordinates):
            bloc_type = choosen_template_point_classes[17]
            for x,y,z in coordinates:
                current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
                if current_block_state=='minecraft:air' or current_block_state==choosen_template_point_classes[1]:
                    schem.setBlock((int(x),int(z),int(y)), bloc_type[0])

        def do_Perennial_Soil(coordinates):
            bloc_type = choosen_template_point_classes[64]
            for x,y,z in coordinates:
                current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
                if current_block_state=='minecraft:air' or current_block_state==choosen_template_point_classes[1]:
                    schem.setBlock((int(x),int(z),int(y)), bloc_type[0])

        def do_Virtual_Points(coordinates):
            bloc_type = choosen_template_point_classes[66]
            for x,y,z in coordinates:
                current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
                if current_block_state=='minecraft:air' or current_block_state==choosen_template_point_classes[1]:
                    schem.setBlock((int(x),int(z),int(y)), bloc_type[0])

        def do_Miscellaneous(coordinates):
            bloc_type = choosen_template_point_classes[67]
            for x,y,z in coordinates:
                current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
                if current_block_state=='minecraft:air' or current_block_state==choosen_template_point_classes[1]:
                    schem.setBlock((int(x),int(z),int(y)), bloc_type[0])


        
        do_No_Class(filtered_points[1])
        do_Small_Vegetation(filtered_points[3])
        do_Medium_Vegetation(filtered_points[4])
        do_High_Vegetation(filtered_points[5])
        do_Building(filtered_points[6])
        do_Water(filtered_points[9])
        do_Bridge(filtered_points[17])
        do_Perennial_Soil(filtered_points[64])
        do_Virtual_Points(filtered_points[66])
        do_Miscellaneous(filtered_points[67])




        # --- Save schematic ---
        schematic_filename = f"b_{batch_num}_of_{total_batches}~x_{batch_origin_mc_x}~z_{batch_origin_mc_z}"
        schematic_rel_path = f"{las_name_schem}/{schematic_filename}.schem" # Relative path for commands

        schem.save(str(folder_save_myschem), schematic_filename, mcschematic.Version.JE_1_21)
        logger.info(f"Saved schematic: {folder_save_myschem / (schematic_filename + '.schem')}")

        # --- Add Commands to MCFunction ---
        text_mcfunction += f'\n/say Placing Batch {batch_num}/{total_batches} at X={batch_origin_mc_x} Z={batch_origin_mc_z}\n'
        text_mcfunction += f'/tp @s {batch_origin_mc_x} 0 {batch_origin_mc_z}\n'
        text_mcfunction += f'//schematic load {schematic_rel_path}\n'
        text_mcfunction += f'//paste -a\n' 



    # ---------------------------- Finalize MCFunction --------------------------- #
    text_mcfunction += '\nsay Lidar placement complete!\n'
    text_mcfunction += 'gamemode creative @s\n' # Set player back to creative

    with open(mcfunction_filepath, 'w') as f:
        f.write(text_mcfunction)
    logger.info(f"Generated MCFunction file: {mcfunction_filepath}")
    logger.info("--- Processing Finished ---")