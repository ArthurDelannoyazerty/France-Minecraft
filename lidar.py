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

from scipy.interpolate import LinearNDInterpolator
# NEW: Import warnings to suppress potential division-by-zero in IDW if needed
import warnings
from collections import defaultdict


from shapely.geometry import mapping, Polygon # Added Polygon
from shapely.ops import unary_union
from shapely.vectorized import contains
from pathlib import Path
from utils.logger import setup_logging
from script import find_occupied_voxels_vectorized
from tqdm import tqdm


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


def download_ign_available_tiles(output_filepath:str, force_download:bool=False):
    # If file already exists and no force, then do not download file
    if os.path.isfile(output_filepath) and not force_download:
        logger.info(f"File {output_filepath} already exists. Skipping download.")
        return

    wfs_url = "https://data.geopf.fr/private/wfs/?service=WFS&version=2.0.0&apikey=interface_catalogue&request=GetFeature&typeNames=IGNF_LIDAR-HD_TA:nuage-dalle&outputFormat=application/json"

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
    
    # Prepare the 4 sample offsets within a cell.
    # Here we choose 4 positions placed evenly inside the cell.
    # They are offset by 1/4 and 3/4 of the cell size from the lower left corner.
    offsets = np.array([[0.25, 0.25],
                        [0.75, 0.25],
                        [0.25, 0.75],
                        [0.75, 0.75]]) * grid_cell_size
    
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
    FORCE_DOWNLOAD_ALL_TILES_AVAILABLE = False
    filepath_all_tiles_geojson = Path('data/data_grille/all_tiles_available.geojson')
    laz_folderpath = Path('data/raw_point_cloud')
    # tile_filename = 'LHD_FXX_1016_6293_PTS_C_LAMB93_IGN69.copc.laz'
    tile_filename = 'LHD_FXX_0440_6718_PTS_C_LAMB93_IGN69.copc.laz'
    tile_filepath = laz_folderpath / tile_filename
    las_name_base = tile_filepath.stem 
    
    # Minecraft parameters
    LOWEST_MINECRAFT_POINT = -60
    HIGHEST_MINECRAFT_POINT = 319
    GROUND_CLASS = 2
    GROUND_BLOCK_TOP = "minecraft:grass_block"
    GROUND_BLOCK_BELOW = "minecraft:dirt"
    GROUND_THICKNESS = 8

    # Processing parameters
    PERCENTAGE_TO_REMOVE_NON_GROUND = 0     # Decimation for non-ground features
    VOXEL_SIDE = 0.5
    MIN_POINTS_PER_VOXEL_NON_GROUND = 3     # Filtering for non-ground features
    BATCH_PER_PRODUCT_SIDE = 2              # Split 1 tile into BATCH_PER_PRODUCT_SIDE*BATCH_PER_PRODUCT_SIDE batches

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
    download_ign_available_tiles(filepath_all_tiles_geojson, FORCE_DOWNLOAD_ALL_TILES_AVAILABLE)
    # download_available_tile(filepath_all_tiles_geojson, laz_folderpath)

    # ------------------------------ Load Lidar Data ----------------------------- #
    logger.info(f"Loading lidar file: {tile_filepath}")
    try:
        las = laspy.read(tile_filepath)
        logger.info(f"Loaded {len(las.points)} points.")
    except Exception as e:
        logger.error(f"Failed to load LAS/LAZ file: {e}")
        exit(1)

    # ------------------------- Calculate Global Z Offset ------------------------ #
    lowest_coordinate = las.z.min() # Use overall min if no ground points
    z_axis_translate = LOWEST_MINECRAFT_POINT - lowest_coordinate
    logger.info(f"Calculated Z translation: {z_axis_translate:.2f} (Real min Z: {lowest_coordinate:.2f} -> MC Y: {LOWEST_MINECRAFT_POINT})")


    # -------------------------- Batch Processing Setup -------------------------- #
    las_name_schem = las_name_base.split('.')[0]
    folder_save_myschem = Path(f"data/myschems/") / las_name_schem
    folder_save_myschem.mkdir(parents=True, exist_ok=True)

    # Calculate batch boundaries
    las_x_min, las_x_max = round(las.x.min()), round(las.x.max())
    las_y_min, las_y_max = round(las.y.min()), round(las.y.max())
    las_x_len = las_x_max - las_x_min
    las_y_len = las_y_max - las_y_min # Should be similar if square tile
    batch_x_len = las_x_len / BATCH_PER_PRODUCT_SIDE
    batch_y_len = las_y_len / BATCH_PER_PRODUCT_SIDE # Use Y length for Y batches

    batch_limit_list = [
        (
            las_x_min +      j  * batch_x_len, # xmin
            las_y_min +      i  * batch_y_len, # ymin
            las_x_min + (1 + j) * batch_x_len, # xmax
            las_y_min + (1 + i) * batch_y_len  # ymax
        )
        for i in range(BATCH_PER_PRODUCT_SIDE) # Rows (Y)
        for j in range(BATCH_PER_PRODUCT_SIDE) # Columns (X)
    ]

    # ---------------------------- Generate MCFunction --------------------------- #
    mcfunction_folderpath = Path('data/mcfunctions')
    mcfunction_folderpath.mkdir(parents=True, exist_ok=True)
    mcfunction_filepath = mcfunction_folderpath / (las_name_schem + '.mcfunction')

    
    
    text_mcfunction = '# Auto-generated MCFunction for placing lidar data\n'
    text_mcfunction += '/gamemode spectator @s\n'
    text_mcfunction += '/say Starting lidar placement...\n'

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
        logger.info("Processing Ground Layer (Interpolation & Voxelization)...")
        ground_mask = batch_points.classification == GROUND_CLASS
        batch_ground_points = batch_points[ground_mask]
        potential_ground_voxels_relative = set() # Store relative integer coords

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
                min_points_per_voxel=1 # Use 1 for filled ground
            )

            # Convert float meter origins to relative integer block coordinates
            for coord_m in voxel_origins_ground_relative_m:
                mc_x = int(coord_m[0])
                mc_y = int(coord_m[2]) # Lidar Z -> MC Y
                mc_z = int(coord_m[1]) # Lidar Y (flipped) -> MC Z
                potential_ground_voxels_relative.add((mc_x, mc_y, mc_z))
                try:
                    schem.setBlock((mc_x, mc_y, mc_z),   GROUND_BLOCK_TOP)
                    schem.setBlock((mc_x, mc_y-1, mc_z), GROUND_BLOCK_BELOW)
                    schem.setBlock((mc_x, mc_y-2, mc_z), GROUND_BLOCK_BELOW)
                    schem.setBlock((mc_x, mc_y-3, mc_z), GROUND_BLOCK_BELOW)
                    schem.setBlock((mc_x, mc_y-4, mc_z), GROUND_BLOCK_BELOW)
                    schem.setBlock((mc_x, mc_y-5, mc_z), GROUND_BLOCK_BELOW)
                    schem.setBlock((mc_x, mc_y-6, mc_z), GROUND_BLOCK_BELOW)
                    schem.setBlock((mc_x, mc_y-7, mc_z), GROUND_BLOCK_BELOW)
                except:
                    pass
        else:
            logger.warning(f"Batch {batch_num} has no ground points.")



        schematic_filename = f"b_{batch_num}_of_{total_batches}~x_{batch_origin_mc_x}~z_{batch_origin_mc_z}"
        schematic_rel_path = f"{las_name_schem}/{schematic_filename}.schem" # Relative path for commands

        schem.save(str(folder_save_myschem), schematic_filename, mcschematic.Version.JE_1_21_1)
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