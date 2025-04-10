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




def process_point_cloud(points, cell_size=0.5):
    """
    Processes a 3D point cloud by:
      1. Creating a 2D grid that covers the full XY extent of the point cloud.
      2. Flattening the point cloud onto that grid and flagging the empty cells.
      3. For each empty cell, adding 4 new points located at evenly-spaced positions inside the cell.
         Their z-values are obtained via a linear interpolation using the original points.
    
    Parameters:
      points (np.ndarray): Nx3 numpy array of points (x, y, z).
      cell_size (float): grid cell size (default 0.5 meters).
    
    Returns:
      np.ndarray: The augmented array of points (original + the new points).
    """
    # display_point_cloud(points)

    # Determine grid bounds (using the points' x-y projection)
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)
    
    # Determine the number of cells required in each dimension (ceiling)
    n_cells_x = int(np.floor((x_max - x_min) / cell_size))
    n_cells_y = int(np.floor((y_max - y_min) / cell_size))
    
    # Create an occupancy grid that marks cells with any point present.
    # Compute cell indices (i for x, j for y) for each point.
    ix = ((points[:, 0] - x_min) / cell_size).astype(np.int32)
    iy = ((points[:, 1] - y_min) / cell_size).astype(np.int32)

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
                        [0.75, 0.75]]) * cell_size
    
    # Compute the lower left coordinate for each empty cell.
    empty_cell_origin = np.empty((empty_cells.shape[0], 2))
    empty_cell_origin[:, 0] = x_min + empty_cells[:, 0] * cell_size
    empty_cell_origin[:, 1] = y_min + empty_cells[:, 1] * cell_size
    
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
    # Select *one* tile for processing in this example
    # TODO: Add logic here to select tile based on order, intersection etc.
    # For now, hardcode a tile known to exist after download step.
    tile_filename = 'LHD_FXX_1016_6293_PTS_C_LAMB93_IGN69.copc.laz' # Example filename
    tile_filepath = laz_folderpath / tile_filename
    las_name_base = tile_filepath.stem # Use stem for cleaner naming

    # Minecraft parameters
    LOWEST_MINECRAFT_POINT = -60
    HIGHEST_MINECRAFT_POINT = 319 # Not actively used for clipping here, but good to have
    GROUND_CLASS = 2
    GROUND_BLOCK_TOP = "minecraft:grass_block"
    GROUND_BLOCK_BELOW = "minecraft:dirt"
    GROUND_THICKNESS = 8 # Total thickness including top layer

    # Processing parameters
    PERCENTAGE_TO_REMOVE_NON_GROUND = 0 # Decimation for non-ground features
    VOXEL_SIDE = 1.0 # Minecraft block size = 1 meter
    MIN_POINTS_PER_VOXEL_NON_GROUND = 3 # Filtering for non-ground features
    BATCH_PER_PRODUCT_SIDE = 5 # Split 1km tile into 5x5 = 25 batches

    # Ground filling parameters
    INTERPOLATION_GRID_SIZE = 1.0 # Resolution for finding holes (meters)
    INTERPOLATION_K_NEIGHBORS = 8   # How many neighbors to use for IDW
    INTERPOLATION_SEARCH_FACTOR = 3.0 # Search radius = factor * grid_size
    INTERPOLATION_IDW_POWER = 2

    # Block mapping (using the 'natural' template from the user)
    order_bloc_creation = [66, 64, 17, 9, 5, 4, 3, 67, 1, 6] # Removed ground (2) - handled separately
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
        1 : ["minecraft:stone"], # No Class
       # 2 : ["minecraft:grass_block"], # Ground - Handled separately
        3 : ["minecraft:short_grass"], # Small Veg
        4 : ["minecraft:moss_block"],  # Medium Veg
        5 : ["minecraft:oak_leaves"],  # High Veg
        6 : ["minecraft:stone_bricks"],# Building
        9 : ["minecraft:blue_stained_glass"], # Water
        17: ["minecraft:polished_blackstone"], # Bridge
        64: ["minecraft:dirt_path"],   # Perennial Soil
        66: ["minecraft:diorite"],     # Virtual Points
        67: ["minecraft:basalt"],      # Miscellaneous
    }


    # -------------------------- Optional: Download step ------------------------- #
    filepath_all_tiles_geojson         = Path('data/data_grille/all_tiles_available.geojson')
    download_ign_available_tiles(filepath_all_tiles_geojson, FORCE_DOWNLOAD_ALL_TILES_AVAILABLE)

    laz_folderpath = Path('data/raw_point_cloud')
    # download_available_tile(filepath_all_tiles_geojson, laz_folderpath)

    tile_filepath = Path('data/raw_point_cloud/LHD_FXX_1016_6293_PTS_C_LAMB93_IGN69.copc.laz')
    las = laspy.read(tile_filepath)


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

    # Calculate batch boundaries (using original scale, before meters conversion)
    las_x_min, las_x_max = las.x.min(), las.x.max()
    las_y_min, las_y_max = las.y.min(), las.y.max()
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
    mcfunction_filepath = Path('data/mcfunctions') / (las_name_schem + '.mcfunction')
    text_mcfunction = '# Auto-generated MCFunction for placing lidar data\n'
    text_mcfunction += '/gamemode spectator @s\n' # Use @s for safety
    text_mcfunction += '/say Starting lidar placement...\n'

    # ------------------------------ Main Batch Loop ----------------------------- #
    total_batches = BATCH_PER_PRODUCT_SIDE * BATCH_PER_PRODUCT_SIDE
    for index_batch, (xmin, ymin, xmax, ymax) in enumerate(batch_limit_list):
        batch_num = index_batch + 1
        logger.info(f"\n--- Processing Batch {batch_num}/{total_batches} ---")
        logger.info(f"Bounds (X): {xmin:.2f} - {xmax:.2f}, (Y): {ymin:.2f} - {ymax:.2f}")

        batch_points = las[(las.x<=xmax) & (las.x>=xmin) & (las.y<=ymax) & (las.y>=ymin)]

        if len(batch_points) == 0:
            logger.warning(f"Batch {batch_num} contains no points. Skipping.")
            text_mcfunction += f'# Skipping empty batch {batch_num}\n'
            continue

        logger.info(f"Batch {batch_num}: Found {len(batch_points)} points.")

        # Define the origin for this batch (in Minecraft coordinates)
        # Use the minimum corner, flip Y
        batch_origin_mc_x = int(np.floor(xmin)) # Convert mm to m for MC coords
        batch_origin_mc_z = int(np.floor(-ymax)) # Convert mm to m, flip Y, use MAX Y as MIN Z
        # Note: We will calculate voxel positions relative to this origin.

        # Use a dictionary to store final voxel coordinates and block types for this batch
        # Key: (x, y, z) tuple relative to schematic origin, Value: block_string
        # Minecraft Y corresponds to Lidar Z
        final_voxels = {}

        # -------------------------- 1. Process Ground Layer ------------------------- #
        logger.info("Processing Ground Layer...")
        batch_ground_points = batch_points.points[batch_points.classification == GROUND_CLASS]

        if len(batch_ground_points) > 0:
            # Transform coordinates to meters, apply Z offset, flip Y
            x =  batch_ground_points.x.array / 100.0
            y =  batch_ground_points.y.array / 100.0 * -1 # Flip Y
            z = (batch_ground_points.z.array / 100.0) + z_axis_translate
            xyz_ground_transformed = np.vstack([x, y, z]).T


            # Apply hole filling
            logger.info("Applying ground hole filling...")
            filled_ground_points = process_point_cloud(
                xyz_ground_transformed,
            )

            logger.info("Voxelizing filled ground points...")
            # Use min_points_per_voxel=1 for filled ground
            voxel_origins_ground = find_occupied_voxels_vectorized(
                filled_ground_points,
                voxel_size=VOXEL_SIDE,
                min_points_per_voxel=1 # IMPORTANT: Use 1 for filled ground
            )
            voxel_origins_ground_relative = voxel_origins_ground - [xmin, -ymin, 0]

            logger.info(f"Found {voxel_origins_ground_relative.shape[0]} ground voxels.")

            # Add ground voxels to the final list
            for coord in voxel_origins_ground_relative:
                 # Convert float origins to integer block coordinates
                 mc_x = int(coord[0])
                 mc_y = int(coord[2]) # Lidar Z -> MC Y
                 mc_z = int(coord[1]) # Lidar Y -> MC Z

                 # Add Grass Layer
                 final_voxels[(mc_x, mc_y, mc_z)] = GROUND_BLOCK_TOP

                 # Add Dirt Layers Below (up to GROUND_THICKNESS)
                 for i in range(1, GROUND_THICKNESS):
                     dirt_coord = (mc_x, mc_y - i, mc_z)
                     # Ensure dirt doesn't go below world limit
                     if dirt_coord[1] >= LOWEST_MINECRAFT_POINT:
                        # Only add dirt if the space isn't already taken by grass from a lower elevation
                        # (though this shouldn't happen if ground is mostly surface)
                        if dirt_coord not in final_voxels:
                           final_voxels[dirt_coord] = GROUND_BLOCK_BELOW
                     else:
                         break # Stop going deeper if below world limit

        else:
            logger.warning(f"Batch {batch_num} has no ground points. Ground layer will be empty.")


        # -------------------------- 2. Process Other Layers ------------------------- #
        logger.info("Processing Other Layers...")
        for point_class in order_bloc_creation: # Iterate through non-ground classes
            class_name = points_classes_name.get(point_class, f"Unknown ({point_class})")
            logger.debug(f"Processing class: {class_name} ({point_class})")

            batch_class_points = batch_points.points[batch_points.classification == point_class]

            if len(batch_class_points) == 0:
                logger.debug(f"No points for class {class_name} in this batch.")
                continue

            # Transform coordinates
            x =  batch_class_points.x.array / 100.0
            y =  batch_class_points.y.array / 100.0 * -1 # Flip Y
            z = (batch_class_points.z.array / 100.0) + z_axis_translate
            xyz_class_transformed = np.vstack([x, y, z]).T

            # Optional Decimation for non-ground
            if PERCENTAGE_TO_REMOVE_NON_GROUND > 0:
                 xyz_class_transformed = decimate_array(xyz_class_transformed, PERCENTAGE_TO_REMOVE_NON_GROUND)
                 if xyz_class_transformed.shape[0] == 0:
                     logger.debug(f"All points decimated for class {class_name}.")
                     continue


            voxel_origins_class = find_occupied_voxels_vectorized(
                xyz_class_transformed,
                voxel_size=VOXEL_SIDE,
                min_points_per_voxel=MIN_POINTS_PER_VOXEL_NON_GROUND
            )
            voxel_origins_class_relative = voxel_origins_class - [xmin, -ymin, 0]

            block = choosen_template_point_classes[point_class][0] # Get block type
            logger.debug(f"Found {voxel_origins_class_relative.shape[0]} voxels for class {class_name}.")

            # Add voxels, potentially overwriting ground/lower layers
            for coord in voxel_origins_class_relative:
                mc_x = int(np.floor(coord[0] / VOXEL_SIDE))
                mc_y = int(np.floor(coord[2] / VOXEL_SIDE)) # Lidar Z -> MC Y
                mc_z = int(np.floor(coord[1] / VOXEL_SIDE)) # Lidar Y -> MC Z
                voxel_coord = (mc_x, mc_y, mc_z)

                # Check if coordinate is within valid MC height range
                if LOWEST_MINECRAFT_POINT <= mc_y < HIGHEST_MINECRAFT_POINT + 1 : # +1 because range is exclusive at top
                    final_voxels[voxel_coord] = block
                # else:
                    # logger.debug(f"Skipping voxel at {voxel_coord} (Y={mc_y}) - outside valid height range.")


        # ----------------------- 3. Create and Save Schematic ----------------------- #
        if not final_voxels:
            logger.warning(f"Batch {batch_num} resulted in no voxels. Skipping schematic generation.")
            text_mcfunction += f'# Skipping empty batch {batch_num} schematic\n'
            continue

        logger.info(f"Creating schematic for batch {batch_num} with {len(final_voxels)} blocks.")
        schem = mcschematic.MCSchematic()
        for (vx, vy, vz), block_id in final_voxels.items():
            # Coordinates are already relative to the schematic origin
            schem.setBlock((vx, vy, vz), block_id)

        # Naming convention: batch number, total batches, and MC origin coords
        schematic_filename = f"b_{batch_num}_of_{total_batches}~x_{batch_origin_mc_x}~z_{batch_origin_mc_z}"
        schematic_rel_path = f"{las_name_schem}/{schematic_filename}.schem" # Relative path for commands

        # Try saving the schematic
        try:
            schem.save(str(folder_save_myschem), schematic_filename, mcschematic.Version.JE_1_21_1)
            logger.info(f"Saved schematic: {folder_save_myschem / (schematic_filename + '.schem')}")

            # ----------------------- 4. Add Commands to MCFunction ---------------------- #
            # Teleport command to the batch origin (bottom-north-west corner in MC coords)
            # Add a height offset so player isn't inside the ground immediately
            tp_y = LOWEST_MINECRAFT_POINT + 10 # Adjust as needed
            # Find the lowest ground point Y in this batch to teleport above it?
            ground_y_values = [vy for (vx, vy, vz), block in final_voxels.items() if block in [GROUND_BLOCK_TOP, GROUND_BLOCK_BELOW]]
            if ground_y_values:
                tp_y = max(ground_y_values) + 5 # TP 5 blocks above highest ground/dirt block
            tp_y = max(tp_y, LOWEST_MINECRAFT_POINT + 5) # Ensure it's not below world


            text_mcfunction += f'\n/say Placing Batch {batch_num}/{total_batches} at X={batch_origin_mc_x} Z={batch_origin_mc_z}\n'
            text_mcfunction += f'/tp @s {batch_origin_mc_x} {tp_y} {batch_origin_mc_z}\n'
            text_mcfunction += f'//schematic load {schematic_rel_path}\n'
            text_mcfunction += f'//paste -a\n' 

        except Exception as e:
             logger.error(f"Failed to save schematic or generate commands for batch {batch_num}: {e}")
             text_mcfunction += f'# ERROR: Failed to save schematic for batch {batch_num}\n'


    # ---------------------------- Finalize MCFunction --------------------------- #
    text_mcfunction += '\nsay Lidar placement complete!\n'
    text_mcfunction += 'gamemode creative @s\n' # Set player back to creative

    with open(mcfunction_filepath, 'w') as f:
        f.write(text_mcfunction)
    logger.info(f"Generated MCFunction file: {mcfunction_filepath}")
    logger.info("--- Processing Finished ---")