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

from shapely.geometry import mapping
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
    if os.path.isfile(output_filepath) and not force_download: return

    wfs_url = "https://data.geopf.fr/private/wfs/?service=WFS&version=2.0.0&apikey=interface_catalogue&request=GetFeature&typeNames=IGNF_LIDAR-HD_TA:nuage-dalle&outputFormat=application/json"
    
    # First request to initialize the geojson and know the total number of features
    logger.info('First download for all features available')
    response = requests.get(wfs_url)
    if response.status_code != 200:
        logger.info(f"Failed to retrieve data for rul : {wfs_url}. HTTP Status code: {response.status_code}")
        exit(1)
    
    geojson = response.json()
    total_features = geojson['totalFeatures']
    number_returned = geojson['numberReturned']
    logger.info(f'First download finished. Total Feature : {total_features}  | Number Returned : {number_returned}')

    start_index = number_returned
    while start_index<total_features:
        logger.info(f'Downloading features from index {start_index} / {total_features}')
        wfs_url_indexed = f'{wfs_url}&startIndex={start_index}'
        response = requests.get(wfs_url_indexed)
        if response.status_code != 200:
            logger.info(f"Failed to retrieve data for rul : {wfs_url_indexed}. HTTP Status code: {response.status_code}")
            exit(1)
        response = response.json()
        current_features = response['features']
        geojson['features'].extend(current_features)
        number_returned = geojson['numberReturned']
        start_index += number_returned
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=4, ensure_ascii=False)


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



if __name__=='__main__':
    log_name = Path(__file__).stem
    setup_logging(log_name)
    logger = logging.getLogger(__name__)

    init_folders()

    # Download and merge all tiles availables
    FORCE_DOWNLOAD_ALL_TILES_AVAILABLE = False
    filepath_all_tiles_geojson         = Path('data/data_grille/all_tiles_available.geojson')
    download_ign_available_tiles(filepath_all_tiles_geojson, FORCE_DOWNLOAD_ALL_TILES_AVAILABLE)

    laz_folderpath = Path('data/raw_point_cloud')
    # download_available_tile(filepath_all_tiles_geojson, laz_folderpath)

    tile_filepath = Path('data/raw_point_cloud/LHD_FXX_1016_6293_PTS_C_LAMB93_IGN69.copc.laz')
    las = laspy.read(tile_filepath)
    

    points_classes_name = {
        1 : "No Class",
        2 : "Ground",
        3 : "Small Vegetation (0-50cm)",
        4 : "Medium Vegetation (50-150 cm)",
        5 : "High Vegetation (+150 cm)",
        6 : "Building",
        9 : "Water",
        17: "Bridge",
        64: "Perennial Soil",
        66: "Virtual Points",
        67: "Miscellaneous"
    }
    template_points_classes_full_stone = {
        1 : ["minecraft:stone"],
        2 : ["minecraft:stone"],
        3 : ["minecraft:stone"],
        4 : ["minecraft:stone"],
        5 : ["minecraft:stone"],
        6 : ["minecraft:stone"],
        9 : ["minecraft:stone"],
        17: ["minecraft:stone"],
        64: ["minecraft:stone"],
        66: ["minecraft:stone"],
        67: ["minecraft:stone"],
    }
    template_points_classes_wool = {
        # 1 : ["minecraft:black_wool"],
        2 : ["minecraft:brown_wool"],
        3 : ["minecraft:lime_wool"],
        4 : ["minecraft:green_wool"],
        5 : ["minecraft:cyan_wool"],
        6 : ["minecraft:gray_wool"],
        9 : ["minecraft:blue_wool"],
        # 17: ["minecraft:purple_wool"],
        # 64: ["minecraft:yellow_wool"],
        # 66: ["minecraft:pink_wool"],
        # 67: ["minecraft:magenta_wool"],
    }
    template_points_classes_wool_old = {
        1 : ["minecraft:wool 15"],
        2 : ["minecraft:wool 12"],
        3 : ["minecraft:wool 5"],
        4 : ["minecraft:wool 13"],
        5 : ["minecraft:wool 9"],
        6 : ["minecraft:wool 7"],
        9 : ["minecraft:wool 11"],
        17: ["minecraft:wool 10"],
        64: ["minecraft:wool 4"],
        66: ["minecraft:wool 6"],
        67: ["minecraft:wool 2"],
    }

    choosen_template_point_classes = template_points_classes_wool
    
    LOWEST_MINECRAFT_POINT = -60
    HIGHEST_MINECRAFT_POINT = 319
    lowest_coordinate  = las.xyz[:,2].min()
    highest_coordinate = las.xyz[:,2].max()

    z_axis_translate = LOWEST_MINECRAFT_POINT - lowest_coordinate


    PERCENTAGE_TO_REMOVE = 40
    VOXEL_SIDE = 1
    BATCH_PER_PRODUCT_SIDE = 4

    las_x_length = las.x.max() - las.x.min()
    las_y_length = las.y.max() - las.y.min()

    for batch in range(BATCH_PER_PRODUCT_SIDE):
        x_start = las.x.min() +   batch    * (las_x_length/BATCH_PER_PRODUCT_SIDE)
        x_end   = las.x.min() +  (batch+1) * (las_x_length/BATCH_PER_PRODUCT_SIDE)
        y_start = las.y.min() +   batch    * (las_y_length/BATCH_PER_PRODUCT_SIDE)
        y_end   = las.y.min() +  (batch+1) * (las_y_length/BATCH_PER_PRODUCT_SIDE)
    
        batch_las = las[(las.x<=x_end) & (las.x>=x_start) & (las.y<=y_end) & (las.y>=y_start)]

        schem = mcschematic.MCSchematic()
        for point_class in choosen_template_point_classes.keys():
            print(f'Processing point class : {point_class}')

            x = las.points[las.classification == point_class].x.array / 100
            y = las.points[las.classification == point_class].y.array / 100
            z = las.points[las.classification == point_class].z.array / 100 + z_axis_translate
            xyz = np.vstack([x,y,z]).T

            if len(xyz)==0: continue
            print(f'Point : {xyz[0]}')

            xyz = decimate_array(xyz, PERCENTAGE_TO_REMOVE)

            voxel_origins_opt = find_occupied_voxels_vectorized(xyz, voxel_size=VOXEL_SIDE)
            # voxel_origins_opt_centered = voxel_origins_opt - voxel_origins_opt[0] 

            A = voxel_origins_opt
            B = np.array([0,0])
            R = 30
            C = A[np.linalg.norm(A[:,:2] - B, axis=1) > R]

            LIMIT = 100
            i=0
            block = choosen_template_point_classes[point_class][0]
            for coord in tqdm(C):
                if i>LIMIT: break
                i+= 1
                schem.setBlock( (int(coord[0]), int(coord[2]), int(coord[1])), block)
    
        schem.save(f"data/myschems", f"test_schematic{batch}", mcschematic.Version.JE_1_21_1)