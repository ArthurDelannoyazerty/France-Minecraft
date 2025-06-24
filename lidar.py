import rasterio
import laspy
import logging
import os
import requests
import json
import numpy as np
import mcschematic
import pyproj
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
# NEW: Import warnings to suppress potential division-by-zero in IDW if needed


from shapely.geometry import Polygon, shape
from pathlib import Path
from utils.logger import setup_logging
from script import find_occupied_voxels_vectorized
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


def init_folders():
    logger.info('Create folders for the project')
    Path('data/data_grille'    ).mkdir(parents=True, exist_ok=True)
    Path('data/orders'         ).mkdir(parents=True, exist_ok=True)
    Path('data/raw_point_cloud').mkdir(parents=True, exist_ok=True)
    Path('data/logs'           ).mkdir(parents=True, exist_ok=True)
    Path('data/myschems'       ).mkdir(parents=True, exist_ok=True)
    Path('data/mcfunctions'    ).mkdir(parents=True, exist_ok=True)



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





def do_No_Class(coordinates, choosen_template_point_classes:dict[int, list[str]], schem:mcschematic.MCSchematic):
    bloc_type = choosen_template_point_classes[1]
    for x,y,z in coordinates:
        current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
        if current_block_state=='minecraft:air':
            schem.setBlock((int(x),int(z),int(y)), bloc_type[0])

def do_Small_Vegetation(coordinates, choosen_template_point_classes:dict[int, list[str]], schem:mcschematic.MCSchematic):
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

def do_Medium_Vegetation(coordinates, choosen_template_point_classes:dict[int, list[str]], schem:mcschematic.MCSchematic):
    bloc_type = choosen_template_point_classes[4]
    for x,y,z in coordinates:
        current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
        if current_block_state=='minecraft:air' or current_block_state==choosen_template_point_classes[1]:
            schem.setBlock((int(x),int(z),int(y)), bloc_type[0])

def do_High_Vegetation(coordinates, choosen_template_point_classes:dict[int, list[str]], schem:mcschematic.MCSchematic):
    bloc_type = choosen_template_point_classes[5]
    for x,y,z in coordinates:
        current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
        if current_block_state=='minecraft:air' or current_block_state==choosen_template_point_classes[1]:
            schem.setBlock((int(x),int(z),int(y)), bloc_type[0])

def do_Building(coordinates, choosen_template_point_classes:dict[int, list[str]], schem:mcschematic.MCSchematic):
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

def do_Water(coordinates, choosen_template_point_classes:dict[int, list[str]], schem:mcschematic.MCSchematic):
    bloc_type = choosen_template_point_classes[9]
    for x,y,z in coordinates:
        current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
        if current_block_state=='minecraft:air' or current_block_state==choosen_template_point_classes[1]:
            schem.setBlock((int(x),int(z),int(y)), bloc_type[0])

def do_Bridge(coordinates, choosen_template_point_classes:dict[int, list[str]], schem:mcschematic.MCSchematic):
    bloc_type = choosen_template_point_classes[17]
    for x,y,z in coordinates:
        current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
        if current_block_state=='minecraft:air' or current_block_state==choosen_template_point_classes[1]:
            schem.setBlock((int(x),int(z),int(y)), bloc_type[0])

def do_Perennial_Soil(coordinates, choosen_template_point_classes:dict[int, list[str]], schem:mcschematic.MCSchematic):
    bloc_type = choosen_template_point_classes[64]
    for x,y,z in coordinates:
        current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
        if current_block_state=='minecraft:air' or current_block_state==choosen_template_point_classes[1]:
            schem.setBlock((int(x),int(z),int(y)), bloc_type[0])

def do_Virtual_Points(coordinates, choosen_template_point_classes:dict[int, list[str]], schem:mcschematic.MCSchematic):
    bloc_type = choosen_template_point_classes[66]
    for x,y,z in coordinates:
        current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
        if current_block_state=='minecraft:air' or current_block_state==choosen_template_point_classes[1]:
            schem.setBlock((int(x),int(z),int(y)), bloc_type[0])

def do_Miscellaneous(coordinates, choosen_template_point_classes:dict[int, list[str]], schem:mcschematic.MCSchematic):
    bloc_type = choosen_template_point_classes[67]
    for x,y,z in coordinates:
        current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
        if current_block_state=='minecraft:air' or current_block_state==choosen_template_point_classes[1]:
            schem.setBlock((int(x),int(z),int(y)), bloc_type[0])






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

    zone_geojson_filepath = Path('data/zone_test_2.geojson')

    lidar_folderpath      = Path('data/tiles/lidar/')
    mnt_folderpath        = Path('data/tiles/mnt')
    schematic_folderpath  = Path('data/myschems')
    mcfunction_folderpath = Path('data/mcfunctions')

    SEARCH_FOR_TILE_IN_ZONE = False


    # Minecraft parameters
    MANUAL_Z_AXIS_TRANSLATE = True  # If True, you must set LOWEST_MINECRAFT_POINT to the lowest point of the MNT
    MANUAL_Z_AXIS_TRANSLATE_VALUE = -2000  # If MANUAL_Z_AXIS_TRANSLATE is True, this value will be used to translate the Z axis of the MNT to the Minecraft world
    LOWEST_MINECRAFT_POINT = -2031          # If normal minecraft : -60
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
        
        lidar_intersecting_feature = [{'bbox': [1015000, 6292000, 1016000, 6293000],
            'geometry': {'coordinates': [[[1016000, 6292000],
                                            [1016000, 6293000],
                                            [1015000, 6293000],
                                            [1015000, 6292000],
                                            [1016000, 6292000]]],
                        'type': 'Polygon'},
            'geometry_name': 'geom',
            'id': 'nuage-dalle.278401',
            'properties': {'fid': 278401,
                            'name': 'LHD_FXX_1015_6293_PTS_C_LAMB93_IGN69.copc.laz',
                            'url': 'https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/RQ/LHD_FXX_1015_6293_PTS_C_LAMB93_IGN69.copc.laz'},
            'type': 'Feature'},
            {'bbox': [1016000, 6292000, 1017000, 6293000],
            'geometry': {'coordinates': [[[1017000, 6292000],
                                            [1017000, 6293000],
                                            [1016000, 6293000],
                                            [1016000, 6292000],
                                            [1017000, 6292000]]],
                        'type': 'Polygon'},
            'geometry_name': 'geom',
            'id': 'nuage-dalle.278426',
            'properties': {'fid': 278426,
                            'name': 'LHD_FXX_1016_6293_PTS_C_LAMB93_IGN69.copc.laz',
                            'url': 'https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/RQ/LHD_FXX_1016_6293_PTS_C_LAMB93_IGN69.copc.laz'},
            'type': 'Feature'}]

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
        mnt_intersecting_feature = [{'bbox': [1015000, 6292000, 1016000, 6293000],
            'geometry': {'coordinates': [[[1016000, 6292000],
                                            [1016000, 6293000],
                                            [1015000, 6293000],
                                            [1015000, 6292000],
                                            [1016000, 6292000]]],
                        'type': 'Polygon'},
            'geometry_name': 'geom',
            'id': 'mnt-dalle.168029',
            'properties': {'fid': 168029,
                            'name': 'LHD_FXX_1015_6293_MNT_O_0M50_LAMB93_IGN69.tif',
                            'srs': 2154,
                            'url': 'https://data.geopf.fr/wms-r/LHD_FXX_1015_6293_MNT_O_0M50_LAMB93_IGN69.tif?SERVICE=WMS&VERSION=1.3.0&EXCEPTIONS=text/xml&REQUEST=GetMap&LAYERS=IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93&FORMAT=image/geotiff&STYLES=&CRS=EPSG:2154&BBOX=1014999.75,6292000.25,1015999.75,6293000.25&WIDTH=2000&HEIGHT=2000&FILENAME=LHD_FXX_1015_6293_MNT_O_0M50_LAMB93_IGN69.tif'},
            'type': 'Feature'},
            {'bbox': [1016000, 6292000, 1017000, 6293000],
            'geometry': {'coordinates': [[[1017000, 6292000],
                                            [1017000, 6293000],
                                            [1016000, 6293000],
                                            [1016000, 6292000],
                                            [1017000, 6292000]]],
                        'type': 'Polygon'},
            'geometry_name': 'geom',
            'id': 'mnt-dalle.168575',
            'properties': {'fid': 168575,
                            'name': 'LHD_FXX_1016_6293_MNT_O_0M50_LAMB93_IGN69.tif',
                            'srs': 2154,
                            'url': 'https://data.geopf.fr/wms-r/LHD_FXX_1016_6293_MNT_O_0M50_LAMB93_IGN69.tif?SERVICE=WMS&VERSION=1.3.0&EXCEPTIONS=text/xml&REQUEST=GetMap&LAYERS=IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93&FORMAT=image/geotiff&STYLES=&CRS=EPSG:2154&BBOX=1015999.75,6292000.25,1016999.75,6293000.25&WIDTH=2000&HEIGHT=2000&FILENAME=LHD_FXX_1016_6293_MNT_O_0M50_LAMB93_IGN69.tif'},
            'type': 'Feature'}]


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
        try:
            bbox = lidar_feature['bbox']
        except:
            bbox = shape(lidar_feature['geometry']).bounds
            bbox = [int(e) for e in bbox]

        lidar_bbox_str  = '-'.join(map(str, bbox))
        if lidar_bbox_str not in tiles:
            tiles[lidar_bbox_str] = {}
        tiles[lidar_bbox_str]['lidar'] = {
            'filepath': lidar_folderpath / lidar_feature['properties']['name'],
            'bbox': bbox
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
        logger.info("MNT loaded")

        # --------------------------------- clean MNT -------------------------------- #
        logger.info("Cleaning MNT data...")
        mnt_array:np.ndarray = mnt.read(1)
        mnt_array[0][0] = -9999.0
        
        def replace_errors_with_neighbor_mean(arr):
            """Replace -9999.0 values with mean of valid neighbors (8-connected)."""
            result = arr.copy()
            error_mask = (arr == -9999.0)
            
            for i, j in np.argwhere(error_mask):
                # Get 3x3 neighborhood, handling boundaries
                neighbors = arr[max(0, i-1):i+2, max(0, j-1):j+2]
                # Get valid neighbors (not -9999.0 and not the center cell)
                valid = neighbors[(neighbors != -9999.0)]
                if len(valid) > 1:  # Exclude center cell
                    valid = valid[valid != arr[i, j]]  # Remove center if it somehow got included
                
                if len(valid) > 0:
                    result[i, j] = np.mean(valid)
            return result
        
        
        mnt_array = replace_errors_with_neighbor_mean(mnt_array)

        # Pooling the MNT array to transform a mnt resolution of 0.5m to 1m
        M, N = mnt_array.shape
        K, L = 2, 2
        MK, NL = M//K, N//L
        pooled_mnt_array:np.ndarray = mnt_array.reshape(MK, K, NL, L).mean(axis=(1, 3))  # Average pooling
        mnt_array = pooled_mnt_array.astype(np.int32)          # round the values, int16 ok but strange errors in mcschematic so int32 instead
        lowest_coordinate = mnt_array.min()

        # Transform the coordinate for the minecraft world
        mnt_array = mnt_array.T
        
        logger.info("MNT data cleaned")

        # ------------------------------ Load Lidar Data ----------------------------- #
        logger.info(f"Loading lidar file: {lidar_tile_filepath} ...")
        lidar = laspy.read(lidar_tile_filepath)
        logger.info(f"Lidar loaded | containing {len(lidar.points)} points.")


        # -------------------------------- Clean Lidar ------------------------------- #

        tile_min_x, tile_min_y, tile_max_x, tile_max_y = tile_data['lidar']['bbox']
        lidar.x = np.array(lidar.x) - tile_min_x
        lidar.y = (-np.array(lidar.y) + tile_min_y + (tile_max_y - tile_min_y))



        # ------------------------- Calculate Global Z Offset ------------------------ #
        if MANUAL_Z_AXIS_TRANSLATE:
            z_axis_translate = MANUAL_Z_AXIS_TRANSLATE_VALUE
        else:
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


        for batch_x in tqdm(range(BATCH_PER_PRODUCT_SIDE), desc='Processing batches X axis', position=0):
            for batch_y in tqdm(range(BATCH_PER_PRODUCT_SIDE), desc='Processing batches Y axis', leave=False, position=1):

                schem = mcschematic.MCSchematic()

                # ------------------------ Calculate batch coordinates ----------------------- #
                xmin_relative = batch_size * batch_x
                xmax_relative = batch_size * (batch_x + 1)
                ymin_relative = batch_size * batch_y
                ymax_relative = batch_size * (batch_y + 1)                    

                xmin_absolute = tile_min_x + xmin_relative
                xmax_absolute = tile_min_x + xmax_relative
                ymin_absolute = tile_min_y + ymin_relative
                ymax_absolute = tile_min_y + ymax_relative

                # ------------------------------ MNT batch data ------------------------------ #
                mnt_batch_array:np.ndarray = mnt_array[xmin_relative:xmax_relative, ymin_relative:ymax_relative]
                mnt_batch_array = mnt_batch_array + z_axis_translate


                # ------------------------ Write MNT data to schematic ----------------------- #
                for x in tqdm(range(mnt_batch_array.shape[0]), desc='Placing MNT block batch', leave=False):
                    for y in range(mnt_batch_array.shape[1]):
                        z = mnt_batch_array[x, y]

                        schem.setBlock((x, z, y), GROUND_BLOCK_TOP)
                        for i in range(1, GROUND_THICKNESS+1):
                            if z-i > LOWEST_MINECRAFT_POINT:
                                schem.setBlock((x, z-i, y), GROUND_BLOCK_BELOW)
                

                # ----------------------------- Lidar batch data ----------------------------- #
                lidar_batch:laspy.LasData = lidar[(lidar.x<=xmax_relative) & 
                                                   (lidar.x>=xmin_relative) & 
                                                   (lidar.y<=ymax_relative) & 
                                                   (lidar.y>=ymin_relative)]
                
                point_classes_no_ground =  [1, 3, 4, 5, 6, 9, 17, 64, 66, 67]

                # Voxelize the points for each class
                point_coordinates = {point_class:list() for point_class in point_classes_no_ground}     # {1: [(x1,y1,z1),...], ...}

                for point_class in tqdm(point_classes_no_ground, desc='Voxelize lidar points', leave=False):
                    mask = lidar_batch.classification == point_class
                    batch_ground_points_no_ground = lidar_batch[mask]

                    x = batch_ground_points_no_ground.x
                    y = batch_ground_points_no_ground.y
                    z = batch_ground_points_no_ground.z + z_axis_translate
                    xyz_no_ground = np.vstack([x, y, z]).T

                    points_relative_to_voxel_origin = xyz_no_ground - [xmin_relative, ymin_relative, 0]
                    
                    voxel_origins_relative_m = find_occupied_voxels_vectorized(
                        points_relative_to_voxel_origin,
                        voxel_size=VOXEL_SIDE,
                        min_points_per_voxel=0
                    )
                    point_coordinates[point_class] = voxel_origins_relative_m

                dominant_per_voxel, filtered_points = dominant_voxel_points(point_coordinates)


                # ----------------------- Write lidar data to schematic ---------------------- #
                do_No_Class(         filtered_points[1], choosen_template_point_classes, schem)
                do_Small_Vegetation( filtered_points[3], choosen_template_point_classes, schem)
                do_Medium_Vegetation(filtered_points[4], choosen_template_point_classes, schem)
                do_High_Vegetation(  filtered_points[5], choosen_template_point_classes, schem)
                do_Building(         filtered_points[6], choosen_template_point_classes, schem)
                do_Water(            filtered_points[9], choosen_template_point_classes, schem)
                do_Bridge(           filtered_points[17], choosen_template_point_classes, schem)
                do_Perennial_Soil(   filtered_points[64], choosen_template_point_classes, schem)
                do_Virtual_Points(   filtered_points[66], choosen_template_point_classes, schem)
                do_Miscellaneous(    filtered_points[67], choosen_template_point_classes, schem)


                # --------------------------- Save batch schematic --------------------------- #
                schem_batch_filename = f'xmin~{xmin_absolute}_ymin~{ymin_absolute}_size~{tile_edge_size}'
                schem.save(str(schematic_folderpath), schem_batch_filename, mcschematic.Version.JE_1_21)

                # ---------------------------- Add batch functions --------------------------- #
                text_mcfunction += f'\n/say Placing Batch {batch_x*BATCH_PER_PRODUCT_SIDE + batch_y + 1}/{BATCH_PER_PRODUCT_SIDE**2} at X={xmin_absolute} Z={ymin_absolute}\n'
                text_mcfunction += f'/tp @s {xmin_absolute} 0 {ymin_absolute}\n'
                text_mcfunction += f'//schematic load {schem_batch_filename}\n'
                text_mcfunction +=  '//paste -a\n'

        # ---------------------------- Finalize MCFunction --------------------------- #
        text_mcfunction += '\nsay Lidar placement complete!\n'
        text_mcfunction += 'gamemode creative @s\n' # Set player back to creative
        with open(mcfunction_filepath, 'w') as f: 
            f.write(text_mcfunction)
        logger.info(f"Generated MCFunction file: {mcfunction_filepath}")
        logger.info("--- Processing Finished ---")
