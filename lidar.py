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


from shapely.geometry import Polygon, shape, box
from shapely.ops import transform
from pathlib import Path
from utils.logger import setup_logging
from script import find_occupied_voxels_vectorized
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from osm_test import get_road_coordinates_by_type, get_terrain_coordinates_by_type

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
            break


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
    point_coordinates: Dict[Class, List[Point]],
    grid_size:float
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
            # Skip points that are outside the manual batch size limit
            if x>=grid_size or y>=grid_size or x<=0 or y<=0:
                continue
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
            # Skip points that are outside the manual batch size limit
            if x>=grid_size or y>=grid_size or x<=0 or y<=0:
                continue
            voxel = (math.floor(x), math.floor(y), math.floor(z))
            if dominant_per_voxel[voxel] == cls:
                filtered_points[cls].append((x,y,z))

    return dominant_per_voxel, filtered_points





def do_No_Class(coordinates:list[tuple[np.float64, np.float64, np.float64]], block_template:dict[str, dict[str, str]], schem:mcschematic.MCSchematic):
    no_class_placed_blocks = list()
    for x,y,z in coordinates:
        current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
        if current_block_state=='minecraft:air':
            schem.setBlock((int(x),int(z),int(y)), block_template['lidar']['No Class'])
            no_class_placed_blocks.append((int(x),int(y),int(z)))
    # Filter the blocks
    block_to_delete = list()
    for x,y,z in no_class_placed_blocks:
        block_below = schem.getBlockDataAt((int(x),int(z-1),int(y)))
        block_front = schem.getBlockDataAt((int(x+1),int(z),int(y)))
        block_back = schem.getBlockDataAt((int(x-1),int(z),int(y)))
        block_left = schem.getBlockDataAt((int(x),int(z),int(y-1)))
        block_right = schem.getBlockDataAt((int(x),int(z),int(y+1)))
        nb_adjacent_no_class_block = 0
        if block_front==block_template['lidar']['No Class']:
            nb_adjacent_no_class_block += 1
        if block_back==block_template['lidar']['No Class']:
            nb_adjacent_no_class_block += 1
        if block_left==block_template['lidar']['No Class']:
            nb_adjacent_no_class_block += 1
        if block_right==block_template['lidar']['No Class']:
            nb_adjacent_no_class_block += 1
        
        if nb_adjacent_no_class_block<2 or block_below=='minecraft:air':    # delete if the block have less then 2 adjacent no class block
            block_to_delete.append((int(x),int(y),int(z)))

    # actually delete the unwanted blocks (doing that after the filter to not delete the block while scanning)
    for x,y,z in block_to_delete:
        schem.setBlock((int(x),int(z),int(y)), 'minecraft:air')

    # tqdm.write(f'{len(coordinates)} No Class points | {len(no_class_placed_blocks)} block placed | {len(block_to_delete)} block deleted')

def do_Small_Vegetation(coordinates, block_template:dict[str, dict[str, str]], schem:mcschematic.MCSchematic):
    for x,y,z in coordinates:
        current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
        if current_block_state==block_template['mnt']['ground_top']:
            above_block_state = schem.getBlockDataAt((int(x),int(z+1),int(y)))
            if above_block_state==block_template['mnt']['ground_top']:
                continue
            else: 
                z += 1
        schem.setBlock((int(x),int(z),int(y)), block_template['lidar']['Small Vegetation'])

def do_Medium_Vegetation(coordinates, block_template:dict[str, dict[str, str]], schem:mcschematic.MCSchematic):
    for x,y,z in coordinates:
        current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
        if current_block_state=='minecraft:air' or current_block_state==block_template['lidar']['No Class']:
            schem.setBlock((int(x),int(z),int(y)), block_template['lidar']['Medium Vegetation'])

def do_High_Vegetation(coordinates, block_template:dict[str, dict[str, str]], schem:mcschematic.MCSchematic):
    for x,y,z in coordinates:
        current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
        if current_block_state=='minecraft:air' or current_block_state==block_template['lidar']['No Class']:
            schem.setBlock((int(x),int(z),int(y)), block_template['lidar']['High Vegetation'])

def do_Building(coordinates, block_template:dict[str, dict[str, str]], schem:mcschematic.MCSchematic):
    for x,y,z in coordinates:
        current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
        if current_block_state=='minecraft:air' or current_block_state==block_template['lidar']['No Class']:
            schem.setBlock((int(x),int(z),int(y)), block_template['lidar']['Building'])
        # Extend the building block to the ground
        for z_below in range(int(z), LOWEST_MINECRAFT_POINT, -1):
            below_block_state = schem.getBlockDataAt((int(x),int(z_below),int(y)))
            if below_block_state==block_template['mnt']['ground_below']:
                break
            schem.setBlock((int(x),int(z_below),int(y)), block_template['lidar']['Building'])

def do_Water(coordinates, block_template:dict[str, dict[str, str]], schem:mcschematic.MCSchematic):
    for x,y,z in coordinates:
        current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
        if current_block_state=='minecraft:air' or current_block_state==block_template['lidar']['No Class']:
            schem.setBlock((int(x),int(z),int(y)), block_template['lidar']['Water'])

def do_Bridge(coordinates, block_template:dict[str, dict[str, str]], schem:mcschematic.MCSchematic):
    for x,y,z in coordinates:
        current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
        if current_block_state=='minecraft:air' or current_block_state==block_template['lidar']['No Class']:
            schem.setBlock((int(x),int(z),int(y)), block_template['lidar']['Bridge'])

def do_Perennial_Soil(coordinates, block_template:dict[str, dict[str, str]], schem:mcschematic.MCSchematic):
    for x,y,z in coordinates:
        current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
        if current_block_state=='minecraft:air' or current_block_state==block_template['lidar']['No Class']:
            schem.setBlock((int(x),int(z),int(y)), block_template['lidar']['Perennial Soil'])

def do_Virtual_Points(coordinates, block_template:dict[str, dict[str, str]], schem:mcschematic.MCSchematic):
    for x,y,z in coordinates:
        current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
        if current_block_state=='minecraft:air' or current_block_state==block_template['lidar']['No Class']:
            schem.setBlock((int(x),int(z),int(y)), block_template['lidar']['Virtual Points'])

def do_Miscellaneous(coordinates, block_template:dict[str, dict[str, str]], schem:mcschematic.MCSchematic):
    for x,y,z in coordinates:
        current_block_state = schem.getBlockDataAt((int(x),int(z),int(y)))
        if current_block_state=='minecraft:air' or current_block_state==block_template['lidar']['No Class']:
            schem.setBlock((int(x),int(z),int(y)), block_template['lidar']['Miscellaneous'])






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

    zone_geojson_filepath = Path('data/zone_test_caussol.geojson')

    lidar_folderpath      = Path('data/tiles/lidar/')
    mnt_folderpath        = Path('data/tiles/mnt')
    schematic_folderpath  = Path('data/myschems')
    mcfunction_folderpath = Path('data/mcfunctions')

    SEARCH_FOR_TILE_IN_ZONE = True
    FORCE_TILE_GENERATION = False

    DO_MNT = True
    DO_OSM = True
    DO_LIDAR = True


    # Minecraft parameters
    MANUAL_Z_AXIS_TRANSLATE = True  # If True, you must set LOWEST_MINECRAFT_POINT to the lowest point of the MNT
    MANUAL_Z_AXIS_TRANSLATE_VALUE = -2000  # If MANUAL_Z_AXIS_TRANSLATE is True, this value will be used to translate the Z axis of the MNT to the Minecraft world
    LOWEST_MINECRAFT_POINT = -2031          # If normal minecraft : -60
    HIGHEST_MINECRAFT_POINT = 2025          # If normal minecraft : 319

    GROUND_THICKNESS = 16

    # Processing parameters
    PERCENTAGE_TO_REMOVE_NON_GROUND = 0     # Decimation for non-ground features
    VOXEL_SIDE = 0.5
    MIN_POINTS_PER_VOXEL_NON_GROUND = 3     # Filtering for non-ground features
    BATCH_PER_PRODUCT_SIDE = 4              # Must be divisible by 2. Split 1 tile into BATCH_PER_PRODUCT_SIDE * BATCH_PER_PRODUCT_SIDE batches

    # Ground filling parameters
    INTERPOLATION_GRID_CELL_SIZE = 1.0      # Grid resolution for point cloud interpolation

    # Block mapping
    block_template = {
        "lidar":{
            'No Class':         'minecraft:stone',
            'Small Vegetation': 'minecraft:short_grass',
            'Medium Vegetation':'minecraft:moss_block',
            'High Vegetation':  'minecraft:oak_leaves',
            'Building':         'minecraft:stone_bricks',
            'Water':            'minecraft:blue_stained_glass',
            'Bridge':           'minecraft:polished_blackstone',
            'Perennial Soil':   'minecraft:iron_block',
            'Virtual Points':   'minecraft:diorite',
            'Miscellaneous':    'minecraft:basalt'
        },
        "mnt":{
            'ground_top':   'minecraft:grass_block',
            'ground_below': 'minecraft:dirt'
        },
        "osm":{
            "motorway":     "minecraft:black_concrete",
            "trunk":        "minecraft:gray_concrete",
            "primary":      "minecraft:light_gray_concrete",
            "secondary":    "minecraft:andesite",
            "tertiary":     "minecraft:polished_andesite",
            "unclassified": "minecraft:gravel",
            "residential":  "minecraft:cobblestone",
            "service":      "minecraft:dirt_path",
            "living_street":"minecraft:stone_bricks",
            "pedestrian":   "minecraft:smooth_stone",
            "footway":      "minecraft:stone_slab",
            "cycleway":     "minecraft:green_concrete",
            "path":         "minecraft:coarse_dirt",
            "track":        "minecraft:sand",
            "steps":        "minecraft:oak_stairs",
            "bridleway":    "minecraft:spruce_planks",
            "raceway":      "minecraft:red_concrete",
            "bus_guideway": "minecraft:yellow_concrete",
            "corridor":     "minecraft:quartz_block",
            "elevator":     "minecraft:iron_bars",
            "escalator":    "minecraft:polished_diorite_slab",
            "platform":     "minecraft:polished_granite",
            "proposed":     "minecraft:light_blue_concrete",
            "construction": "minecraft:orange_concrete"
        }
    }

    # point class : min nb points per voxel 
    lidar_point_class =  {
        1:2,            # No Class
        # 2:9999,       # Ground --> managed by mnt
        3:2,            # Small Vegetation
        4:2,            # Medium Vegetation
        5:2,            # High Vegetation
        6:2,            # Building
        9:0,            # Water
        17:0,           # Bridge
        64:0,           # Perennial Soil
        66:0,           # Virtual Points
        67:0            # Miscellaneous
    }

    road_block_template = {
        "motorway": 20.0,  # Multi-lane highway, including median
        "trunk": 15.0,     # Major non-motorway road
        "primary": 12.0,
        "secondary": 10.0,
        "tertiary": 8.0,
        "unclassified": 6.0,
        "residential": 5.0,
        "service": 4.0,
        "living_street": 4.0,
        "pedestrian": 3.0,
        "footway": 2.0,
        "cycleway": 2.5,
        "path": 2.0,
        "track": 3.0,
        "steps": 1.5,
        "bridleway": 2.0,
        "raceway": 15.0, # Varies greatly, this is a general estimate
        "bus_guideway": 5.0,
        "corridor": 3.0,
        "elevator": 1.5,
        "escalator": 2.0,
        "platform": 5.0, # Varies, e.g., train platform
        "proposed": 5.0, # Default for proposed roads
        "construction": 5.0 # Default for roads under construction
    }



    # -------------------------- Download step ------------------------- #
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
        except KeyError:
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
    with logging_redirect_tqdm():
        for tile_bbox, tile_data in tqdm(tiles.items(), desc='Processing tiles'):

            # -------------------------- Initialize mc function -------------------------- #

            mcfunction_filepath:Path = mcfunction_folderpath / (tile_bbox + '.mcfunction')
            text_mcfunction = '# Auto-generated MCFunction for placing lidar data\n'
            text_mcfunction += '/gamerule doDaylightCycle false\n'
            text_mcfunction += '/time set day\n'
            text_mcfunction += '/gamerule randomTickSpeed 0\n'
            text_mcfunction += '/gamemode spectator @s\n'
            text_mcfunction += '/say Starting lidar placement...\n'

            # If the tile already exists and we do not force the tile re-generation, then do the next tile
            if mcfunction_filepath.exists() and not FORCE_TILE_GENERATION:
                logger.info(f'Tile {tile_bbox} already exists. Skipping it')
                continue
            
            logger.info(f"Processing tile with bbox: {tile_bbox}")

            # Check if both lidar and mnt are available for this tile
            if 'lidar' not in tile_data or 'mnt' not in tile_data:
                logger.warning(f"Skipping tile {tile_bbox} as it does not have both lidar and mnt data.")
                continue

            lidar_tile_filepath = Path(tile_data['lidar']['filepath'])
            mnt_tile_filepath   = Path(tile_data['mnt']['filepath'])

            # ------------------------------- Download & Process OSM ------------------------------- #
            logger.info("Fetching and processing OSM data...")
            
            # bbox to tile polygon
            zone_polygon_4326 = box(*tile_data['lidar']['bbox'])
            transformer = pyproj.Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform
            zone_polygon_2154 = transform(transformer, zone_polygon_4326)

            try:
                osm_roads = get_road_coordinates_by_type(zone_polygon_2154)
                osm_terrains = get_terrain_coordinates_by_type(zone_polygon_2154)
            except:
                logger.exception(f'Error happened during getting OSM data. Skipping tile bbox {tile_bbox}')
                continue

            # Clean OSM for MC
            osm_roads    = {k:[(x,-y+1000) for x,y in v] for k,v in osm_roads.items()}
            osm_terrains = {k:[(x,-y+1000) for x,y in v] for k,v in osm_terrains.items()}
        
            logger.info("OSM data fetched and processed.")

            # --------------------------------- Load MNT --------------------------------- #
            logger.info(f"Loading MNT file: {mnt_tile_filepath} ...")
            mnt = rasterio.open(mnt_tile_filepath)
            logger.info("MNT loaded")

            # --------------------------------- clean MNT -------------------------------- #
            logger.info("Cleaning MNT data...")
            mnt_array:np.ndarray = mnt.read(1)
            
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
                    ymin_absolute = -tile_min_y + ymin_relative     # Invert the Y axis to oreder the vertical tiling (bc north is inverse of lambert esqg)
                    ymax_absolute = -tile_min_y + ymax_relative

                    
                    # tqdm.write(f'BATCH : {xmin_relative=} {xmax_relative=} {ymin_relative=} {ymax_relative=}')

                    if DO_MNT:
                        # ------------------------------ MNT batch data ------------------------------ #
                        mnt_batch_array:np.ndarray = mnt_array[xmin_relative:xmax_relative, ymin_relative:ymax_relative]
                        mnt_batch_array = mnt_batch_array + z_axis_translate


                        # ------------------------ Write MNT data to schematic ----------------------- #
                        for x in tqdm(range(mnt_batch_array.shape[0]), desc='Placing MNT block batch', leave=False):
                            for y in range(mnt_batch_array.shape[1]):
                                z = mnt_batch_array[x, y]

                                schem.setBlock((x, z, y), block_template['mnt']['ground_top'])
                                for i in range(1, GROUND_THICKNESS+1):
                                    if z-i > LOWEST_MINECRAFT_POINT:
                                        schem.setBlock((x, z-i, y), block_template['mnt']['ground_below'])
                        # tqdm.write(f'MNT : {mnt_batch_array.shape=} {mnt_batch_array.min()=} | {mnt_batch_array.max()=}')

                    # ------------------------------ OSM batch data ------------------------------ #

                    if DO_OSM:
                        # debug_coord = list()
                        nb_road_block_available = 0
                        nb_road_block_placed = 0
                        for road_type, road_data in tqdm(osm_roads.items(), desc='Placing OSM roads', leave=False):
                            try:
                                for road_point_x, road_point_y in tqdm(road_data, desc=f'Placing OSM points ({road_type})', leave=False):
                                    nb_road_block_available += 1
                                    if xmin_relative <= road_point_x < xmax_relative and ymin_relative <= road_point_y < ymax_relative:
                                        road_point_x = int(road_point_x) - xmin_relative
                                        road_point_y = int(road_point_y) - ymin_relative
                                        # debug_coord.append((road_point_x, road_point_y))
                                        road_block_height = mnt_batch_array[road_point_x, road_point_y]
                                        current_block = schem.getBlockDataAt((road_point_x, road_block_height, road_point_y))
                                        if current_block == block_template['mnt']['ground_top']:
                                            schem.setBlock((road_point_x, road_block_height, road_point_y), block_template['osm'][road_type])
                                            nb_road_block_placed += 1
                            except: continue
                        # debug_coord = np.array(debug_coord)
                        # tqdm.write(f'OSM : {debug_coord[:,0].min()=} {debug_coord[:,0].max()=} {debug_coord[:,1].min()=} {debug_coord[:,1].max()=}')
                        # tqdm.write(f'Batch blocks available : {nb_road_block_available} | Placed : {nb_road_block_placed}')
                    

                    if DO_LIDAR:
                        # ----------------------------- Lidar batch data ----------------------------- #
                        lidar_batch:laspy.LasData = lidar[(lidar.x<=xmax_relative) & 
                                                        (lidar.x>=xmin_relative) & 
                                                        (lidar.y<=ymax_relative) & 
                                                        (lidar.y>=ymin_relative)]
                        

                        # Voxelize the points for each class
                        point_coordinates = {point_class:list() for point_class in lidar_point_class.keys()}     # {1: [(x1,y1,z1),...], ...}

                        for point_class, min_points_per_voxel in tqdm(lidar_point_class.items(), desc='Voxelize lidar points', leave=False):
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
                                min_points_per_voxel=min_points_per_voxel
                            )
                            point_coordinates[point_class] = voxel_origins_relative_m

                        dominant_per_voxel, filtered_points = dominant_voxel_points(point_coordinates, grid_size=batch_size)



                        # ----------------------- Write lidar data to schematic ---------------------- #
                        do_Small_Vegetation( filtered_points[3],  block_template, schem)
                        do_Medium_Vegetation(filtered_points[4],  block_template, schem)
                        do_High_Vegetation(  filtered_points[5],  block_template, schem)
                        do_Building(         filtered_points[6],  block_template, schem)
                        do_Water(            filtered_points[9],  block_template, schem)
                        do_Bridge(           filtered_points[17], block_template, schem)
                        do_Perennial_Soil(   filtered_points[64], block_template, schem)
                        do_Virtual_Points(   filtered_points[66], block_template, schem)
                        do_Miscellaneous(    filtered_points[67], block_template, schem)
                        do_No_Class(         filtered_points[1],  block_template, schem)


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
