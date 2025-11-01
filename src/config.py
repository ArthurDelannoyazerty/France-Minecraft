from pathlib import Path

# --- FOLDER PATHS ---
# Base data directory
DATA_DIR = Path('data')

# Specific subdirectories
GRID_DIR = DATA_DIR / 'data_grille'
RAW_POINT_CLOUD_DIR = DATA_DIR / 'raw_point_cloud'
LOG_DIR = DATA_DIR / 'logs'
SCHEMATICS_DIR = DATA_DIR / 'myschems'
MCFUNCTIONS_DIR = DATA_DIR / 'mcfunctions'
LIDAR_TILES_DIR = DATA_DIR / 'tiles' / 'lidar'
MNT_TILES_DIR = DATA_DIR / 'tiles' / 'mnt'

# --- DATA SOURCE & DOWNLOAD ---
FORCE_DOWNLOAD_CATALOG = False
LIDAR_CATALOG_FILE = GRID_DIR / 'lidar_public_tiles_available.geojson'
MNT_CATALOG_FILE = GRID_DIR / 'mnt_public_tiles_available.geojson'

# Path to the GeoJSON file defining the area of interest
ZONE_GEOJSON_FILE = DATA_DIR / 'zone_test_caussol.geojson'

# --- PROCESSING FLAGS ---
# Set to False to use hardcoded test tiles instead of searching in the zone
SEARCH_FOR_TILE_IN_ZONE = True
# If True, will regenerate schematics and mcfunctions even if they exist
FORCE_TILE_GENERATION = False

# Toggle main processing steps
DO_MNT_PROCESSING = True
DO_OSM_PROCESSING = True
DO_LIDAR_PROCESSING = True

# --- MINECRAFT WORLD PARAMETERS ---
    
# --- MINECRAFT WORLD PARAMETERS ---
# Set to True to automatically create a backup .zip of the world before writing.
AUTO_BACKUP_WORLD = True

# The dimension to write to ('minecraft:overworld', 'minecraft:the_nether', 'minecraft:the_end')
TARGET_DIMENSION = "minecraft:overworld"

# If True, Z-axis is manually offset by MANUAL_Z_AXIS_OFFSET
MANUAL_Z_AXIS_TRANSLATE = True
MANUAL_Z_AXIS_OFFSET = -2000

# Minecraft world height limits
LOWEST_MINECRAFT_POINT = -60
HIGHEST_MINECRAFT_POINT = 319

# How many blocks deep the ground layer should be
GROUND_THICKNESS = 16

  
# --- TILE PROCESSING PARAMETERS ---
# A 1km x 1km tile is split into (N x N) batches.
# BATCH_PER_SIDE = 4 means the tile is split into 16 batches (4x4).
BATCHES_PER_TILE_SIDE = 4

# --- LIDAR PROCESSING PARAMETERS ---
VOXEL_SIZE = 0.5  # Side length of a voxel in meters

# Defines the minimum number of lidar points required within a voxel for it to be considered.
# This helps filter out noise. Mapped by lidar classification ID.
# Lidar class 2 (Ground) is handled by the MNT raster, so it's omitted here.
LIDAR_CLASS_MIN_POINTS = {
    1: 2,   # No Class
    3: 2,   # Small Vegetation
    4: 2,   # Medium Vegetation
    5: 2,   # High Vegetation
    6: 2,   # Building
    9: 0,   # Water (0 means any point is considered)
    17: 0,  # Bridge
    64: 0,  # Perennial Soil
    66: 0,  # Virtual Points
    67: 0,  # Miscellaneous
}


# --- BLOCK MAPPING ---
# Maps different data classifications to Minecraft block IDs.
BLOCK_MAPPING = {
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

# --- OSM PROCESSING PARAMETERS ---
TARGET_CRS = "EPSG:2154"  # Lambert-93, a projected CRS in meters
RASTER_RESOLUTION_METERS = 1.0  # 1 meter resolution for OSM rasterization

# Defines rendering priority for OSM tags. Higher integer value = drawn on top.
OSM_FEATURE_VALUE_MAP = {
    "natural": {"sand": 1, "glacier": 2, "bare_rock": 3, "rock": 4, "scrub": 5,
                "heath": 6, "wood": 7, "grassland": 8, "wetland": 9, "shingle": 10},
    "landuse": {"forest": 11, "farmland": 12, "meadow": 13, "grass": 14, "quarry": 15,
                "residential": 16, "industrial": 17, "recreation_ground": 18},
    "highway": {"motorway": 31, "trunk": 32, "primary": 33, "secondary": 34, "tertiary": 35,
                "unclassified": 36, "residential": 37, "service": 38, "living_street": 39,
                "pedestrian": 40, "footway": 41, "cycleway": 42, "path": 43,
                "track": 44, "steps": 45, "bridleway": 46, "raceway": 47,
                "bus_guideway": 48, "corridor": 49, "elevator": 50, "escalator": 51,
                "platform": 52, "proposed": 53, "construction": 54},
    "surface": {"dirt": 21, "gravel": 22, "sand": 23, "grass": 24, "mud": 25,
                "paved": 26, "asphalt": 27}
}

# Defines the physical width of roads in meters for buffering.
OSM_ROAD_WIDTH_MAP = {
    "motorway": 13.0, "trunk": 10.0, "primary": 10.0, "secondary": 8.5,
    "tertiary": 7.5, "unclassified": 6.0, "residential": 6.0, "service": 3.0,
    "living_street": 3.0, "pedestrian": 1.0, "footway": 2.0, "cycleway": 2.5,
    "path": 3.0, "track": 3.0, "steps": 1.5, "bridleway": 2.5, "raceway": 12.0,
    "bus_guideway": 3.25, "corridor": 3.0, "elevator": 2.0, "escalator": 2.0,
    "platform": 5.0, "proposed": 3.0, "construction": 3.0,
}

# Defines the drawing order priority for different OSM feature types.
OSM_TAG_TYPE_PRIORITY = ["highway", "surface", "landuse", "natural"]