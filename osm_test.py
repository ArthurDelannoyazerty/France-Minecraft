import overpy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyproj
from shapely.ops import transform as shapely_transform
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Polygon
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio.transform
import numpy as np


# --- Configuration Constants ---
GEOJSON_FILEPATH = Path("data/zone_test.geojson")
TARGET_CRS = "EPSG:2154"  # Lambert-93, a projected CRS in meters
RASTER_RESOLUTION_METERS = 1.0  # 1 meter resolution


# Define a mapping from specific tags to integer values
# Higher values will be drawn on top (last in rasterize, brighter in colormap)
# Values start from 1, as 0 is reserved for background/unclassified
FEATURE_VALUE_MAP = {
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


# Typical real-world widths in meters for different highway types
# These values are approximate and can be adjusted
ROAD_WIDTH_MAP = {
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



# Define a priority for tag types (highways on top, then surface, etc.)
TAG_TYPE_PRIORITY = ["highway", "surface", "landuse", "natural"]


def build_overpass_query(polygon: Polygon) -> str:
    """
    Builds an Overpass QL query string for natural, landuse, and surface features
    within a given polygon.
    """
    coords = polygon.exterior.coords
    # Overpass expects "lat lon" pairs, so swap x, y from shapely (lon, lat)
    poly_str = " ".join(f"{lat} {lon}" for lon, lat in coords)

    query = f"""
        [out:json][timeout:60];
        (
        // LAND COVER & NATURAL FEATURES
        way["natural"~"^(sand|glacier|bare_rock|rock|scrub|heath|wood|grassland|wetland|shingle)$"](poly:"{poly_str}");
        way["landuse"~"^(forest|farmland|meadow|grass|quarry|residential|industrial|recreation_ground)$"](poly:"{poly_str}");

        // SURFACE TYPES (from roads, paths, etc.)
        way["surface"~"^(dirt|gravel|sand|grass|mud|paved|asphalt)$"](poly:"{poly_str}");

        way["highway"](poly:"{poly_str}"); // Roads

        // Also include similar tags on relations
        relation["natural"](poly:"{poly_str}");
        relation["landuse"](poly:"{poly_str}");
        );
        out body;
        >;
        out skel qt;
    """

    return query


def query_overpass(query: str) -> overpy.Result:
    """Executes an Overpass QL query and returns the result."""
    api = overpy.Overpass()
    return api.query(query)


def process_overpass_result(result: overpy.Result) -> gpd.GeoDataFrame:
    """
    Processes Overpass API result to extract geometries and tags into a GeoDataFrame.
    """
    features = []
    for way in result.ways:
        tags = way.tags
        # Only include ways that have at least one of our target tag types
        if any(k in tags for k in TAG_TYPE_PRIORITY):
            nodes = way.nodes
            if len(nodes) < 2:
                continue

            way_coords = [(float(n.lon), float(n.lat)) for n in nodes]

            # If the way is closed, treat it as a Polygon, otherwise as a LineString
            if len(way_coords) > 3 and way_coords[0] == way_coords[-1]:
                geom = Polygon(way_coords)
            else:
                geom = LineString(way_coords)
            features.append({"geometry": geom, **tags})

    if not features:
        print("No features found in Overpass response.")
        return gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")

    return gpd.GeoDataFrame(features, crs="EPSG:4326")

# def filter_out_roads(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
#     """Filters out the road types"""
#     return gdf[~gdf['highway'].isin(FEATURE_VALUE_MAP.get('highway', {}).keys())].copy()


def assign_feature_values(
    gdf: gpd.GeoDataFrame, feature_map: Dict[str, Dict[str, int]], priority: List[str]
) -> gpd.GeoDataFrame:
    """
    Assigns a numerical value to each feature based on its tags and a defined priority.
    Also assigns a 'buffer_distance' for highway types.
    """

    def _get_value_and_buffer_distance(row: Dict[str, Any]) -> Tuple[int, float]:
        value = 0
        buffer_dist = 0.0 # Default no buffer
        for tag_type in priority: # Iterate through tag types based on priority
            if tag_type in row and row[tag_type] in feature_map.get(tag_type, {}): # Check if tag exists in row and in the feature_map
                value = feature_map[tag_type][row[tag_type]]
                # If it's a highway, get its width and calculate buffer distance (radius)
                if tag_type == "highway" and row[tag_type] in ROAD_WIDTH_MAP:
                    buffer_dist = ROAD_WIDTH_MAP[row[tag_type]] / 2.0
                break # Found the highest priority tag, stop searching

        return value, buffer_dist

    # Apply the function to get both value and buffer_distance
    gdf[['value', 'buffer_distance']] = gdf.apply(lambda row: pd.Series(_get_value_and_buffer_distance(row)), axis=1)
    # Sort by value to ensure correct z-ordering during rasterization (lower values first)
    return gdf.sort_values(by='value', ascending=True)


def rasterize_geometries(
    gdf: gpd.GeoDataFrame, 
    resolution_meters: float, 
    target_crs: str,
    clip_polygon_proj: Polygon, # New parameter for clipping
) -> Tuple[np.ndarray, List[float], rasterio.transform.Affine]:
    """
    Rasterizes GeoDataFrame geometries to a specified resolution in a target CRS.
    Returns the raster array, its extent, and the affine transform.
    """
    if gdf.empty:
        print("GeoDataFrame is empty, cannot rasterize.")
        return np.array([]), [], rasterio.transform.Affine.identity()

    # Reproject to a projected CRS for meter-based resolution
    gdf_proj = gdf.to_crs(target_crs)

    # Clip the GeoDataFrame to the exact bounds of the input polygon
    # This ensures that only relevant geometries are considered for rasterization
    gdf_proj_clipped = gdf_proj.clip(clip_polygon_proj)

    if gdf_proj_clipped.empty:
        print("GeoDataFrame is empty after clipping, cannot rasterize.")
        return np.array([]), [], rasterio.transform.Affine.identity()

    # Use the bounds of the clip_polygon_proj to define the raster extent
    minx, miny, maxx, maxy = clip_polygon_proj.bounds

    width_proj = maxx - minx
    height_proj = maxy - miny

    out_shape = (int(np.ceil(height_proj / resolution_meters)),
                 int(np.ceil(width_proj / resolution_meters)))

    # Create the affine transform for the projected CRS based on the clip_polygon_proj bounds
    geometries_to_rasterize = []
    transform = rasterio.transform.from_bounds(
        west=minx,
        south=miny,
        east=maxx,
        north=maxy,
        width=out_shape[1],
        height=out_shape[0],
    )

    # Prepare geometries for rasterization, applying buffer if necessary
    # Iterate over the *clipped* GeoDataFrame
    for idx, row in gdf_proj_clipped.iterrows():
        geom = row.geometry
        value = row['value']
        # Check if 'buffer_distance' column exists and get its value, default to 0.0
        buffer_dist = row.get('buffer_distance', 0.0)

        if isinstance(geom, LineString) and buffer_dist > 0:
            # Buffer the LineString to create a Polygon representing the road width
            buffered_geom = geom.buffer(buffer_dist, cap_style=3, join_style=3)
            geometries_to_rasterize.append((buffered_geom, value))
        else:
            geometries_to_rasterize.append((geom, value))

    # Rasterize the geometries using their assigned 'value'
    raster = rasterize(
        geometries_to_rasterize, # Use the list of (buffered_geom, value)
        out_shape=out_shape,
        transform=transform,
        fill=0,  # Default value for areas not covered by features
        default_value=0,
        all_touched=True,  # Include all pixels touched by geometries
    )

    extent = [minx, maxx, miny, maxy]
    return raster, extent, transform


def create_custom_colormap(feature_map: Dict[str, Dict[str, int]]) -> mcolors.ListedColormap:
    """
    Creates a custom colormap based on the feature value map.
    """
    # Example colors (you'd want to refine these for better visual distinction)
    # Ensure the list is long enough for all assigned values (up to max_value)
    colors_list = [
        (0, 0, 0, 0),  # Value 0 (transparent or background)
        # Natural (1-10) - earthy/natural tones
        (0.8, 0.8, 0.6), (0.9, 0.9, 0.9), (0.5, 0.5, 0.5), (0.4, 0.4, 0.4), (0.6, 0.7, 0.5),
        (0.7, 0.8, 0.6), (0.2, 0.5, 0.2), (0.6, 0.8, 0.4), (0.4, 0.6, 0.8), (0.7, 0.7, 0.7),
        # Landuse (11-20) - more human-influenced/broader categories
        (0.1, 0.4, 0.1), (0.8, 0.7, 0.3), (0.5, 0.7, 0.3), (0.4, 0.6, 0.2), (0.7, 0.7, 0.7),
        (0.9, 0.4, 0.4), (0.6, 0.3, 0.3), (0.4, 0.8, 0.8), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), # Placeholder for 19, 20,
        # Surface (21-30) - road/path colors (should be distinct and on top)
        (0.6, 0.4, 0.2), (0.5, 0.5, 0.5), (0.8, 0.8, 0.6), (0.4, 0.6, 0.2), (0.5, 0.3, 0.1),
        (0.3, 0.3, 0.3), (0.2, 0.2, 0.2),
        # Highway (31-54) - road colors, ensure enough distinct colors
        # Values 31-35
        (0.4, 0.4, 0.4), (0.4, 0.4, 0.4), (0.5, 0.5, 0.5), (0.6, 0.6, 0.6), (0.7, 0.7, 0.7),
        # Surface (21-30) - road/path colors (should be distinct and on top)
        (0.6, 0.4, 0.2), (0.5, 0.5, 0.5), (0.8, 0.8, 0.6), (0.4, 0.6, 0.2), (0.5, 0.3, 0.1),
        (0.3, 0.3, 0.3), (0.2, 0.2, 0.2),
    ]

    # Ensure the list is long enough for all values, fill with black if not enough
    max_value = max(v for d in feature_map.values() for v in d.values())
    while len(colors_list) <= max_value:
        colors_list.append((0, 0, 0, 1))  # Default to black for unassigned values

    return mcolors.ListedColormap(colors_list)


def display_raster(
    raster: np.ndarray, 
    extent: List[float], 
    custom_cmap: mcolors.ListedColormap,
    feature_map: Dict[str, Dict[str, int]], 
    resolution_meters: float,
    output_filepath: str
) -> None:
    """
    Displays the rasterized features with a custom colormap and colorbar.
    """
    if raster.size == 0:
        print(f"No raster data to display for {output_filepath}.") # Added for clarity
        return

    fig, ax = plt.subplots(figsize=(12, 12))
    max_value = max(v for d in feature_map.values() for v in d.values())

    im = ax.imshow(raster, extent=extent, cmap=custom_cmap, vmin=0, vmax=max_value)
    ax.set_title(f"Rasterized Features (Resolution: {resolution_meters}m)")
    ax.set_xlabel("Easting (meters)")
    ax.set_ylabel("Northing (meters)")

    # Build the reverse mapping for colorbar labels
    colorbar_labels_map = {}
    for tag_type, values_dict in feature_map.items():
        for tag_value, int_id in values_dict.items():
            colorbar_labels_map[int_id] = f"{tag_type}: {tag_value}"

    # Prepare colorbar ticks and labels
    unique_assigned_values = sorted(np.unique(raster).tolist())
    # Filter out 0 (background) if present
    unique_assigned_values = [v for v in unique_assigned_values if v != 0]

    cbar_ticks = unique_assigned_values
    cbar_tick_labels = [colorbar_labels_map.get(v, 'Unknown') for v in unique_assigned_values]

    cbar = fig.colorbar(im, ax=ax, ticks=cbar_ticks, orientation='vertical', shrink=0.75)
    cbar.ax.set_yticklabels(cbar_tick_labels)
    cbar.set_label('Feature Type')

    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.close(fig) # Close the figure to free up memory


def _get_coordinates_from_geometry(geom, origin_x: float, origin_y: float) -> List[Tuple[float, float]]:
    """
    Extracts coordinates from a shapely geometry and makes them relative to an origin.
    Handles LineString, Polygon, MultiPolygon.
    """
    coords = []
    if geom.geom_type == 'LineString':
        for x, y in geom.coords:
            coords.append((x - origin_x, y - origin_y))
    elif geom.geom_type == 'Polygon':
        # Only exterior for now, could add interiors if needed
        for x, y in geom.exterior.coords:
            coords.append((x - origin_x, y - origin_y))
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            for x, y in poly.exterior.coords:
                coords.append((x - origin_x, y - origin_y))
    return coords


def get_road_coordinates_by_type(polygon_wgs84: Polygon) -> Dict[str, List[Tuple[float, float]]]:
    """
    Extracts road coordinates by type, relative to the zone's origin.

    Args:
        polygon_wgs84 (Polygon): Polygon of the zone.

    Returns:
        Dict[str, List[Tuple[float, float]]]: Dictionary where keys are road types
            and values are lists of (x, y) coordinates relative to the zone's origin.
            If a road type is not found, its list of coordinates will be empty.
    """
    # Initialize the dictionary with all possible road types and empty lists
    road_coords_by_type: Dict[str, List[Tuple[float, float]]] = {
        road_type: [] for road_type in ROAD_WIDTH_MAP.keys()
    }

    # 1. Reproject polygon to TARGET_CRS to get its bounds in meters
    transformer = pyproj.Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True).transform
    polygon_proj = shapely_transform(transformer, polygon_wgs84)

    minx, miny, _, _ = polygon_proj.bounds

    # 2. Query Overpass and process result
    query = build_overpass_query(polygon_wgs84)
    overpass_result = query_overpass(query)
    gdf = process_overpass_result(overpass_result)

    # 3. Assign feature values and buffer distances
    gdf_with_values = assign_feature_values(gdf, FEATURE_VALUE_MAP, TAG_TYPE_PRIORITY)

    # 4. Filter for roads and reproject to TARGET_CRS
    roads_gdf = gdf_with_values[gdf_with_values['highway'].notna()].to_crs(TARGET_CRS)

    # 5. Extract and store coordinates
    for _, row in roads_gdf.iterrows():
        highway_type = row['highway']
        geom = row.geometry
        buffer_dist = row.get('buffer_distance', 0.0)

        # Apply buffering for LineStrings to represent road width
        if isinstance(geom, LineString) and buffer_dist > 0:
            buffered_geom = geom.buffer(buffer_dist, cap_style=3, join_style=3)
            # If buffering results in MultiPolygon, iterate through its parts
            if buffered_geom.geom_type == 'MultiPolygon':
                for single_poly in buffered_geom.geoms:
                    road_coords_by_type[highway_type].extend(_get_coordinates_from_geometry(single_poly, minx, miny))
            else:
                road_coords_by_type[highway_type].extend(_get_coordinates_from_geometry(buffered_geom, minx, miny))
        elif geom: # For other geometry types (e.g., existing Polygons with highway tag) or no buffer
            road_coords_by_type[highway_type].extend(_get_coordinates_from_geometry(geom, minx, miny))

    return road_coords_by_type


def get_terrain_coordinates_by_type(polygon_wgs84: Polygon) -> Dict[str, List[Tuple[float, float]]]:
    """
    Extracts terrain coordinates by type, relative to the zone's origin.

    Args:
        polygon_wgs84 (Polygon): Polygon of the zone.

    Returns:
        Dict[str, List[Tuple[float, float]]]: Dictionary where keys are terrain types
            (natural, landuse, surface) and values are lists of (x, y) coordinates
            relative to the zone's origin. If a terrain type is not found, its list
            of coordinates will be empty.
    """
    # Initialize the dictionary with all possible terrain types and empty lists
    terrain_coords_by_type: Dict[str, List[Tuple[float, float]]] = {}
    for tag_type in ["natural", "landuse", "surface"]:
        for terrain_type in FEATURE_VALUE_MAP.get(tag_type, {}).keys():
            terrain_coords_by_type[terrain_type] = []

    # 1. Reproject polygon to TARGET_CRS to get its bounds in meters
    transformer = pyproj.Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True).transform
    polygon_proj = shapely_transform(transformer, polygon_wgs84)

    minx, miny, _, _ = polygon_proj.bounds

    # 2. Query Overpass and process result
    query = build_overpass_query(polygon_wgs84)
    overpass_result = query_overpass(query)
    gdf = process_overpass_result(overpass_result)

    # 3. Assign feature values (not strictly needed for terrain, but good for consistency)
    gdf_with_values = assign_feature_values(gdf, FEATURE_VALUE_MAP, TAG_TYPE_PRIORITY)

    # 4. Filter for terrain and reproject to TARGET_CRS
    # Terrain is anything that is not a 'highway' type
    terrain_gdf = gdf_with_values[gdf_with_values['highway'].isna()].to_crs(TARGET_CRS)

    # 5. Extract and store coordinates
    for _, row in terrain_gdf.iterrows():
        # Determine the specific terrain type (e.g., 'forest', 'sand')
        terrain_type = None
        for tag_type in ["natural", "landuse", "surface"]:
            if tag_type in row and row[tag_type] in FEATURE_VALUE_MAP.get(tag_type, {}):
                terrain_type = row[tag_type]
                break
        
        if terrain_type and terrain_type in terrain_coords_by_type:
            geom = row.geometry
            # For terrain, we generally don't buffer lines unless explicitly specified
            # We just extract the coordinates of the geometry as is
            if geom.geom_type == 'MultiPolygon':
                for single_poly in geom.geoms:
                    terrain_coords_by_type[terrain_type].extend(_get_coordinates_from_geometry(single_poly, minx, miny))
            else:
                terrain_coords_by_type[terrain_type].extend(_get_coordinates_from_geometry(geom, minx, miny))

    return terrain_coords_by_type


def main():
    """
    Main function to orchestrate the process of fetching, processing,
    rasterizing, and displaying OSM data.
    """
    # 1. Load polygon from GeoJSON
    print("Loading and reprojecting polygon from GeoJSON...")
    with open(GEOJSON_FILEPATH, 'r') as f:
        poly_data = json.load(f)["features"][0]["geometry"]
    polygon_wgs84 = Polygon(poly_data["coordinates"][0])

    # Test get_coordinates functions
    road_coords_by_type = get_road_coordinates_by_type(polygon_wgs84)
    print('ROAD COORDINATES:' + '-'*30)
    for road_type, coords in road_coords_by_type.items():
        print('-'*20)
        print(f'{road_type}')
        if len(coords) > 0:
            print(f'\tSize: {len(coords)}  |  minx: {min(coord[0] for coord in coords)}  |  miny: {min(coord[1] for coord in coords)} | maxx: {max(coord[0] for coord in coords)}  |  maxy: {max(coord[1] for coord in coords)}')
        else:
            print(f'\tSize: {len(coords)}')
    
    terrain_coords_by_type = get_terrain_coordinates_by_type(polygon_wgs84)
    print('TERRAIN COORDINATES:' + '-'*30)
    for terrain_type, coords in terrain_coords_by_type.items():
        print('-'*20)
        print(f'{terrain_type}')
        if len(coords) > 0:
            print(f'\tSize: {len(coords)}  |  minx: {min(coord[0] for coord in coords)}  |  miny: {min(coord[1] for coord in coords)} | maxx: {max(coord[0] for coord in coords)}  |  maxy: {max(coord[1] for coord in coords)}')
        else:
            print(f'\tSize: {len(coords)}')

    exit(0)


    # Create a GeoSeries from the polygon to handle reprojection easily
    # The input GeoJSON is assumed to be in WGS84 (EPSG:4326)
    poly_gs = gpd.GeoSeries([polygon_wgs84], crs="EPSG:4326")

    # Reproject the polygon to the target CRS to be used for clipping
    poly_gs_proj = poly_gs.to_crs(TARGET_CRS)
    polygon_proj = poly_gs_proj.iloc[0]

    polygon = polygon_wgs84 # Use original WGS84 polygon for Overpass query

    # 2. Create a combined Overpass query for both features and roads
    print("Building Overpass queries...")
    query = build_overpass_query(polygon)
    overpass_result = query_overpass(query)

    # 3. Process Overpass result into a GeoDataFrame
    gdf = process_overpass_result(overpass_result)


    # 4. Assign numerical values to features
    print("Assigning feature values based on priority and mapping...")
    gdf_with_values = assign_feature_values(gdf, FEATURE_VALUE_MAP, TAG_TYPE_PRIORITY)

    # Separate GeoDataFrames for roads and terrain based on the assigned values
    roads_gdf = gdf_with_values[gdf_with_values['value'] >= 31].copy() # Roads are values >= 31
    terrain_gdf = gdf_with_values[gdf_with_values['value'] < 31].copy() # Terrain is everything else

    # 5. Rasterize geometries
    print("Rasterizing all features...")
    raster_array, extent, _ = rasterize_geometries(gdf_with_values, RASTER_RESOLUTION_METERS, TARGET_CRS, polygon_proj)
    print("Rasterizing roads...")
    raster_array_roads, extent_roads, _ = rasterize_geometries(roads_gdf, RASTER_RESOLUTION_METERS, TARGET_CRS, polygon_proj)
    print("Rasterizing terrain...")
    raster_array_terrain, extent_terrain, _ = rasterize_geometries(terrain_gdf, RASTER_RESOLUTION_METERS, TARGET_CRS, polygon_proj)

    # 6. Create custom colormap
    custom_cmap = create_custom_colormap(FEATURE_VALUE_MAP)
    

    # 7. Display the raster
    print("Displaying raster for all features...")
    display_raster(raster_array,         extent,         custom_cmap, FEATURE_VALUE_MAP, RASTER_RESOLUTION_METERS, output_filepath="raster_all_features.png")
    print("Displaying raster for roads...")
    display_raster(raster_array_roads,   extent_roads,   custom_cmap, FEATURE_VALUE_MAP, RASTER_RESOLUTION_METERS, output_filepath="raster_all_features_roads.png")
    print("Displaying raster for terrain...")
    display_raster(raster_array_terrain, extent_terrain, custom_cmap, FEATURE_VALUE_MAP, RASTER_RESOLUTION_METERS, output_filepath="raster_all_features_terrain.png")


if __name__ == '__main__':
    main()