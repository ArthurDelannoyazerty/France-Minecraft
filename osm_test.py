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
ROAD_WIDTH_MAP = {
    "motorway": 3.5 * 2 + 3.0 * 2,        # 2×3,5 m voies + 2×3 m BAU ≈ 13 m emprise minimal
    "trunk": 3.5 * 2 + 1.5 * 2,          # 2×3,5 m + accotements ≈ 10 m
    "primary": 3.5 * 2 + 1.5 * 2,        # voie principal similaire au "trunk"
    "secondary": 3.25 * 2 + 1.0 * 2,     # voies en zone urbaine ou routes tampons ≈ 8,5 m
    "tertiary": 3.0 * 2 + 0.75 * 2,      # petites routes rurales ≈ 7,5 m
    "unclassified": 3.0 * 2,            # 2 voies sans accotement ≈ 6 m
    "residential": 3.0 * 2,             # voiries urbaines ≈ 6 m
    "service": 3.0,                      # accès ponctuels ≈ 3 m
    "living_street": 3.0,                # zones 20 km/h ≈ 3 m
    "pedestrian": 1.0,                   # voiries partagées
    "footway": 1.0,                      # trottoirs ou chemins piétons ≈ 2 m
    "cycleway": 2.5,                     # piste unidirectionnelle ≈ 2 à 2,5 m
    "path": 3.0,                         # chemin rural ≈ 3 m
    "track": 3.0,                        # voies agricoles ≈ 3 m
    "steps": 1.5,                        # escaliers ≈ 1,5 m
    "bridleway": 2.5,                    # voies équestres ≈ 2–3 m
    "raceway": 12.0,                     # voie sportive ou circuit ≈ 10–15 m
    "bus_guideway": 3.25,                # voie bus ≈ 3–3,5 m
    "corridor": 3.0,                     # couloirs partagés ≈ 3 m
    "elevator": 2.0,                     # ascenseurs extérieurs ≈ 2 m
    "escalator": 2.0,                    # escalators ≈ 2 m
    "platform": 5.0,                     # quai de gare ≈ 5 m
    "proposed": 3.0,                     # estimation parcellaire ≈ 3 m
    "construction": 3.0,                 # chantier temporaire ≈ 3 m
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
    clip_polygon_proj: Polygon,
    type_column: str,
) -> Tuple[Dict[str, np.ndarray], rasterio.transform.Affine]:
    """
    Rasterizes GeoDataFrame geometries, creating a separate raster for each type.
    Each raster is a boolean mask for a specific feature type.
    The function handles reprojection, clipping, and buffering of LineStrings.

    Args:
        gdf: GeoDataFrame with features. Must contain the `type_column` and a
             'buffer_distance' column for LineString features.
        resolution_meters: The desired resolution of the output rasters in meters.
        target_crs: The target CRS for rasterization (e.g., "EPSG:2154").
        clip_polygon_proj: A shapely Polygon in the target CRS to define the
                           raster bounds and clip geometries.
        type_column: The name of the column in the GDF to group features by
                     (e.g., 'highway', 'terrain_type').

    Returns:
        A tuple containing:
        - A dictionary where keys are feature types from `type_column` and values
          are the corresponding numpy raster arrays (masks).
        - The affine transform for the rasters.
    """
    if gdf.empty:
        print("GeoDataFrame is empty, cannot rasterize.")
        return {}, rasterio.transform.Affine.identity()

    gdf_proj = gdf.to_crs(target_crs)
    gdf_proj_clipped = gdf_proj.clip(clip_polygon_proj)

    if gdf_proj_clipped.empty:
        print("GeoDataFrame is empty after clipping, cannot rasterize.")
        return {}, rasterio.transform.Affine.identity()

    minx, miny, maxx, maxy = clip_polygon_proj.bounds
    out_shape = (
        int(np.ceil((maxy - miny) / resolution_meters)),
        int(np.ceil((maxx - minx) / resolution_meters)),
    )
    transform = rasterio.transform.from_bounds(
        west=minx, south=miny, east=maxx, north=maxy,
        width=out_shape[1], height=out_shape[0]
    )

    rasters_by_type = {}
    
    # Sort by value to ensure higher-priority features are processed last,
    # which can matter if geometries of the same type overlap.
    if 'value' in gdf_proj_clipped.columns:
        gdf_proj_clipped = gdf_proj_clipped.sort_values(by='value', ascending=True)

    # Group by the specified type column (e.g., 'highway', 'terrain_type')
    for feature_type, group in gdf_proj_clipped.groupby(type_column):
        if pd.isna(feature_type):
            continue

        geometries_to_rasterize = []
        for _, row in group.iterrows():
            geom = row.geometry
            buffer_dist = row.get('buffer_distance', 0.0)

            if isinstance(geom, LineString) and buffer_dist > 0:
                buffered_geom = geom.buffer(buffer_dist, cap_style=3, join_style=3)
                geometries_to_rasterize.append((buffered_geom, 1))
            elif isinstance(geom, (Polygon, LineString)):
                geometries_to_rasterize.append((geom, 1))

        if not geometries_to_rasterize:
            continue

        # Rasterize this group's geometries into a single mask
        raster = rasterize(
            geometries_to_rasterize,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            default_value=0,
            all_touched=True,
            dtype='uint8'
        )
        rasters_by_type[feature_type] = raster

    return rasters_by_type, transform


def get_road_coordinates_by_type(polygon_wgs84: Polygon) -> Dict[str, List[Tuple[float, float]]]:
    """
    Extracts road coordinates by type using a raster-based method, relative to
    the zone's origin.

    Args:
        polygon_wgs84 (Polygon): Polygon of the zone in WGS84.

    Returns:
        Dict[str, List[Tuple[float, float]]]: Dictionary where keys are road types
            and values are lists of (x, y) coordinates relative to the zone's origin.
            If a road type is not found, its list of coordinates will be empty.
    """
    # 1. Reproject polygon to TARGET_CRS to get its bounds and origin in meters
    transformer = pyproj.Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True).transform
    polygon_proj = shapely_transform(transformer, polygon_wgs84)
    minx, miny, _, _ = polygon_proj.bounds

    # 2. Query Overpass and process result
    query = build_overpass_query(polygon_wgs84)
    overpass_result = query_overpass(query)
    gdf = process_overpass_result(overpass_result)
    if gdf.empty:
        return {road_type: [] for road_type in ROAD_WIDTH_MAP.keys()}

    # 3. Assign feature values and buffer distances
    gdf_with_values = assign_feature_values(gdf, FEATURE_VALUE_MAP, TAG_TYPE_PRIORITY)

    # 4. Filter for roads
    roads_gdf = gdf_with_values[gdf_with_values['highway'].notna()].copy()
    
    # 5. Rasterize roads by type using the 'highway' column
    road_rasters, transform = rasterize_geometries(
        gdf=roads_gdf,
        resolution_meters=RASTER_RESOLUTION_METERS,
        target_crs=TARGET_CRS,
        clip_polygon_proj=polygon_proj,
        type_column='highway'
    )

    # 6. Convert raster masks to relative coordinates
    road_coords_by_type = {}
    for road_type, raster in road_rasters.items():
        # Find pixel indices where the raster is not zero
        rows, cols = np.where(raster > 0)
        
        # Convert pixel indices to world coordinates (in TARGET_CRS)
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        
        # Make coordinates relative to the bottom-left corner (minx, miny)
        # This makes the origin (0,0) of the output coordinate system correspond
        # to the bottom-left corner of the zone's bounding box.
        relative_coords = [(x - minx, y - miny) for x, y in zip(xs, ys)]
        road_coords_by_type[road_type] = relative_coords
        
    # Ensure the final dictionary contains all possible road types
    final_coords = {road_type: [] for road_type in ROAD_WIDTH_MAP.keys()}
    final_coords.update(road_coords_by_type)
    
    return final_coords


def get_terrain_coordinates_by_type(polygon_wgs84: Polygon) -> Dict[str, List[Tuple[float, float]]]:
    """
    Extracts terrain coordinates by type using a raster-based method, relative
    to the zone's origin.

    Args:
        polygon_wgs84 (Polygon): Polygon of the zone in WGS84.

    Returns:
        Dict[str, List[Tuple[float, float]]]: Dictionary where keys are terrain types
            and values are lists of (x, y) coordinates relative to the zone's origin.
            If a terrain type is not found, its list of coordinates will be empty.
    """
    # 1. Reproject polygon to TARGET_CRS to get its bounds and origin in meters
    transformer = pyproj.Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True).transform
    polygon_proj = shapely_transform(transformer, polygon_wgs84)
    minx, miny, _, _ = polygon_proj.bounds
    
    # Initialize the result dictionary with all possible terrain types
    all_terrain_types = {}
    for tag_type in ["natural", "landuse", "surface"]:
        for terrain_type in FEATURE_VALUE_MAP.get(tag_type, {}).keys():
            all_terrain_types[terrain_type] = []

    # 2. Query Overpass and process result
    query = build_overpass_query(polygon_wgs84)
    overpass_result = query_overpass(query)
    gdf = process_overpass_result(overpass_result)
    if gdf.empty:
        return all_terrain_types

    # 3. Assign feature values
    gdf_with_values = assign_feature_values(gdf, FEATURE_VALUE_MAP, TAG_TYPE_PRIORITY)

    # 4. Filter for terrain and assign a 'terrain_type' string for grouping
    terrain_gdf = gdf_with_values[gdf_with_values['highway'].isna()].copy()
    
    def _get_terrain_type(row):
        # Find the highest priority terrain tag based on the defined order
        for tag_type in ["surface", "landuse", "natural"]:
            if tag_type in row and pd.notna(row[tag_type]) and row[tag_type] in FEATURE_VALUE_MAP.get(tag_type, {}):
                return row[tag_type]
        return None
        
    terrain_gdf['terrain_type'] = terrain_gdf.apply(_get_terrain_type, axis=1)
    terrain_gdf.dropna(subset=['terrain_type'], inplace=True)

    # 5. Rasterize terrain by its determined type
    terrain_rasters, transform = rasterize_geometries(
        gdf=terrain_gdf,
        resolution_meters=RASTER_RESOLUTION_METERS,
        target_crs=TARGET_CRS,
        clip_polygon_proj=polygon_proj,
        type_column='terrain_type'
    )
    
    # 6. Convert raster masks to relative coordinates
    terrain_coords_by_type = {}
    for terrain_type, raster in terrain_rasters.items():
        rows, cols = np.where(raster > 0)
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        relative_coords = [(x - minx, y - miny) for x, y in zip(xs, ys)]
        terrain_coords_by_type[terrain_type] = relative_coords

    # Ensure the final dictionary contains all possible terrain types
    all_terrain_types.update(terrain_coords_by_type)
    return all_terrain_types


def display_road_coordinates(
    road_coords_by_type: Dict[str, List[Tuple[float, float]]],
    title: str,
    output_filepath: str
) -> None:
    """
    Displays the extracted road coordinates as a scatter plot with different colors
    for each road type.

    Args:
        road_coords_by_type: Dictionary from road type to list of (x, y) coordinates.
        title: The title for the plot.
        output_filepath: The path to save the output image file.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('black') # Use a black background for better visibility
    ax.set_aspect('equal', adjustable='box')

    # Get a list of road types that actually have coordinates
    found_road_types = [k for k, v in road_coords_by_type.items() if v]
    
    if not found_road_types:
        print("No road coordinates to display.")
        plt.close(fig)
        return

    # Create a color map to assign a unique color to each road type
    # Using 'tab20' which has 20 distinct colors, good for categorical data
    color_map = plt.get_cmap('tab20', len(found_road_types))
    
    for i, (road_type, coords) in enumerate(road_coords_by_type.items()):
        if not coords:
            continue  # Skip empty lists

        # Unpack the list of tuples into two lists: x_vals and y_vals
        x_vals, y_vals = zip(*coords)
        
        ax.scatter(
            x_vals, 
            y_vals, 
            color=color_map(i), 
            label=road_type,
            s=1,          # Use small points for dense data
            marker='.'    # Use a pixel marker
        )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Meters from Origin (X)")
    ax.set_ylabel("Meters from Origin (Y)")
    ax.legend(markerscale=10) # Make legend markers larger and more visible
    plt.grid(True, linestyle='--', alpha=0.2)

    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    print(f"Saved road coordinate plot to: {output_filepath}")
    plt.close(fig) # Close the figure to free up memory


def main():
    """
    Main function to orchestrate the process of fetching and processing OSM data
    to extract road and terrain coordinates.
    """
    # 1. Load polygon from GeoJSON
    print("Loading polygon from GeoJSON...")
    # Make sure to create this file or replace with your own
    geojson_filepath = Path("data/zone_test_tile.geojson")
    if not geojson_filepath.exists():
        print(f"Error: GeoJSON file not found at {geojson_filepath}")
        print("Please create a 'data' directory and place your GeoJSON file in it.")
        # Create a dummy polygon for demonstration if file doesn't exist
        print("Using a dummy polygon over Paris for demonstration.")
        polygon_wgs84 = Polygon.from_bounds(2.34, 48.85, 2.36, 48.86)
    else:
        with open(geojson_filepath, 'r') as f:
            poly_data = json.load(f)
        
        # The example file seems to be in Lambert-93, so we convert to WGS84
        polygon_lambert93 = Polygon(poly_data["geometry"]["coordinates"][0])
        transformer = pyproj.Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform
        polygon_wgs84 = shapely_transform(transformer, polygon_lambert93)

    # 2. Test the refactored coordinate extraction functions
    print("\n--- Getting Road Coordinates ---")
    road_coords_by_type = get_road_coordinates_by_type(polygon_wgs84)
    print('ROAD COORDINATES:' + '-'*30)
    for road_type, coords in road_coords_by_type.items():
        if coords:
            print(f"- {road_type}: Found {len(coords)} coordinate points.")
        else:
            print(f"- {road_type}: Not found in the area.")

    print("\n--- Getting Terrain Coordinates ---")
    terrain_coords_by_type = get_terrain_coordinates_by_type(polygon_wgs84)
    print('TERRAIN COORDINATES:' + '-'*30)
    for terrain_type, coords in terrain_coords_by_type.items():
        if coords:
            print(f"- {terrain_type}: Found {len(coords)} coordinate points.")
        else:
            print(f"- {terrain_type}: Not found in the area.")

    # 3. NEW: Display the extracted road coordinates
    print("\n--- Displaying Road Coordinates ---")
    display_road_coordinates(
        road_coords_by_type,
        title="Extracted Road Network by Type",
        output_filepath="road_coordinates_plot.png"
    )


if __name__ == '__main__':
    main()