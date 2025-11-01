import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import overpy
import pandas as pd
import pyproj
import rasterio
import rasterio.transform
from rasterio.features import rasterize
from shapely.geometry import LineString, Polygon
from shapely.ops import transform as shapely_transform

from src import config


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
        if any(k in tags for k in config.OSM_TAG_TYPE_PRIORITY):
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
    gdf: gpd.GeoDataFrame, feature_map: dict, priority: list
) -> gpd.GeoDataFrame:
    """
    Assigns a numerical value and buffer distance based on configuration.
    """
    def _get_value_and_buffer_distance(row: dict) -> tuple[int, float]:
        value = 0
        buffer_dist = 0.0
        for tag_type in priority:
            if tag_type in row and row[tag_type] in feature_map.get(tag_type, {}):
                value = feature_map[tag_type][row[tag_type]]
                if tag_type == "highway" and row[tag_type] in config.OSM_ROAD_WIDTH_MAP:
                    # Use the road width map from the config file
                    buffer_dist = config.OSM_ROAD_WIDTH_MAP[row[tag_type]] / 2.0
                break
        return value, buffer_dist

    gdf[['value', 'buffer_distance']] = gdf.apply(lambda row: pd.Series(_get_value_and_buffer_distance(row)), axis=1)
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
    transformer = pyproj.Transformer.from_crs("EPSG:4326", config.TARGET_CRS, always_xy=True).transform
    polygon_proj = shapely_transform(transformer, polygon_wgs84)
    minx, miny, _, _ = polygon_proj.bounds

    # 2. Query Overpass and process result
    query = build_overpass_query(polygon_wgs84)
    overpass_result = query_overpass(query)
    gdf = process_overpass_result(overpass_result)
    if gdf.empty:
        return {road_type: [] for road_type in config.OSM_ROAD_WIDTH_MAP.keys()}

    # 3. Assign feature values and buffer distances
    gdf_with_values = assign_feature_values(gdf, config.OSM_FEATURE_VALUE_MAP, config.OSM_TAG_TYPE_PRIORITY)

    # 4. Filter for roads
    roads_gdf = gdf_with_values[gdf_with_values['highway'].notna()].copy()
    
    # 5. Rasterize roads by type using the 'highway' column
    road_rasters, transform = rasterize_geometries(
        gdf=roads_gdf,
        resolution_meters=config.RASTER_RESOLUTION_METERS,
        target_crs=config.TARGET_CRS,
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
    final_coords = {road_type: [] for road_type in config.OSM_ROAD_WIDTH_MAP.keys()}
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
    transformer = pyproj.Transformer.from_crs("EPSG:4326", config.TARGET_CRS, always_xy=True).transform
    polygon_proj = shapely_transform(transformer, polygon_wgs84)
    minx, miny, _, _ = polygon_proj.bounds
    
    # Initialize the result dictionary with all possible terrain types
    all_terrain_types = {}
    for tag_type in ["natural", "landuse", "surface"]:
        for terrain_type in config.OSM_FEATURE_VALUE_MAP.get(tag_type, {}).keys():
            all_terrain_types[terrain_type] = []

    # 2. Query Overpass and process result
    query = build_overpass_query(polygon_wgs84)
    overpass_result = query_overpass(query)
    gdf = process_overpass_result(overpass_result)
    if gdf.empty:
        return all_terrain_types

    # 3. Assign feature values
    gdf_with_values = assign_feature_values(gdf, config.OSM_FEATURE_VALUE_MAP, config.OSM_TAG_TYPE_PRIORITY)

    # 4. Filter for terrain and assign a 'terrain_type' string for grouping
    terrain_gdf = gdf_with_values[gdf_with_values['highway'].isna()].copy()
    
    def _get_terrain_type(row):
        # Find the highest priority terrain tag based on the defined order
        for tag_type in ["surface", "landuse", "natural"]:
            if tag_type in row and pd.notna(row[tag_type]) and row[tag_type] in config.OSM_FEATURE_VALUE_MAP.get(tag_type, {}):
                return row[tag_type]
        return None
        
    terrain_gdf['terrain_type'] = terrain_gdf.apply(_get_terrain_type, axis=1)
    terrain_gdf.dropna(subset=['terrain_type'], inplace=True)

    # 5. Rasterize terrain by its determined type
    terrain_rasters, transform = rasterize_geometries(
        gdf=terrain_gdf,
        resolution_meters=config.RASTER_RESOLUTION_METERS,
        target_crs=config.TARGET_CRS,
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