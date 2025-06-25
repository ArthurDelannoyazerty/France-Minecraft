import overpy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import geopandas as gpd
from shapely.geometry import LineString, Polygon
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio.transform
import numpy as np


# --- Configuration Constants ---
GEOJSON_FILEPATH = Path("data/zone_test_caussol.geojson")
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
                "pedestrian": 40, "footway": 41, "cycleway": 42, "path": 43},
    "surface": {"dirt": 21, "gravel": 22, "sand": 23, "grass": 24, "mud": 25,
                "paved": 26, "asphalt": 27}
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

def filter_out_roads(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filters out the road types"""
    return gdf[~gdf['highway'].isin(FEATURE_VALUE_MAP.get('highway', {}).keys())].copy()


def assign_feature_values(
    gdf: gpd.GeoDataFrame, feature_map: Dict[str, Dict[str, int]], priority: List[str]
) -> gpd.GeoDataFrame:
    """
    Assigns a numerical value to each feature based on its tags and a defined priority.
    """

    def _get_value(row: Dict[str, Any]) -> int:
        for tag_type in priority: # Iterate through tag types based on priority
            if tag_type in row and row[tag_type] in feature_map.get(tag_type, {}): # Check if tag exists in row and in the feature_map
                return feature_map[tag_type][row[tag_type]] # Return the mapped integer value
        return 0  # Default for unknown or unclassified features

    gdf['value'] = gdf.apply(_get_value, axis=1)
    # Sort by value to ensure correct z-ordering during rasterization (lower values first)
    return gdf.sort_values(by='value', ascending=True)


def rasterize_geometries(
    gdf: gpd.GeoDataFrame, 
    resolution_meters: float, 
    target_crs: str,
    invert: bool = False # False = terrain ; True = road
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

    # Determine bounds and shape for the raster in the projected CRS
    bounds_proj = gdf_proj.total_bounds
    minx, miny, maxx, maxy = bounds_proj

    width_proj = maxx - minx
    height_proj = maxy - miny

    out_shape = (int(np.ceil(height_proj / resolution_meters)),
                 int(np.ceil(width_proj / resolution_meters)))

    # Create the affine transform for the projected CRS
    transform = rasterio.transform.from_bounds(
        west=minx,
        south=miny,
        east=maxx,
        north=maxy,
        width=out_shape[1],
        height=out_shape[0],
    )

    # Rasterize the geometries using their assigned 'value'
    raster = rasterize(
        [(geom, value) for geom, value in zip(gdf_proj.geometry, gdf_proj['value'])],
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
        # Highway 31-43
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
        print("No raster data to display.")
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


def main():
    """
    Main function to orchestrate the process of fetching, processing,
    rasterizing, and displaying OSM data.
    """
    # 1. Load polygon from GeoJSON
    print("Loading polygon from GeoJSON...")
    with open(GEOJSON_FILEPATH, 'r') as f:
        poly_data = json.load(f)["features"][0]["geometry"]
    polygon = Polygon(poly_data["coordinates"][0])

    # 2. Create a combined Overpass query for both features and roads
    print("Building Overpass queries...")
    query = build_overpass_query(polygon)
    overpass_result = query_overpass(query)

    # 3. Process Overpass result into a GeoDataFrame
    gdf = process_overpass_result(overpass_result)

    # 4. Assign numerical values to features
    print("Assigning feature values based on priority and mapping...")
    gdf = assign_feature_values(gdf, FEATURE_VALUE_MAP, TAG_TYPE_PRIORITY)
    roads_gdf = gdf[gdf['value'] >= 31].copy()
    terrain_gdf = filter_out_roads(gdf)  # Exclude road-related features for terrain

    # 5. Rasterize geometries
    raster_array, extent, _ = rasterize_geometries(gdf, RASTER_RESOLUTION_METERS, TARGET_CRS)
    raster_array_roads, extent_roads, _ = rasterize_geometries(roads_gdf, RASTER_RESOLUTION_METERS, TARGET_CRS)
    raster_array_terrain, extent_terrain, _ = rasterize_geometries(terrain_gdf, RASTER_RESOLUTION_METERS, TARGET_CRS)

    # 6. Create custom colormap
    custom_cmap = create_custom_colormap(FEATURE_VALUE_MAP)


    # 7. Display the raster
    display_raster(raster_array,         extent,         custom_cmap, FEATURE_VALUE_MAP, RASTER_RESOLUTION_METERS, output_filepath="raster_all_features.png")
    display_raster(raster_array_roads,   extent_roads,   custom_cmap, FEATURE_VALUE_MAP, RASTER_RESOLUTION_METERS, output_filepath="raster_all_features_roads.png")
    display_raster(raster_array_terrain, extent_terrain, custom_cmap, FEATURE_VALUE_MAP, RASTER_RESOLUTION_METERS, output_filepath="raster_all_features_terrain.png")


if __name__ == '__main__':
    main()