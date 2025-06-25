import overpy
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, Polygon


import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors





geojson_filepath = Path("data/zone_test_caussol.geojson")
poly = json.loads(open(geojson_filepath).read())["features"][0]["geometry"]
coords = poly["coordinates"][0]
poly_str = " ".join(f"{lat} {lon}" for lon, lat in coords)

query = f"""
    [out:json][timeout:60];
    (
    // LAND COVER & NATURAL FEATURES
    way["natural"~"^(sand|glacier|bare_rock|rock|scrub|heath|wood|grassland|wetland|shingle)$"](poly:"{poly_str}");
    way["landuse"~"^(forest|farmland|meadow|grass|quarry|residential|industrial|recreation_ground)$"](poly:"{poly_str}");

    // SURFACE TYPES (from roads, paths, etc.)
    way["surface"~"^(dirt|gravel|sand|grass|mud|paved|asphalt)$"](poly:"{poly_str}");

    // Also include similar tags on relations
    relation["natural"](poly:"{poly_str}");
    relation["landuse"](poly:"{poly_str}");
    );
    out body;
    >;
    out skel qt;
"""

api = overpy.Overpass()
res = api.query(query)




features = []
for way in res.ways:
    tags = way.tags
    if any(k in tags for k in ("landuse", "natural", "surface")):
        nodes = way.nodes
        if len(nodes) < 2:
            continue
        
        way_coords = [(float(n.lon), float(n.lat)) for n in nodes]
        
        # If the way is closed, treat it as a Polygon
        if len(way_coords) > 3 and way_coords[0] == way_coords[-1]:
            geom = Polygon(way_coords)
        else:
            geom = LineString(way_coords)
        features.append({"geometry": geom, **tags})

# --- Feature Categorization and Value Assignment ---

# Define a mapping from specific tags to integer values
# Higher values will be drawn on top (last in rasterize, brighter in colormap)
feature_value_map = {
    "natural": {
        "sand": 1, "glacier": 2, "bare_rock": 3, "rock": 4, "scrub": 5,
        "heath": 6, "wood": 7, "grassland": 8, "wetland": 9, "shingle": 10,
    },
    "landuse": {
        "forest": 11, "farmland": 12, "meadow": 13, "grass": 14, "quarry": 15,
        "residential": 16, "industrial": 17, "recreation_ground": 18,
    },
    "surface": {
        "dirt": 21, "gravel": 22, "sand": 23, "grass": 24, "mud": 25,
        "paved": 26, "asphalt": 27,
    }
}

# Define a priority for tag types (surface should be on top)
tag_type_priority = ["surface", "landuse", "natural"]

def get_feature_value(row):
    """
    Assigns a numerical value to a feature based on its tags and a defined priority.
    """
    for tag_type in tag_type_priority:
        # Check if the row has this tag_type as a column and its value is in our map
        if tag_type in row and row[tag_type] in feature_value_map[tag_type]:
            return feature_value_map[tag_type][row[tag_type]]
    return 0 # Default for unknown or unclassified features

gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
# Apply this function to create a 'value' column in the GeoDataFrame
gdf['value'] = gdf.apply(get_feature_value, axis=1)

# Sort the GeoDataFrame by the 'value' column to ensure correct z-ordering during rasterization
gdf = gdf.sort_values(by='value', ascending=True)




# --- Rasterization and Display ---

# 1. Determine bounds and resolution for the raster
bounds = gdf.total_bounds
width = bounds[2] - bounds[0]
height = bounds[3] - bounds[1]
resolution = 0.0001  # Adjust as needed for desired resolution
out_shape = (int(height / resolution), int(width / resolution))

# 2. Create the affine transform
transform = rasterio.transform.from_bounds(
    west=bounds[0],
    south=bounds[1],
    east=bounds[2],
    north=bounds[3],
    width=out_shape[1],
    height=out_shape[0],
)

# 3. Rasterize the geometries
#    Assign the numerical 'value' to areas covered by features, 0 otherwise.
raster = rasterize( # type: ignore
    [(feature.geometry, feature['value']) for idx, feature in gdf.iterrows()],
    out_shape=out_shape,
    transform=transform,
    fill=0,
    all_touched=True  # Include all pixels touched by geometries
)


# --- Custom Colormap and Display ---

# Example colors (you'd want to refine these for better visual distinction)
# Ensure the list is long enough for all assigned values (up to max_value)
colors_list = [
    (0, 0, 0, 0), # Value 0 (transparent or background)
    # Natural (1-10) - earthy/natural tones
    (0.8, 0.8, 0.6), # sand (light brown)
    (0.9, 0.9, 0.9), # glacier (white)
    (0.5, 0.5, 0.5), # bare_rock (grey)
    (0.4, 0.4, 0.4), # rock (dark grey)
    (0.6, 0.7, 0.5), # scrub (light green)
    (0.7, 0.8, 0.6), # heath (pale green)
    (0.2, 0.5, 0.2), # wood (dark green)
    (0.6, 0.8, 0.4), # grassland (medium green)
    (0.4, 0.6, 0.8), # wetland (light blue)
    (0.7, 0.7, 0.7), # shingle (grey-white)
    # Landuse (11-20) - more human-influenced/broader categories
    (0.1, 0.4, 0.1), # forest (darker green)
    (0.8, 0.7, 0.3), # farmland (yellow-brown)
    (0.5, 0.7, 0.3), # meadow (vibrant green)
    (0.4, 0.6, 0.2), # grass (landuse) (medium green)
    (0.7, 0.7, 0.7), # quarry (light grey)
    (0.9, 0.4, 0.4), # residential (light red)
    (0.6, 0.3, 0.3), # industrial (dark red)
    (0.4, 0.8, 0.8), # recreation_ground (cyan)
    # Surface (21-30) - road/path colors (should be distinct and on top)
    (0.6, 0.4, 0.2), # dirt (brown)
    (0.5, 0.5, 0.5), # gravel (medium grey)
    (0.8, 0.8, 0.6), # sand (surface) (light brown)
    (0.4, 0.6, 0.2), # grass (surface) (medium green)
    (0.5, 0.3, 0.1), # mud (dark brown)
    (0.3, 0.3, 0.3), # paved (darker grey)
    (0.2, 0.2, 0.2), # asphalt (very dark grey)
]

# Ensure the list is long enough for all values, fill with black if not enough
max_value = max(v for d in feature_value_map.values() for v in d.values())
while len(colors_list) <= max_value:
    colors_list.append((0, 0, 0, 1)) # Default to black for unassigned values




# 4. Display the raster
fig, ax = plt.subplots(figsize=(12, 12)) # Make figure slightly larger for colorbar
extent = [bounds[0], bounds[2], bounds[1], bounds[3]]  # for matplotlib's imshow
# Create a custom colormap from the defined colors
custom_cmap = mcolors.ListedColormap(colors_list)

im = ax.imshow(raster, extent=extent, cmap=custom_cmap, vmin=0, vmax=max_value)
ax.set_title("Rasterized Features by Type")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Add a colorbar to explain the colors
# Build the reverse mapping for colorbar labels
colorbar_labels_map = {}
for tag_type, values_dict in feature_value_map.items():
    for tag_value, int_id in values_dict.items():
        colorbar_labels_map[int_id] = tag_value # Use the specific tag value as label

# Get unique assigned values (excluding 0 for background)
unique_assigned_values = sorted([v for v in gdf['value'].unique() if v != 0])

# Prepare colorbar ticks and labels
cbar_ticks = unique_assigned_values
cbar_tick_labels = [colorbar_labels_map.get(v, 'Unknown') for v in unique_assigned_values]

cbar = fig.colorbar(im, ax=ax, ticks=cbar_ticks, orientation='vertical', shrink=0.75)
cbar.ax.set_yticklabels(cbar_tick_labels)
cbar.set_label('Feature Type')

plt.show()