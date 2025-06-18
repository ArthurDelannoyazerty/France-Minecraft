from tqdm import tqdm
import geopandas as gpd
import numpy as np
from shapely.geometry import box, shape
import math
import osmnx as ox
import matplotlib.pyplot as plt # Keeping plot graph as per original prompt line, though not essential for grid output

ox.settings.cache_folder = 'data/cache'


# --- Configuration ---
print("--- Configuration ---")
place_name = 'Grasse, France'
print(f"Place name: {place_name}")
grid_cell_size = 0.5
print(f"Grid cell size: {grid_cell_size} meters")
target_crs = "EPSG:2154"
print(f"Target CRS: {target_crs}")

# Define approximate road widths (in meters) based on highway type
road_widths_meters = {
    'motorway': 20.0,
    'trunk': 18.0,
    'primary': 15.0,
    'secondary': 12.0,
    'tertiary': 10.0,
    'unclassified': 6.0,
    'residential': 5.0,
    'service': 4.0,
    'living_street': 6.0,
    'pedestrian': 3.0,
    'footway': 2.0,
    'cycleway': 2.5,
    'path': 1.5
}
default_road_width_meters = 5.0
print(f"Defined road widths for various types. Default width: {default_road_width_meters}m")

# Output file path
output_grid_path = 'grasse_road_grid_0_25m.gpkg'
print(f"Output grid path: {output_grid_path}")

print("\n--- Data Acquisition and Processing ---")

# Download road network data for the specified place
print(f"Downloading road network for {place_name}...")
geo = {
  "coordinates": [[
      [-0.4546182727669361, 47.511519817232085],
      [-0.4546182727669361, 47.50255307635143],
      [-0.43919008610097876, 47.50255307635143],
      [-0.43919008610097876, 47.511519817232085],
      [-0.4546182727669361, 47.511519817232085]
    ]],
  "type": "Polygon"
}
polygon = shape(geo)
G = ox.graph_from_polygon(polygon)
print("Download complete.")

# Optional: Plot the downloaded graph
# print("Plotting graph (optional)...")
# ox.plot_graph(G)
# plt.show()

# Convert graph edges to a GeoDataFrame
print("Converting graph edges to GeoDataFrame...")
edges = ox.graph_to_gdfs(G, nodes=False)
print(f"Converted {len(edges)} graph edges.")

# Reproject the edges to the target CRS (Lambert-93)
print(f"Reprojecting edges to {target_crs}...")
original_crs = edges.crs
edges = edges.to_crs(target_crs)
print(f"Reprojected from {original_crs} to {edges.crs}.")



# --- Buffer Roads by Type ---
print("\n--- Buffering Roads by Type ---")

# Function to get buffer distance
def get_buffer_distance(highway_tags):
    if isinstance(highway_tags, list):
        for tag in highway_tags:
            if tag in road_widths_meters:
                return road_widths_meters[tag] / 2.0
        return default_road_width_meters / 2.0
    elif isinstance(highway_tags, str):
        return road_widths_meters.get(highway_tags, default_road_width_meters) / 2.0
    else:
        return default_road_width_meters / 2.0

# Calculate buffer distance for each road segment
print("Calculating buffer distance for each road segment based on type...")
edges['buffer_distance'] = edges.apply(lambda row: get_buffer_distance(row.get('highway')), axis=1)
print("Buffer distances calculated.")

# Filter out edges with zero buffer distance
initial_edge_count = len(edges)
edges = edges[edges['buffer_distance'] > 0]
print(f"Filtered out {initial_edge_count - len(edges)} edges with zero buffer distance.")

# Buffer the road geometries to create surface polygons
print(f"Buffering road geometries using calculated distances...")
road_surface_geometries = edges.geometry.buffer(edges['buffer_distance'])
print("Buffering complete.")

# Create a GeoDataFrame from the buffered geometries
print("Creating GeoDataFrame from buffered geometries...")
road_surface_gdf = gpd.GeoDataFrame(geometry=road_surface_geometries, crs=target_crs)
print(f"Created GeoDataFrame with {len(road_surface_gdf)} buffered geometries.")

# Dissolve overlapping road surface polygons
print("Dissolving overlapping road surface polygons...")
# Check if road_surface_gdf is empty before dissolving
if road_surface_gdf.empty:
    print("Warning: No road surface geometries created. Cannot dissolve or grid.")
    # Create an empty geodataframe for the output and save
    empty_grid = gpd.GeoDataFrame({'geometry': []}, crs=target_crs)
    empty_grid.to_file(output_grid_path, driver='GPKG')
    print(f"Saved empty grid to {output_grid_path}.")
    exit() # Exit the script as there's nothing to grid

all_roads_polygon = road_surface_gdf.dissolve().geometry.iloc[0]
print("Dissolve complete.")


# --- Generate Grid (Accelerated) ---
print("\n--- Generating Grid ---")

# Get the bounding box of the dissolved road geometry
minx, miny, maxx, maxy = all_roads_polygon.bounds
print(f"Bounding box of road surface (in {target_crs}): ({minx}, {miny}, {maxx}, {maxy})")

# Calculate grid dimensions
nx = math.ceil((maxx - minx) / grid_cell_size)
ny = math.ceil((maxy - miny) / grid_cell_size)
print(f"Grid dimensions: {nx} cells in X, {ny} cells in Y.")
print(f"Total potential grid cells covering extent: {nx * ny}")

print(f"Generating {nx*ny} grid cell geometries using NumPy...")
# Generate coordinates for the lower-left corners of all cells
x_coords = minx + np.arange(nx) * grid_cell_size
y_coords = miny + np.arange(ny) * grid_cell_size

# Use meshgrid to get all combinations of x and y coordinates
# Then flatten them
x_grid, y_grid = np.meshgrid(x_coords, y_coords)
all_x1 = x_grid.ravel()
all_y1 = y_grid.ravel()

# Calculate upper-right coordinates
all_x2 = all_x1 + grid_cell_size
all_y2 = all_y1 + grid_cell_size

# Create shapely box geometries using a list comprehension and zip
# This is much faster than nested pure Python loops
grid_cells = [box(x1, y1, x2, y2) for x1, y1, x2, y2 in tqdm(zip(all_x1, all_y1, all_x2, all_y2), total=len(all_x1))]
print(f"Generated {len(grid_cells)} grid cell geometries.")






# Create grid GeoDataFrame
print("Creating grid GeoDataFrame...")
grid_gdf = gpd.GeoDataFrame({'geometry': grid_cells}, crs=target_crs)
print("Grid GeoDataFrame created.")

# --- Select Grid Cells Intersecting Roads ---
print("\n--- Selecting Intersecting Grid Cells ---")

# Prepare dissolved road polygon for spatial join
print("Preparing dissolved road surface for spatial join...")
all_roads_dissolved_gdf = gpd.GeoDataFrame({'geometry': [all_roads_polygon]}, crs=target_crs)
all_roads_dissolved_gdf['dissolve_id'] = 1
print("Preparation complete.")

# Perform spatial join
print("Performing spatial join ('intersects') to find grid cells covering roads...")
intersecting_grid = gpd.sjoin(grid_gdf, all_roads_dissolved_gdf, how="inner", predicate="intersects")
print("Spatial join complete.")

# Keep only the geometry column
initial_intersecting_count = len(intersecting_grid)
intersecting_grid = intersecting_grid[['geometry']]
print(f"Selected {len(intersecting_grid)} grid cells that intersect the road surface.")


# --- Save the Result ---
print("\n--- Saving Result ---")
print(f"Saving the resulting grid to {output_grid_path}...")
intersecting_grid.to_file(output_grid_path, driver='GPKG')
# intersecting_grid('a.json')
print("Save complete.")

print("\n--- Process Finished ---")