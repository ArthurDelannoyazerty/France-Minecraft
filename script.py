import numpy as np
import pyvista as pv
import time
import math # Keep if comparing with original find function

# --- Updated Point Generation ---

def create_normal_xy_points(num_points=30,
                            cube_side_length=10.0,
                            mean_xy=None,
                            std_dev_xy=1.5, # Controls spread on X/Y
                            clip_to_bounds=True):
    """
    Generates points within a cube defined by [0, cube_side_length]^3.
    X and Y coordinates follow a normal distribution centered at mean_xy
    with standard deviation std_dev_xy.
    Z coordinates follow a uniform distribution [0, cube_side_length).

    Args:
        num_points (int): The number of points to generate.
        cube_side_length (float): The side length of the cube.
        mean_xy (float, optional): The center (mean) for the normal distribution
                                   on the X and Y axes. Defaults to the midpoint
                                   of the cube_side_length.
        std_dev_xy (float): The standard deviation for the normal distribution
                            on the X and Y axes. Controls the horizontal spread.
        clip_to_bounds (bool): If True (default), clips the generated X and Y
                               coordinates to stay within [0, cube_side_length].

    Returns:
        numpy.ndarray: An array of shape (num_points, 3) containing the points [x, y, z].
    """
    if mean_xy is None:
        mean_xy = cube_side_length / 2.0

    # Generate X and Y coordinates from a normal distribution
    x_coords = np.random.normal(loc=mean_xy, scale=std_dev_xy, size=num_points)
    y_coords = np.random.normal(loc=mean_xy, scale=std_dev_xy, size=num_points)

    # Generate Z coordinates from a uniform distribution
    z_coords = np.random.uniform(low=0.0, high=cube_side_length, size=num_points)

    # Optionally clip X and Y to the cube boundaries
    if clip_to_bounds:
        x_coords = np.clip(x_coords, 0.0, cube_side_length)
        y_coords = np.clip(y_coords, 0.0, cube_side_length)
        # Z is already within bounds due to np.random.uniform definition

    # Combine the coordinates into a single array
    points = np.column_stack((x_coords, y_coords, z_coords))

    return points


# --- Voxel Finding (Keep the optimized version) ---

def find_occupied_voxels_vectorized(points_array, voxel_size=0.5):
    """
    Efficiently identifies unique voxel origins containing points using vectorized operations.
    """
    if points_array.size == 0:
        return np.empty((0, 3))
    # Ensure points are within calculation range if they weren't clipped
    # This prevents issues with floor division if points fall slightly below 0
    points_array_clamped = np.maximum(points_array, 0)

    voxel_indices = np.floor(points_array_clamped / voxel_size)
    voxel_origins_all = voxel_indices * voxel_size
    unique_origins = np.unique(voxel_origins_all, axis=0)
    return unique_origins


# --- PyVista Plotting Function (Keep the efficient version) ---

def plot_with_pyvista(points_array, occupied_voxel_origins, voxel_size=0.5, cube_side_length=10.0):
    """
    Displays points and voxel outlines using PyVista for better performance.
    (Code remains the same as the previous PyVista example)
    """
    plotter = pv.Plotter(window_size=[800, 800])
    plotter.set_background("white")

    # 1. Add Scatter points
    if points_array.size > 0:
        point_cloud = pv.PolyData(points_array)
        plotter.add_points(point_cloud,
                           render_points_as_spheres=True,
                           point_size=5,
                           color='red',
                           opacity=0.8,
                           label='Data Points')
        print(f"Added {points_array.shape[0]} points to PyVista plotter.")

    # 2. Add Voxel outlines efficiently
    num_voxels = occupied_voxel_origins.shape[0]
    if num_voxels > 0:
        print(f"Creating {num_voxels} voxel wireframes for PyVista...")
        v_rel = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ]) * voxel_size
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
            [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]
        ])
        all_vertices = np.zeros((num_voxels * 8, 3))
        all_lines = np.zeros((num_voxels * 12, 3), dtype=int)
        for i, origin in enumerate(occupied_voxel_origins):
            base_vertex_index = i * 8
            all_vertices[base_vertex_index : base_vertex_index + 8] = v_rel + origin
            lines_for_this_voxel = np.hstack((np.full((12, 1), 2), edges + base_vertex_index))
            all_lines[i * 12 : (i + 1) * 12] = lines_for_this_voxel
        voxel_mesh = pv.PolyData(all_vertices, lines=all_lines)
        plotter.add_mesh(voxel_mesh, style='wireframe', color='blue',
                         line_width=1, opacity=0.5,
                         label=f'Occupied Voxels ({voxel_size}m)')
        print("Added voxel wireframes to PyVista plotter.")

    # Add context: Outline of the main cube
    outline_cube = pv.Cube(bounds=(0, cube_side_length, 0, cube_side_length, 0, cube_side_length))
    plotter.add_mesh(outline_cube, style='wireframe', color='grey', line_width=1, opacity=0.3, label='Bounding Cube')

    # Configure view, axes, legend
    plotter.show_grid(color='grey', location='outer')
    plotter.add_legend(bcolor=None, border=False)
    plotter.camera_position = 'iso'
    plotter.camera.azimuth = 30
    plotter.camera.elevation = 20

    print("\nShowing PyVista plot window...")
    plotter.show()
    print("PyVista plot window closed.")


# --- Main Execution ---
if __name__=='__main__':
    # Parameters
    NUM_POINTS = 5000 # Can use a decent number with PyVista
    CUBE_SIDE = 10.0
    VOXEL_SIDE = 1
    XY_STD_DEV = 1.8 # Adjust this to change clustering (smaller = tighter)

    # 1. Generate points with normal distribution on X/Y
    print(f"Generating {NUM_POINTS} points...")
    print(f"X/Y distribution: Normal(mean={CUBE_SIDE/2:.1f}, std_dev={XY_STD_DEV:.1f})")
    print(f"Z distribution: Uniform(0, {CUBE_SIDE:.1f})")
    points = create_normal_xy_points(num_points=NUM_POINTS,
                                     cube_side_length=CUBE_SIDE,
                                     std_dev_xy=XY_STD_DEV,
                                     clip_to_bounds=True) # Clip points outside 0-10

    # 2. Find occupied voxels (optimized)
    print("\nFinding occupied voxels...")
    start_time = time.time()
    voxel_origins_opt = find_occupied_voxels_vectorized(points, voxel_size=VOXEL_SIDE)
    end_time = time.time()
    print(f"Found {voxel_origins_opt.shape[0]} unique occupied voxels ({end_time - start_time:.4f}s)")

    # 3. Plot using PyVista
    print("\n--- Plotting with PyVista ---")
    start_plot_time = time.time()
    plot_with_pyvista(points, voxel_origins_opt,
                      voxel_size=VOXEL_SIDE,
                      cube_side_length=CUBE_SIDE)
    end_plot_time = time.time()
    print(f"PyVista plot setup time: {end_plot_time - start_plot_time:.4f} seconds")