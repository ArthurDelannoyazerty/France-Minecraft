from pathlib import Path

import laspy
import numpy as np
import rasterio
from tqdm.auto import tqdm

from src import config
from src.processing.lidar_utils import (
    dominant_voxel_points,
    find_occupied_voxels_vectorized,
)
from src.processing.osm_processor import get_road_coordinates_by_type
from src.writers.base_writer import BaseWriter


class TileProcessor:
    """
    Processes a single pair of MNT and LIDAR tiles to generate Minecraft data.
    """
    def __init__(self, lidar_path: Path, mnt_path: Path, tile_bbox: list[int], writer: BaseWriter):
        self.lidar_path = lidar_path
        self.mnt_path = mnt_path
        self.tile_bbox = tile_bbox
        self.writer = writer
        
        self.tile_min_x, self.tile_min_y, self.tile_max_x, self.tile_max_y = tile_bbox
        self.tile_edge_size = self.tile_max_x - self.tile_min_x

    def run(self) -> list[dict]:
        """
        Executes the full processing pipeline for the tile.
        Returns a list of metadata for each generated batch.
        """
        # 1. Load and prepare data
        mnt_array = self._load_and_clean_mnt()
        lidar_data = self._load_and_clean_lidar()
        osm_roads = self._get_osm_data()
        z_axis_offset = self._calculate_z_offset(mnt_array)

        # 2. Batch processing
        batch_size = self.tile_edge_size // config.BATCHES_PER_TILE_SIDE
        generated_artifacts = []

        for batch_x_idx in tqdm(range(config.BATCHES_PER_TILE_SIDE), desc='Batch X', position=0):
            for batch_y_idx in tqdm(range(config.BATCHES_PER_TILE_SIDE), desc='Batch Y', leave=False, position=1):
                
                # Calculate coordinates for the current batch
                xmin_rel = batch_size * batch_x_idx
                ymin_rel = batch_size * batch_y_idx
                
                xmin_abs = self.tile_min_x + xmin_rel
                # Y is inverted for Minecraft's coordinate system
                ymin_abs = -self.tile_min_y + ymin_rel 
                
                batch_info = {
                    'abs_x': xmin_abs, 'abs_y': ymin_abs, 'size': batch_size,
                    'batch_indices': (batch_x_idx, batch_y_idx)
                }

                self.writer.initialize_batch(batch_info)

                # a. Process MNT
                mnt_batch_array = mnt_array[xmin_rel : xmin_rel + batch_size, ymin_rel : ymin_rel + batch_size]
                mnt_batch_mc = mnt_batch_array + z_axis_offset
                self.writer.set_ground_level(
                    mnt_batch_mc, 
                    config.BLOCK_MAPPING['mnt']['ground_top'],
                    config.BLOCK_MAPPING['mnt']['ground_below'],
                    config.GROUND_THICKNESS
                )

                # b. Process OSM
                self._process_osm_batch(osm_roads, mnt_batch_mc, batch_info)
                
                # c. Process LIDAR
                self._process_lidar_batch(lidar_data, batch_info, z_axis_offset)
                
                # d. Save batch
                batch_name = f'xmin~{xmin_abs}_ymin~{ymin_abs}_size~{self.tile_edge_size}'
                artifact_name = self.writer.save_batch(batch_name)
                
                batch_info['artifact_name'] = artifact_name
                generated_artifacts.append(batch_info)
        
        return generated_artifacts

    def _load_and_clean_mnt(self) -> np.ndarray:
        """Loads the MNT raster, cleans errors, and resamples."""
        with rasterio.open(self.mnt_path) as mnt_dataset:
            mnt_array = mnt_dataset.read(1)
        
        # Replace -9999.0 values with mean of valid neighbors
        error_mask = (mnt_array == -9999.0)
        for i, j in np.argwhere(error_mask):
            neighbors = mnt_array[max(0, i-1):i+2, max(0, j-1):j+2]
            valid_neighbors = neighbors[neighbors != -9999.0]
            if valid_neighbors.size > 0:
                mnt_array[i, j] = np.mean(valid_neighbors)
        
        # Average pooling from 0.5m to 1m resolution
        M, N = mnt_array.shape
        pooled_mnt_array = mnt_array.reshape(M//2, 2, N//2, 2).mean(axis=(1, 3))
        
        return pooled_mnt_array.T.astype(np.int32) # Transpose for X,Y orientation

    def _load_and_clean_lidar(self) -> laspy.LasData:
        """Loads lidar data and normalizes its coordinates to be relative to the tile origin."""
        lidar = laspy.read(self.lidar_path)
        lidar.x = np.array(lidar.x) - self.tile_min_x
        # Invert Y axis and make it relative
        lidar.y = (-np.array(lidar.y) + self.tile_min_y + self.tile_edge_size)
        return lidar

    def _calculate_z_offset(self, mnt_array: np.ndarray) -> int:
        """Calculates the vertical offset to align the terrain with Minecraft's world height."""
        if config.MANUAL_Z_AXIS_TRANSLATE:
            return config.MANUAL_Z_AXIS_OFFSET
        else:
            lowest_point = mnt_array.min()
            return config.LOWEST_MINECRAFT_POINT - lowest_point
    

    def _get_osm_data(self) -> dict:
        """Fetches and processes OSM data for the entire tile."""
        if not config.DO_OSM_PROCESSING:
            return {}
        
        # Create a polygon representing the tile's bounding box in its native CRS (EPSG:2154)
        tile_poly_lambert93 = box(*self.tile_bbox)
        
        # Convert the polygon to WGS84 (EPSG:4326) for the Overpass API query
        transformer = pyproj.Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform
        tile_poly_wgs84 = transform(transformer, tile_poly_lambert93)
        
        # Fetch road data
        osm_roads = get_road_coordinates_by_type(tile_poly_wgs84)

        # Invert Y-axis to match the internal coordinate system (where Y is flipped)
        return {
            road_type: [(x, self.tile_edge_size - y) for x, y in coords]
            for road_type, coords in osm_roads.items()
        }


    def _process_osm_batch(self, osm_roads: dict, mnt_batch_mc: np.ndarray, batch_info: dict):
        """Places OSM road blocks for a single batch."""
        if not config.DO_OSM_PROCESSING or not osm_roads:
            return
            
        size = batch_info['size']
        xmin_rel = batch_info['batch_indices'][0] * size
        ymin_rel = batch_info['batch_indices'][1] * size
        xmax_rel = xmin_rel + size
        ymax_rel = ymin_rel + size

        for road_type, coords in osm_roads.items():
            if road_type not in config.BLOCK_MAPPING['osm']:
                continue
            
            block_id = config.BLOCK_MAPPING['osm'][road_type]
            road_points_in_batch = []

            for x, y in coords:
                if xmin_rel <= x < xmax_rel and ymin_rel <= y < ymax_rel:
                    # Make coordinates relative to the batch origin
                    batch_x = int(x - xmin_rel)
                    batch_y = int(y - ymin_rel)
                    
                    # Get height from MNT data and add to list
                    z = mnt_batch_mc[batch_x, batch_y]
                    road_points_in_batch.append((batch_x, batch_y, z))

            if road_points_in_batch:
                self.writer.place_blocks(road_points_in_batch, block_id, {'is_road': True})
    

    def _process_lidar_batch(self, lidar_data, mnt_batch_mc, batch_info: dict, z_offset: int):
        """Processes lidar points for a single batch."""
        if not config.DO_LIDAR_PROCESSING:
            return

        size = batch_info['size']
        xmin_rel = batch_info['batch_indices'][0] * size
        ymin_rel = batch_info['batch_indices'][1] * size

        # Filter points within the current batch bounds
        mask = (lidar_data.x >= xmin_rel) & (lidar_data.x < xmin_rel + size) & \
               (lidar_data.y >= ymin_rel) & (lidar_data.y < ymin_rel + size)
        lidar_batch = lidar_data[mask]

        if len(lidar_batch.points) == 0:
            return
            
        point_coords_by_class = {}
        for cls_id, min_points in config.LIDAR_CLASS_MIN_POINTS.items():
            class_mask = lidar_batch.classification == cls_id
            points = lidar_batch[class_mask]
            
            xyz = np.vstack([points.x, points.y, points.z + z_offset]).T
            # Make coordinates relative to the batch origin
            xyz_relative = xyz - [xmin_rel, ymin_rel, 0]
            
            voxel_origins = find_occupied_voxels_vectorized(
                xyz_relative,
                voxel_size=config.VOXEL_SIZE,
                min_points_per_voxel=min_points
            )
            point_coords_by_class[cls_id] = voxel_origins.tolist()

        # Determine dominant class per voxel and place blocks
        dominant_voxels = dominant_voxel_points(point_coords_by_class, grid_size=size)
        
        # Group coordinates by dominant class
        coords_by_class = {cls_id: [] for cls_id in config.LIDAR_CLASS_MIN_POINTS}
        for (vx, vy, vz), cls_id in dominant_voxels.items():
            coords_by_class[cls_id].append((vx, vy, vz))

        self.writer.place_blocks(coords_by_class.get(6), config.BLOCK_MAPPING['lidar']['Building'], {'extend_to_ground': True})
        self.writer.place_blocks(coords_by_class.get(17), config.BLOCK_MAPPING['lidar']['Bridge'])
        self.writer.place_blocks(coords_by_class.get(9), config.BLOCK_MAPPING['lidar']['Water'])
        self.writer.place_blocks(coords_by_class.get(5), config.BLOCK_MAPPING['lidar']['High Vegetation'])
        self.writer.place_blocks(coords_by_class.get(4), config.BLOCK_MAPPING['lidar']['Medium Vegetation'])
        self.writer.place_blocks(coords_by_class.get(3), config.BLOCK_MAPPING['lidar']['Small Vegetation'])
        self.writer.place_blocks(coords_by_class.get(64), config.BLOCK_MAPPING['lidar']['Perennial Soil'])
        self.writer.place_blocks(coords_by_class.get(66), config.BLOCK_MAPPING['lidar']['Virtual Points'])
        self.writer.place_blocks(coords_by_class.get(67), config.BLOCK_MAPPING['lidar']['Miscellaneous'])
        self.writer.place_blocks(coords_by_class.get(1), config.BLOCK_MAPPING['lidar']['No Class'])