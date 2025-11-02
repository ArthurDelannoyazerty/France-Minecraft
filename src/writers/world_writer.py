import logging
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import amulet
import numpy as np
from amulet.api.block import Block
from amulet.api.errors import ChunkDoesNotExist, ChunkLoadError
from amulet.utils.world_utils import block_coords_to_chunk_coords
from tqdm.auto import tqdm

from src import config
from src.writers.base_writer import BaseWriter

logger = logging.getLogger(__name__)


class WorldWriter(BaseWriter):
    """
    A writer that directly modifies a Minecraft world file using the Amulet library.
    This writer handles absolute coordinates and is optimized for performance by
    caching block changes and writing them chunk by chunk.
    """

    def __init__(self, world_path: str, dimension: str = "minecraft:overworld"):
        self.world_path = Path(world_path)
        self.dimension = dimension
        if not self.world_path.exists() or not self.world_path.is_dir():
            raise FileNotFoundError(f"Minecraft world not found at path: {self.world_path}")

        if config.AUTO_BACKUP_WORLD:
            self._backup_world()

        logger.info(f"Loading Minecraft world from: {self.world_path}")
        self.world = amulet.load_level(str(self.world_path))



        self.air = Block("universal_minecraft", "air")
        self._block_cache: Dict[str, Block] = {}
        self._block_palette_cache: Dict[Block, int] = {}
        
        # Buffer to hold block changes, organized by chunk coordinates
        self._chunk_buffer = defaultdict(dict)
        self.batch_origin_x = 0
        self.batch_origin_z = 0

    def _backup_world(self):
        """Creates a timestamped zip backup of the world directory."""
        backup_dir = self.world_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_filename = backup_dir / f"{self.world_path.name}-{timestamp}"
        
        logger.info(f"Creating backup of '{self.world_path.name}'...")
        shutil.make_archive(str(backup_filename), 'zip', self.world_path)
        logger.info(f"World backup saved to '{backup_filename}.zip'")

    def _get_block(self, block_id: str) -> Block:
        """
        Retrieves an Amulet Block object from a string ID, using a cache.
        Handles complex block states like 'minecraft:oak_stairs[facing=east]'.
        """
        if block_id not in self._block_cache:
            namespace, base_name_and_properties = block_id.split(':', 1)
            if '[' in base_name_and_properties:
                base_name, properties_str = base_name_and_properties.split('[', 1)
                properties_str = properties_str.rstrip(']')
                properties = dict(prop.split('=', 1) for prop in properties_str.split(','))
            else:
                base_name = base_name_and_properties
                properties = {}
            
            self._block_cache[block_id] = Block(namespace, base_name, properties)
        return self._block_cache[block_id]

    def initialize_batch(self, batch_info: Dict):
        """
        Sets the absolute world coordinates for the origin of the new batch
        and flushes the buffer from the previous batch.
        """
        self._flush_chunk_buffer()
        self.batch_origin_x = batch_info['abs_x']
        self.batch_origin_z = batch_info['abs_y']
        logger.debug(f"Initialized batch with origin (X:{self.batch_origin_x}, Z:{self.batch_origin_z})")

    def _add_to_buffer(self, abs_x: int, abs_y: int, abs_z: int, block_id: int):
        """Adds a block's internal ID to the chunk buffer."""
        # MODIFICATION: Use height limits from config instead of from the world file.
        # This allows placing blocks outside of vanilla limits when using mods.
        if not (config.LOWEST_MINECRAFT_POINT <= abs_y < config.HIGHEST_MINECRAFT_POINT):
            return  # Don't place blocks outside the configured height limits

        cx, cz = block_coords_to_chunk_coords(abs_x, abs_z)
        offset_x, offset_z = abs_x - 16 * cx, abs_z - 16 * cz
        self._chunk_buffer[(cx, cz)][(offset_x, abs_y, offset_z)] = block_id

    def set_ground_level(self, ground_array: np.ndarray, block_top: str, block_below: str, thickness: int):
        """
        Adds ground level blocks to the buffer for chunk-based writing.
        """
        top_block_univ = self._get_block(block_top)
        below_block_univ = self._get_block(block_below)
        
        # The internal ID in the world's global palette
        top_block_id = self.world.block_palette.get_add_block(top_block_univ)
        below_block_id = self.world.block_palette.get_add_block(below_block_univ)

        done_log = False

        height, width = ground_array.shape
        for x_rel in tqdm(range(height), desc='Buffering Ground', leave=False):
            done_log=False
            for y_rel in range(width):
                abs_x = self.batch_origin_x + x_rel
                abs_z = self.batch_origin_z + y_rel
                abs_y_top = int(ground_array[x_rel, y_rel])

                self._add_to_buffer(abs_x, abs_y_top, abs_z, top_block_id)
                
                if not done_log:
                    logger.info(f'bloack added. X Y Z: {abs_x} {abs_y_top} {abs_z}')

                for i in range(1, thickness + 1):
                    self._add_to_buffer(abs_x, abs_y_top - i, abs_z, below_block_id)

    def place_blocks(self, coordinates: List[Tuple[int, int, int]], block_id: str, options: Dict = None):
        """
        Adds a series of blocks to the buffer for chunk-based writing.
        """
        if not coordinates:
            return
            
        options = options or {}
        block_univ = self._get_block(block_id)
        internal_block_id = self.world.block_palette.get_add_block(block_univ)
        
        coords_to_place = []
        for x_rel, y_rel, z_rel in coordinates:
            abs_x = self.batch_origin_x + int(x_rel)
            abs_y = int(z_rel)
            abs_z = self.batch_origin_z + int(y_rel)
            coords_to_place.append((abs_x, abs_y, abs_z))

        if options.get('extend_to_ground'):
            self._place_buildings(coords_to_place, internal_block_id)
        else:
            for abs_x, abs_y, abs_z in coords_to_place:
                 self._add_to_buffer(abs_x, abs_y, abs_z, internal_block_id)

    def _place_buildings(self, coordinates: List[Tuple[int, int, int]], block_id: int):
        """
        Adds building blocks to the buffer, extending them down to the ground.
        Note: This version is less efficient as it needs to read existing block data.
        """
        for abs_x, abs_y, abs_z in coordinates:
            # Place the top block of the building
            self._add_to_buffer(abs_x, abs_y, abs_z, block_id)

            # MODIFICATION: Use the lower bound from config for the loop limit.
            # This ensures buildings can extend down into negative Y coordinates.
            for y_below in range(abs_y - 1, config.LOWEST_MINECRAFT_POINT, -1):
                try:
                    # This is slow, but necessary to know where the ground is.
                    # A better approach might be to pass the MNT array to this function.
                    if self.world.get_block(abs_x, y_below, abs_z, self.dimension).base_name != 'air':
                        break
                except ChunkLoadError:
                    pass  # If chunk doesn't exist, it's effectively air.
                self._add_to_buffer(abs_x, y_below, abs_z, block_id)

    def _flush_chunk_buffer(self):
        """Writes the buffered block changes to the world."""
        if not self._chunk_buffer:
            return
        
        logger.info(f"Writing data for {len(self._chunk_buffer)} chunks...")
        for (cx, cz), blocks in tqdm(self._chunk_buffer.items(), desc="Writing Chunks", leave=False):
            try:

                try:
                    # First, try to get the chunk.
                    chunk = self.world.get_chunk(cx, cz, self.dimension)
                except (ChunkDoesNotExist, ChunkLoadError):
                    # If it doesn't exist or fails to load (common for new areas), create it.
                    chunk = self.world.create_chunk(cx, cz, self.dimension)
                
                # Create a numpy array of block IDs to write
                # This is much faster than setting block by block
                block_array = chunk.blocks
                
                for (x, y, z), block_internal_id in blocks.items():
                    # Only write if the block is currently air
                    # if chunk.block_palette[block_array[x, y, z]] == self.air:
                    block_array[x, y, z] = block_internal_id
                
                chunk.blocks = block_array
                chunk.changed = True

            except ChunkLoadError:
                # If a chunk doesn't exist, Amulet should create it. If it fails to load,
                # it might be corrupted, but we can try creating a new one to place blocks.
                try:
                    logger.warning(f"Could not load chunk at ({cx}, {cz}). Trying to create a new one.")
                    chunk = self.world.create_chunk(cx, cz, self.dimension)
                    block_array = chunk.blocks
                    for (x, y, z), block_internal_id in blocks.items():
                        block_array[x, y, z] = block_internal_id
                    chunk.blocks = block_array
                    chunk.changed = True
                except Exception as e:
                    logger.error(f"Failed to create new chunk at ({cx}, {cz}): {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred while writing to chunk ({cx}, {cz}): {e}")
        
        self._chunk_buffer.clear()

    def save_batch(self, batch_name: str) -> str:
        """
        For this writer, saving is deferred. This method now flushes the chunk buffer.
        """
        self._flush_chunk_buffer()
        return f"Batch {batch_name} has been processed and buffered for writing."

    def finalize(self):
        """
        Flushes any remaining blocks in the buffer, saves all changes, and closes the world.
        """
        if self.world:
            logger.info("Finalizing process...")
            self._flush_chunk_buffer()
            logger.info("Saving all changes to the Minecraft world. This may take a moment.")
            self.world.save()
            self.world.close()
            logger.info("World saved successfully.")