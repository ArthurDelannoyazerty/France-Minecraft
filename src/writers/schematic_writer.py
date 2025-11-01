from typing import Dict, List, Tuple

import mcschematic
import numpy as np
from tqdm.auto import tqdm

from src import config 
from src.writers.base_writer import BaseWriter


class SchematicWriter(BaseWriter):
    """
    A writer that generates Minecraft .schematic files using MCSchematic.
    """
    def __init__(self, output_dir: str, version=mcschematic.Version.JE_1_21):
        self.output_dir = output_dir
        self.version = version
        self.schem = None
        self.ground_level = None

    def initialize_batch(self, batch_info: Dict):
        """Initializes a new, empty schematic for the batch."""
        self.schem = mcschematic.MCSchematic()
        self.ground_level = None # Reset ground level for new batch

    def set_ground_level(self, ground_array: np.ndarray, block_top: str, block_below: str, thickness: int):
        """Populates the schematic with MNT ground data."""
        self.ground_level = ground_array # Store for later use
        for x in tqdm(range(ground_array.shape[0]), desc='Placing MNT', leave=False):
            for y in range(ground_array.shape[1]):
                z = ground_array[x, y]
                self.schem.setBlock((x, z, y), block_top)
                for i in range(1, thickness + 1):
                    if z - i > config.LOWEST_MINECRAFT_POINT:
                        self.schem.setBlock((x, z - i, y), block_below)

    def place_blocks(self, coordinates: List[Tuple[int, int, int]], block_id: str, options: Dict = None):
        """Places blocks in the schematic, with special handling for certain types."""
        options = options or {}
        
        # Specific logic for buildings
        if options.get('extend_to_ground'):
            self._place_buildings(coordinates, block_id)
            return

        # Generic block placement
        for x, y, z in coordinates:
            # Prevent replacing existing solid blocks, but allow replacing air or temporary blocks
            current_block = self.schem.getBlockDataAt((int(x), int(z), int(y)))
            if current_block == 'minecraft:air' or current_block == config.BLOCK_MAPPING['lidar']['No Class']:
                self.schem.setBlock((int(x), int(z), int(y)), block_id)
                
    def _place_buildings(self, coordinates: List[Tuple[int, int, int]], block_id: str):
        """Place building blocks and extend them down to the ground."""
        for x, y, z in coordinates:
            # Place the top block
            self.schem.setBlock((int(x), int(z), int(y)), block_id)
            # Extend downwards until it hits a non-air block
            for z_below in range(int(z) - 1, config.LOWEST_MINECRAFT_POINT, -1):
                if self.schem.getBlockDataAt((int(x), z_below, int(y))) != 'minecraft:air':
                    break
                self.schem.setBlock((int(x), z_below, int(y)), block_id)

    def save_batch(self, batch_name: str) -> str:
        """Saves the schematic to a file."""
        if self.schem:
            self.schem.save(self.output_dir, batch_name, self.version)
            return f"{batch_name}.schem"
        return ""