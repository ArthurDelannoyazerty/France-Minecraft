from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


class BaseWriter(ABC):
    """
    Abstract base class for writing Minecraft data.
    Defines the interface that all writer implementations must follow.
    """

    @abstractmethod
    def initialize_batch(self, batch_info: Dict):
        """
        Prepare the writer for a new batch of blocks.
        'batch_info' can contain metadata like absolute coordinates.
        """
        pass

    @abstractmethod
    def place_blocks(self, coordinates: List[Tuple[int, int, int]], block_id: str, options: Dict = None):
        """
        Place a series of blocks of the same type.
        
        Args:
            coordinates: A list of (x, y, z) tuples for block positions.
            block_id: The Minecraft block ID string (e.g., 'minecraft:stone').
            options: A dictionary for any extra parameters (e.g., 'extend_to_ground').
        """
        pass
    
    @abstractmethod
    def set_ground_level(self, ground_array: np.ndarray, block_top: str, block_below: str, thickness: int):
        """
        Set the entire ground level from a 2D heightmap.

        Args:
            ground_array: 2D numpy array of Z (height) values.
            block_top: The block ID for the surface.
            block_below: The block ID for the layers underneath.
            thickness: The number of 'block_below' layers.
        """
        pass

    @abstractmethod
    def save_batch(self, batch_name: str) -> str:
        """
        Save the current batch to a file or database.
        Returns the name or identifier of the saved artifact (e.g., filename).
        """
        pass

    def finalize(self):
        """Perform any final cleanup or saving actions."""
        pass