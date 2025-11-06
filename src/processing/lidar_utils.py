import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

Point = Tuple[float, float, float]
Class = str

def find_occupied_voxels_vectorized(points_array: np.ndarray,
                                    voxel_size: float = 0.5,
                                    min_points_per_voxel: int = 3):
    """Efficiently identifies unique voxel origins using vectorized operations."""
    if points_array.size == 0:
        return np.empty((0, 3))

    voxel_indices = np.round(points_array / voxel_size).astype(np.int32)
    unique_indices, counts = np.unique(voxel_indices, axis=0, return_counts=True)
    
    threshold_mask = counts >= min_points_per_voxel
    filtered_indices = unique_indices[threshold_mask]
    
    return filtered_indices * voxel_size


def dominant_voxel_points(point_coordinates: Dict[Class, List[Point]], grid_size: float) -> Dict[Tuple[int, int, int], Class]:
    """
    Determines the dominant class for each voxel based on point counts.
    """
    voxel_counts: Dict[Tuple[int, int, int], Counter] = defaultdict(Counter)
    for cls, pts in point_coordinates.items():
        for x, y, z in pts:
            if not (0 <= x < grid_size and 0 <= y < grid_size):
                continue
            voxel = (math.floor(x), math.floor(y), math.floor(z))
            voxel_counts[voxel][cls] += 1

    dominant_per_voxel: Dict[Tuple[int, int, int], Class] = {}
    for voxel, counts in voxel_counts.items():
        # Sort by count (desc) and then class name (asc) to break ties
        dominant_cls, _ = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        dominant_per_voxel[voxel] = dominant_cls
        
    return dominant_per_voxel