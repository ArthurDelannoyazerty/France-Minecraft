import json
import logging
from pathlib import Path

from tqdm.auto import tqdm

import config
from processing.tile_processor import TileProcessor
from src.data_manager import (
    download_ign_catalogs,
    download_tiles,
    find_intersecting_tiles,
    init_folders,
)
from writers.schematic_writer import SchematicWriter

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_mcfunction(tile_bbox_str: str, batch_artifacts: list):
    """Generates a .mcfunction file to place all the schematic batches for a tile."""
    filepath = config.MCFUNCTIONS_DIR / f'{tile_bbox_str}.mcfunction'
    
    with open(filepath, 'w') as f:
        f.write('# Auto-generated MCFunction for placing lidar data\n')
        f.write('/gamerule doDaylightCycle false\n')
        f.write('/time set day\n')
        
        for i, batch in enumerate(batch_artifacts):
            f.write(f'\n/say Placing Batch {i+1}/{len(batch_artifacts)} at X={batch["abs_x"]} Z={batch["abs_y"]}\n')
            f.write(f'/tp @s {batch["abs_x"]} 100 {batch["abs_y"]}\n')
            f.write(f'/schematic load {Path(batch["artifact_name"]).stem}\n')
            f.write( '//paste -a\n')

        f.write('\nsay Lidar placement complete!\n')
    logging.info(f"Generated MCFunction: {filepath}")

def main():
    """Main function to run the entire data processing and generation pipeline."""
    
    # 1. Initialization
    init_folders()
    download_ign_catalogs()

    # 2. Find relevant tiles
    with open(config.ZONE_GEOJSON_FILE, 'r') as f:
        zone_geojson = json.load(f)
    
    compatible_tiles = find_intersecting_tiles(zone_geojson)
    if not compatible_tiles:
        logging.warning("No compatible MNT/LIDAR tiles found for the given zone. Exiting.")
        return

    # 3. Download required tile files
    download_tiles(compatible_tiles.values())

    # 4. Initialize the desired writer
    # To switch to a different output, you would just change this line:
    # writer = DirectToWorldWriter(...)
    writer = SchematicWriter(output_dir=str(config.SCHEMATICS_DIR))

    # 5. Process each tile
    for tile_bbox_str, tile_data in tqdm(compatible_tiles.items(), desc="Processing All Tiles"):
        
        mcfunction_file = config.MCFUNCTIONS_DIR / f'{tile_bbox_str}.mcfunction'
        if mcfunction_file.exists() and not config.FORCE_TILE_GENERATION:
            logging.info(f"Tile {tile_bbox_str} already processed. Skipping.")
            continue
            
        logging.info(f"Processing tile with bbox: {tile_bbox_str}")

        processor = TileProcessor(
            lidar_path=tile_data['lidar']['filepath'],
            mnt_path=tile_data['mnt']['filepath'],
            tile_bbox=tile_data['lidar']['bbox'],
            writer=writer
        )
        
        # This runs the full pipeline for one tile
        generated_artifacts = processor.run()

        # 6. Create a master mcfunction for the processed tile
        if isinstance(writer, SchematicWriter):
            create_mcfunction(tile_bbox_str, generated_artifacts)

    logging.info("--- Pipeline Finished ---")

if __name__ == '__main__':
    main()