from mcrcon import MCRcon
from pathlib import Path
import time
import os

RCON_HOST = "localhost"
RCON_PORT = 25575
RCON_PASSWORD = "aaamc"
SCHEM_DIR = "/path/to/schematics"
PLAYER_NAME = "your_name"

tile_name = 'LHD_FXX_1016_6293_PTS_C_LAMB93_IGN69'
schems_folderpath = Path('data/myschems') / tile_name

with MCRcon(RCON_HOST, RCON_PASSWORD, port=RCON_PORT) as mcr:


    mcr.command(f"gamemode spectator {PLAYER_NAME}")
    for filename in os.listdir(schems_folderpath):
        splited_filename = filename.split('.')[0].split('~')
        x = splited_filename[1].split('_')[1]
        y = splited_filename[2].split('_')[1]
        print(filename, x, y)

        # mcr.command(f"tp {PLAYER_NAME} {x} 0 {y}")
        # mcr.command(f"say TP")
        # time.sleep(10)
        mcr.command(f"schematic load {tile_name}/{filename}")
        mcr.command(f"say LOAD")
        time.sleep(2)
        mcr.command(f"//paste -a 0 0 0")
        mcr.command(f"say PASTE")
        time.sleep(2)
        break
        
    mcr.command(f"gamemode creative {PLAYER_NAME}")
