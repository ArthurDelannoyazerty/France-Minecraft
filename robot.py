# import pydirectinput
import pyautogui
import pyperclip 
import time
import json
from tqdm.auto import tqdm
from pathlib import Path


# ---------------------------------- CONFIG ---------------------------------- #
worlds_catalog_filepath = Path('data/worlds_catalog.json')
mcfunction_folderpath = Path('data/mcfunctions/')

TIME_AFTER_TP = 4
TIME_AFTER_SCHEMATIC_LOAD = 3
TIME_AFTER_PASTE = 14


# ---------------------------- Open World catalog ---------------------------- #

if not worlds_catalog_filepath.exists():
    worlds_catalog_filepath.write_text('{}')

worlds_catalog = json.loads(worlds_catalog_filepath.read_text())


# ----------------------------- Input world name ----------------------------- #
world_name = input('Type your world name : ')

confirm = input(f'Is your world name \"{world_name}\" ? (Y/N) ')

if confirm.strip().lower()!='y' and confirm.strip().lower()!='yes':
    print('Stopping program')
    exit(0)


# ------------------ Checking which tile to add to the world ----------------- #

# Open the curent mc world catalog
if world_name not in worlds_catalog:
    worlds_catalog[world_name] = {'mc_function_filename':[]}

existing_tiles_in_mc_world:list[str] = worlds_catalog[world_name]['mc_function_filename']
existing_tiles_in_mc_world

# Print the current world catalog
print('Here are the tile already in that world : ')
for existing_tile in existing_tiles_in_mc_world:
    existing_tile_filename = Path(existing_tile).name
    print(f'\t- {existing_tile_filename}')

# Find the full tile catalog
mcfunction_files = list(mcfunction_folderpath.glob('*.mcfunction'))

# Find the tiles not already in the current mc world
non_existing_mcfunction_files:list[Path] = list()
for mcfunction_file in mcfunction_files:
    if mcfunction_file.name not in existing_tiles_in_mc_world:
        non_existing_mcfunction_files.append(mcfunction_file)

# Print what tiles are not in the current mc world
print('Here are the tile that do not exist in that world : ')
for non_existing_tile in non_existing_mcfunction_files:
    non_existing_tile_filename = Path(non_existing_tile).name
    print(f'\t- {non_existing_tile_filename}')

# print the summary to do
print(f'Total tiles :            {len(mcfunction_files)}')
print(f'Tiles already in world : {len(existing_tiles_in_mc_world)}')
print(f'Tiles to add :           {len(non_existing_mcfunction_files)}')


if len(non_existing_mcfunction_files)==0:
    print('No tile to add. Closing program')
    exit(0)




# -------------------------------- Popup setup ------------------------------- #
time.sleep(2)
box_response = pyautogui.confirm('Select the minecraft window, press the "esc" key, and click on the "OK" button.')

if box_response=='Cancel':
    print('Stopping program')
    exit(0)


# ------------------- Begin the loop over the tiles to add ------------------- #
pyautogui.press('esc')            # Close MC menu


for mcfunction_filepath in tqdm(non_existing_mcfunction_files, desc='Adding tiles to the world', leave=True):
    with open(mcfunction_filepath, 'r') as f:
        mcfunction_lines = f.readlines()
    print(f'------------------------ Reading file : {mcfunction_filepath}')

    for line in mcfunction_lines:
        line = line.strip()
        if len(line)==0: continue
        if line[0]=='#': continue

        if line[0]=='/':
            pyperclip.copy(line)            # Copy command in clipboard
            time.sleep(0.1)
            pyautogui.press('t')            # Open chat
            # time.sleep(0.2)
            pyautogui.hotkey('ctrl', 'v')   # paste command in MC chat
            time.sleep(0.1)
            pyautogui.press('enter')        # execute command

            if   line.startswith('/say'):             print(line)
            if   line.startswith('/tp'):              time.sleep(TIME_AFTER_TP)
            elif line.startswith('//schematic load'): time.sleep(TIME_AFTER_SCHEMATIC_LOAD)
            elif line.startswith('//paste'):          time.sleep(TIME_AFTER_PASTE)
    
    # Save the catalog for each tile added
    worlds_catalog[world_name]['mc_function_filename'].append(mcfunction_filepath.name)
    worlds_catalog_filepath.write_text(json.dumps(worlds_catalog, indent=4))