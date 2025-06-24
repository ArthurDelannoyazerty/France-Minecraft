# import pydirectinput
import pyautogui
import pyperclip 
import time
from pathlib import Path


mcfunction_folderpath = Path('data/mcfunctions/')

TIME_AFTER_TP = 6
TIME_AFTER_SCHEMATIC_LOAD = 3
TIME_AFTER_PASTE = 6


time.sleep(5)
box_response = pyautogui.confirm('Select the minecraft window, press the "esc" key, and click on the "OK" button.')

if box_response=='Cancel':
    print('Stopping program')


pyautogui.press('esc')            # Open chat


mcfunction_files = list(mcfunction_folderpath.glob('*.mcfunction'))
for mcfunction_filepath in mcfunction_files:
        
    with open(mcfunction_filepath, 'r') as f:
        mcfunction_lines = f.readlines()

    for line in mcfunction_lines:
        line = line.strip()
        if len(line)==0: continue
        if line[0]=='#': continue

        if line[0]=='/':

            pyautogui.press('t')            # Open chat
            pyperclip.copy(line)            # Copy command in clipboard
            pyautogui.hotkey('ctrl', 'v')   # paste command in MC chat
            pyautogui.press('enter')        # execute command

            if   line.startswith('/tp'):              time.sleep(TIME_AFTER_TP)
            elif line.startswith('//schematic load'): time.sleep(TIME_AFTER_SCHEMATIC_LOAD)
            elif line.startswith('//paste'):          time.sleep(TIME_AFTER_PASTE)