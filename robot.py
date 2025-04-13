# import pydirectinput
import pyautogui
import pyperclip 
import time
from pathlib import Path


mcfunction_filepath = Path('data/mcfunctions/LHD_FXX_0440_6718_PTS_C_LAMB93_IGN69.mcfunction')

TIME_AFTER_TP = 1
TIME_AFTER_SCHEMATIC_LOAD = 0.2
TIME_AFTER_PASTE = 5

with open(mcfunction_filepath, 'r') as f:
    mcfunction_lines = f.readlines()

time.sleep(5)
box_response = pyautogui.confirm('Select the minecraft window, press the "esc" key, and click on the "OK" button.')

if box_response=='Cancel':
    print('Stopping program')


pyautogui.press('esc')            # Open chat


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