#!/usr/bin/env python3
"""
Launches The New Tetris, navigates to Sprint mode,
creates a save state in slot 1, and exits.
"""
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.emulator.core import Mupen64PlusCore
from src.emulator.input_server import ControllerState, InputServer

if "GALLIUM_DRIVER" not in os.environ:
    os.environ["GALLIUM_DRIVER"] = "d3d12"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

def log(msg=""):
    print(msg, flush=True)

LIB_DIR = PROJECT_DIR / "lib"
DATA_DIR = LIB_DIR / "data"
ROM_PATH = "/mnt/c/code/n64/roms/New Tetris, The (USA).z64"

def press_button(input_server, button, hold_sec=0.05, pause_sec=0.5):
    """Press button in real-time."""
    log(f"  ... Pressing {button} (hold {hold_sec}s, pause {pause_sec}s)")
    state = ControllerState()
    setattr(state, button, 1)
    input_server.set_state(state)
    time.sleep(hold_sec)
    input_server.clear()
    time.sleep(pause_sec)

def main():
    log("=" * 60)
    log("Navigate and Save State")
    log("=" * 60)

    input_server = InputServer(host="127.0.0.1", port=8082)
    input_server.start()

    core = Mupen64PlusCore(
        core_lib_path=str(LIB_DIR / "libmupen64plus.so.2.0.0"),
        plugin_dir=str(LIB_DIR),
        data_dir=str(DATA_DIR),
    )

    core.startup()
    core.load_rom(ROM_PATH)
    core.attach_plugins(
        gfx=str(LIB_DIR / "mupen64plus-video-GLideN64.so"),
        audio=None,
        input=str(LIB_DIR / "mupen64plus-input-bot.so"),
        rsp=str(LIB_DIR / "mupen64plus-rsp-hle.so"),
    )

    log("\n>>> Starting emulation...")
    core.execute()
    log(">>> Waiting 10s for ROM to load...")
    time.sleep(18)  # Wait for emulator and ROM to initialize

    # --- Menu Navigation ---
    log(">>> Navigating menus with definitive sequence and timing...")
    # Sequence: START > START > A > A > A > Down > Right > Down > A
    press_button(input_server, "START_BUTTON", hold_sec=0.1, pause_sec=3.0) # Skip intro
    press_button(input_server, "START_BUTTON", hold_sec=0.1, pause_sec=3.0) # Pass title
    press_button(input_server, "A_BUTTON", hold_sec=0.5, pause_sec=3.0) # Enter menu
    press_button(input_server, "A_BUTTON", hold_sec=0.4, pause_sec=2.5) # Advance
    press_button(input_server, "A_BUTTON", hold_sec=0.4, pause_sec=2.0) # Select player
    press_button(input_server, "D_DPAD", hold_sec=0.1, pause_sec=2.0) # To Marathon
    press_button(input_server, "R_DPAD", hold_sec=0.1, pause_sec=2.0) # To Sprint
    press_button(input_server, "D_DPAD", hold_sec=0.1, pause_sec=2.0) # To sub-selection
    press_button(input_server, "A_BUTTON", hold_sec=0.1, pause_sec=2.0) # Start game

    log(">>> Waiting for game to start...")
    time.sleep(10) # Wait for countdown: 3-2-1-GO

    log(">>> Saving state to slot 1...")
    core.save_state(slot=1)
    time.sleep(1)

    log(">>> Shutting down.")
    core.stop()
    input_server.stop()
    log("Done.")

if __name__ == "__main__":
    main()
