#!/usr/bin/env python3
"""Starts a new game, lets the first piece fall, and saves the state."""

import logging
import os
import argparse
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.emulator.core import Mupen64PlusCore
from src.emulator.input_server import ControllerState, InputServer
from src.emulator.memory import MemoryReader

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
DATA_DIR = LIB_DIR / "lib/data"
ROM_PATH = "/mnt/c/code/n64/roms/New Tetris, The (USA).z64"

def run_sequence(core, input_server, moves):
    for name, state, press_frames, delay_frames in moves:
        log(f"  Move '{name}': press {press_frames}f, delay {delay_frames}f")
        input_server.set_state(state)
        for _ in range(press_frames):
            core.advance_frame()
        input_server.clear()
        for _ in range(delay_frames):
            core.advance_frame()

def main():
    parser = argparse.ArgumentParser(description="Play to second piece and save state.")
    parser.add_argument('slot', type=int, help='The save state slot to use (e.g., 2).')
    args = parser.parse_args()

    log("=" * 60)
    log(f"Play and Save Tool (Slot {args.slot})")
    log("=" * 60)

    # Menu navigation sequence
    # START > START > A > A > A > Down > Right > Down > A
    state_start = ControllerState(START_BUTTON=1)
    state_a = ControllerState(A_BUTTON=1)
    state_down = ControllerState(D_DPAD=1)
    state_right = ControllerState(R_DPAD=1)

    menu_sequence = [
        ("START", state_start, 4, 60),  # Skip intro
        ("START", state_start, 4, 120), # Title screen
        ("A",     state_a,     4, 60),  # Enter menu
        ("A",     state_a,     4, 60),  # Confirm
        ("A",     state_a,     4, 60),  # Select player
        ("DOWN",  state_down,  4, 20),  # To Sprint
        ("RIGHT", state_right, 4, 20),
        ("DOWN",  state_down,  4, 20),
        ("A",     state_a,     4, 60),  # Start game
    ]

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
        input=str(LIB_DIR / "mupen64plus-input-bot.so"),
        rsp=str(LIB_DIR / "mupen64plus-rsp-hle.so"),
    )

    log("\\n>>> Starting emulation...")
    core.execute()
    time.sleep(1)
    core.pause()

    log("\\n>>> Navigating menu...")
    run_sequence(core, input_server, menu_sequence)

    log("\\n>>> Waiting for first piece to drop...")
    # Wait for countdown and for first piece to fall
    for _ in range(700): # Approx 11-12 seconds
        core.advance_frame()

    log(f"\\n>>> Saving state to slot {args.slot}...")
    core.save_state(slot=args.slot)
    time.sleep(1) # Ensure save completes

    log("\\n>>> Shutting down...")
    core.stop()
    input_server.stop()
    log("Done.")

if __name__ == "__main__":
    main()
