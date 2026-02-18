#!/usr/bin/env python3
"""Automated discovery of coordinates by analyzing a sequence of moves."""

import logging
import os
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from collections import defaultdict

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

import argparse

def run_sequence(core, input_server, moves):
    for name, state, press_frames, delay_frames in moves:
        log(f"  Move '{name}': press {press_frames}f, delay {delay_frames}f")
        # Press
        input_server.set_state(state)
        for _ in range(press_frames):
            core.advance_frame()
        # Release and Delay
        input_server.clear()
        for _ in range(delay_frames):
            core.advance_frame()

def main():
    parser = argparse.ArgumentParser(description="Find memory changes after a sequence of moves.")
    parser.add_argument('--slot', type=int, default=1, help='The save state slot to load from.')
    parser.add_argument('moves', type=str, help='A string defining moves, e.g., "R:1:10,L:1:10" for right(1 frame press, 10 frames delay), left(1f press, 10f delay).')
    args = parser.parse_args()

    log("=" * 60)
    log("Simple Memory Diff Tool")
    log(f"Move sequence: {args.moves}")
    log("=" * 60)

    # Parse moves
    move_list = []
    for move_str in args.moves.split(','):
        parts = move_str.split(':')
        direction = parts[0].upper()
        press_frames = int(parts[1])
        delay_frames = int(parts[2])

        state = ControllerState()
        if direction == 'R': state.R_DPAD = 1
        elif direction == 'L': state.L_DPAD = 1
        elif direction == 'D': state.D_DPAD = 1
        elif direction == 'U': state.U_DPAD = 1
        elif direction == 'A': state.A_BUTTON = 1
        else: raise ValueError(f"Unknown move direction: {direction}")
        move_list.append((direction, state, press_frames, delay_frames))


    input_server = InputServer(host="127.0.0.1", port=8082)
    input_server.start()

    core = Mupen64PlusCore(
        core_lib_path=str(LIB_DIR / "libmupen64plus.so.2.0.0"),
        plugin_dir=str(LIB_DIR),
        data_dir=str(DATA_DIR),
    )
    memory = MemoryReader(core, use_debug_api=False)

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
    memory.refresh_pointer()

    log("\\n>>> Preparing for diff...")
    core.load_state(slot=args.slot)
    time.sleep(0.5)
    core.pause()

    # Wait for game to be controllable after load
    log("  Waiting for game to become controllable...")
    for _ in range(60):
        core.advance_frame()

    # --- Take initial snapshot ---
    log("  Taking initial snapshot (A)...")
    snap_a = memory.snapshot()
    u8_a = np.frombuffer(snap_a, dtype=np.uint8)


    # --- Run sequence and take final snapshot ---
    log("  Executing move sequence...")
    run_sequence(core, input_server, move_list)
    log("  Taking final snapshot (B)...")
    snap_b = memory.snapshot()
    u8_b = np.frombuffer(snap_b, dtype=np.uint8)


    # --- Analysis ---
    log("\\n>>> Analyzing results...")
    diff_indices = np.where(u8_a != u8_b)[0]

    log(f"  Found {len(diff_indices)} u8 addresses that changed.")

    for idx in diff_indices:
        addr = 0x80000000 + idx
        val_a = u8_a[idx]
        val_b = u8_b[idx]
        log(f"  ADDR 0x{addr:08X}: {val_a} -> {val_b}  (diff: {int(val_b) - int(val_a)})")


    log("\\n>>> Shutting down...")
    core.stop()
    input_server.stop()
    log("Done.")


if __name__ == "__main__":
    main()
