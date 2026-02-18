#!/usr/bin/env python3
"""Automated discovery of coordinates by analyzing a sequence of moves."""

import logging
import os
import sys
import time
from pathlib import Path
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

def main():
    log("=" * 60)
    log("Y Coordinate Sequence Discovery")
    log("=" * 60)

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

    # --- Execute and Record Sequence ---
    log("\\n>>> Executing move sequence and recording snapshots...")
    core.load_state(slot=1)
    time.sleep(0.5)

    # Sequence of moves: down, down, down, down, down, down
    moves = [
        ("down", ControllerState(D_DPAD=1)),
        ("down", ControllerState(D_DPAD=1)),
        ("down", ControllerState(D_DPAD=1)),
        ("down", ControllerState(D_DPAD=1)),
        ("down", ControllerState(D_DPAD=1)),
        ("down", ControllerState(D_DPAD=1)),
    ]

    snapshots = []
    changed_addresses = set()

    core.pause()
    snapshots.append(memory.snapshot()) # Initial state

    for name, state in moves:
        core.resume()
        input_server.set_state(state)
        time.sleep(1/60)
        input_server.clear()
        time.sleep(4/60) # Give game time to process move
        core.pause()

        snap = memory.snapshot()
        snapshots.append(snap)
        log(f"  Recorded snapshot after moving {name}.")

    # --- Analysis ---
    log("\\n>>> Analyzing results...")

    # Find all addresses that changed at least once
    all_changed_addr = set()
    for i in range(len(snapshots) - 1):
        for j in range(len(snapshots[0])):
            if snapshots[i][j] != snapshots[i+1][j]:
                all_changed_addr.add(0x80000000 + j)

    log(f"  Found {len(all_changed_addr)} unique addresses that changed at some point.")

    candidates = []
    for addr in all_changed_addr:
        phys_addr = addr - 0x80000000
        sequence = [snap[phys_addr] for snap in snapshots]

        # Look for a consistently increasing or decreasing pattern
        diffs = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]

        # All diffs must be non-zero
        if not all(d != 0 for d in diffs):
            continue

        # All diffs must have the same sign (either all drops increase Y or all decrease Y)
        is_monotonic = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)

        if is_monotonic:
            candidates.append((addr, sequence))

    log(f"\\n>>> Found {len(candidates)} candidates with consistent move patterns:")
    for addr, sequence in candidates:
        log(f"  Candidate: 0x{addr:08X} | Sequence: {sequence}")



    log("\\n>>> Shutting down...")
    core.stop()
    input_server.stop()
    log("Done.")


if __name__ == "__main__":
    main()
