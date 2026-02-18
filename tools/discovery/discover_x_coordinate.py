#!/usr/bin/env python3
"""Automated discovery of coordinates by analyzing a sequence of moves."""

import logging
import os
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

def main():
    log("=" * 60)
    log("X Coordinate Sequence Discovery")
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

    # Sequence of moves: right, right, right, left, left, left
    moves = [
        ("right", ControllerState(R_DPAD=1)),
        ("right", ControllerState(R_DPAD=1)),
        ("right", ControllerState(R_DPAD=1)),
        ("left", ControllerState(L_DPAD=1)),
        ("left", ControllerState(L_DPAD=1)),
        ("left", ControllerState(L_DPAD=1)),
    ]

    snapshots = []
    changed_addresses = set()

    core.pause()
    snapshots.append(memory.snapshot()) # Initial state

    for name, state in moves:
        # Press button for 1 frame
        input_server.set_state(state)
        core.advance_frame()
        input_server.clear()

        # Let game process for a few frames
        for _ in range(10):
            core.advance_frame()

        snap = memory.snapshot()
        log(f"  Snapshot {len(snapshots)} length: {len(snap)}")
        snapshots.append(snap)
        log(f"  Recorded snapshot after moving {name}.")

    # --- Analysis ---
    log("\\n>>> Analyzing results...")

    # Find all addresses that changed at least once
    all_changed_addr = set()
    for i in range(len(snapshots) - 1):
        if snapshots[i] == snapshots[i+1]:
            log(f"  Snapshots {i} and {i+1} are identical, skipping diff.")
            continue

        # Manually find differing bytes
        for j in range(len(snapshots[i])):
            if snapshots[i][j] != snapshots[i+1][j]:
                all_changed_addr.add(0x80000000 + j)

    log(f"  Found {len(all_changed_addr)} unique addresses that changed at some point.")

    # --- u8 analysis ---
    u8_candidates = []
    for addr in all_changed_addr:
        phys_addr = addr - 0x80000000
        # only check addresses that are byte-aligned
        if phys_addr % 1 != 0: continue
        sequence = [snap[phys_addr] for snap in snapshots]

        if len(sequence) != 7: continue
        diffs = np.diff(sequence).tolist()
        first_half, second_half = diffs[:3], diffs[3:]

        if (all(d == 1 for d in first_half) and all(d == -1 for d in second_half)) or \
           (all(d == -1 for d in first_half) and all(d == 1 for d in second_half)):
            u8_candidates.append((addr, sequence))

    log(f"\\n>>> Found {len(u8_candidates)} u8 candidates with matching move patterns:")
    for addr, sequence in u8_candidates:
        log(f"  U8 Candidate: 0x{addr:08X} | Sequence: {sequence}")

    # --- u16_le analysis ---
    u16_candidates = []
    # Reinterpret byte snapshots as little-endian unsigned 16-bit integers
    u16_snapshots = [np.frombuffer(s, dtype=np.uint16) for s in snapshots]

    all_changed_u16_addr = set()
    for i in range(len(u16_snapshots) - 1):
        if u16_snapshots[i].shape != u16_snapshots[i+1].shape: continue
        # Find indices where the u16 values are different
        diff_indices = np.where(u16_snapshots[i] != u16_snapshots[i+1])[0]
        for idx in diff_indices:
            all_changed_u16_addr.add(0x80000000 + (idx * 2)) # each index is 2 bytes

    log(f"  Found {len(all_changed_u16_addr)} unique u16 addresses that changed.")

    for addr in all_changed_u16_addr:
        phys_addr = addr - 0x80000000
        idx = phys_addr // 2
        sequence = [snap[idx] for snap in u16_snapshots]

        if len(sequence) != 7: continue
        diffs = np.diff(sequence).tolist()
        first_half, second_half = diffs[:3], diffs[3:]

        # The change might not be 1, just consistent.
        if first_half[0] != 0 and second_half[0] != 0 and \
           all(d == first_half[0] for d in first_half) and \
           all(d == second_half[0] for d in second_half) and \
           first_half[0] == -second_half[0]:
            u16_candidates.append((addr, sequence))

    log(f"\\n>>> Found {len(u16_candidates)} u16_le candidates with matching move patterns:")
    for addr, sequence in u16_candidates:
        log(f"  U16_LE Candidate: 0x{addr:08X} | Sequence: {sequence}")



    log("\\n>>> Shutting down...")
    core.stop()
    input_server.stop()
    log("Done.")


if __name__ == "__main__":
    main()
