#!/usr/bin/env python3
"""Focused memory discovery v4 — find board, piece, and state addresses.

This version uses a much more robust cross-filtering strategy to eliminate noise.

Strategy:
1.  Run three separate actions: fall, move right, and rotate.
2.  Record the set of memory addresses that change during each action.
3.  The true addresses for the piece's state (X, Y, rotation) MUST change
    during all three actions. Therefore, the intersection of the three sets
    of changing addresses will produce a very small, high-confidence list
    of state candidates.
4.  After finding the state candidates, perform a hard drop. When the piece
    lands and the next piece spawns, the state variables will reset.
5.  Diff the memory again to find which other addresses changed at the exact
    same time as the state variables. This isolates the piece type address.
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
from src.emulator.memory import MemoryReader, RDRAM_SIZE

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

def press_button(input_server, button, hold_sec=0.15, pause_sec=0.5):
    """Press button in real-time."""
    state = ControllerState()
    setattr(state, button, 1)
    input_server.set_state(state)
    time.sleep(hold_sec)
    input_server.clear()
    time.sleep(pause_sec)

def snapshot_paused(core, memory):
    """Pause, take snapshot, return it. Leaves emulator paused."""
    core.pause()
    time.sleep(0.1)
    return memory.snapshot()

def diff_snapshots(snap_a, snap_b):
    """Returns dict of {virtual_addr: (old_val, new_val)}."""
    diffs = {}
    for i in range(RDRAM_SIZE):
        if snap_a[i] != snap_b[i]:
            diffs[0x80000000 + i] = (snap_a[i], snap_b[i])
    return diffs

def main():
    log("=" * 60)
    log("Focused Memory Discovery v4 — The New Tetris")
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
        audio=None,
        input=str(LIB_DIR / "mupen64plus-input-bot.so"),
        rsp=str(LIB_DIR / "mupen64plus-rsp-hle.so"),
    )

    log("\\n>>> Starting emulation...")
    core.execute()
    time.sleep(2)
    memory.refresh_pointer()
    log(">>> RDRAM pointer acquired\\n")

    # --- Phase 1: Isolate State Variables (X, Y, Rotation) ---
    log(">>> Phase 1: Finding common state variables...")

    # Action 1: Fall
    log("  Action 1: Natural fall...")
    core.load_state(slot=1)
    time.sleep(1)
    snap_pre_fall = snapshot_paused(core, memory)
    core.resume()
    time.sleep(1)
    snap_post_fall = snapshot_paused(core, memory)
    fall_diffs = diff_snapshots(snap_pre_fall, snap_post_fall)
    fall_changed_addrs = set(fall_diffs.keys())
    log(f"    {len(fall_changed_addrs)} addresses changed.")

    # Action 2: Move Right
    log("  Action 2: Move right...")
    core.load_state(slot=1)
    time.sleep(1)
    snap_pre_move = snapshot_paused(core, memory)
    core.resume()
    press_button(input_server, "R_DPAD", hold_sec=0.1, pause_sec=0.2)
    snap_post_move = snapshot_paused(core, memory)
    move_diffs = diff_snapshots(snap_pre_move, snap_post_move)
    move_changed_addrs = set(move_diffs.keys())
    log(f"    {len(move_changed_addrs)} addresses changed.")

    # Action 3: Rotate
    log("  Action 3: Rotate...")
    core.load_state(slot=1)
    time.sleep(1)
    snap_pre_rot = snapshot_paused(core, memory)
    core.resume()
    press_button(input_server, "A_BUTTON", hold_sec=0.1, pause_sec=0.2)
    snap_post_rot = snapshot_paused(core, memory)
    rot_diffs = diff_snapshots(snap_pre_rot, snap_post_rot)
    rot_changed_addrs = set(rot_diffs.keys())
    log(f"    {len(rot_changed_addrs)} addresses changed.")

    # Analysis 1: Find intersection
    state_candidates = fall_changed_addrs & move_changed_addrs & rot_changed_addrs
    log(f"\\n>>> Found {len(state_candidates)} high-confidence state candidates (X, Y, Rot):")
    for addr in sorted(list(state_candidates)):
        v_fall = fall_diffs[addr]
        v_move = move_diffs[addr]
        v_rot = rot_diffs[addr]
        log(f"  - 0x{addr:08X}")
        log(f"      Fall:   {v_fall[0]:3d} -> {v_fall[1]:3d}")
        log(f"      Move:   {v_move[0]:3d} -> {v_move[1]:3d}")
        log(f"      Rotate: {v_rot[0]:3d} -> {v_rot[1]:3d}")

    # --- Phase 2: Isolate Piece Type ---
    log("\\n>>> Phase 2: Finding piece type address...")
    core.load_state(slot=1)
    time.sleep(1)
    snap_pre_drop = snapshot_paused(core, memory)
    core.resume()
    press_button(input_server, "U_DPAD", hold_sec=0.1, pause_sec=1.0)
    snap_post_drop = snapshot_paused(core, memory)
    drop_diffs = diff_snapshots(snap_pre_drop, snap_post_drop)

    # The state vars should reset. Let's find addrs that changed *with* them.
    state_change_addrs = {k for k, v in drop_diffs.items() if k in state_candidates}

    if not state_change_addrs:
        log("!!! ERROR: State candidates did not change value during hard drop.")
    else:
        log(f"\\nState candidates changed as expected. Now finding correlated changes...")

        # Any other address that changed is a piece type candidate
        type_candidates = set(drop_diffs.keys()) - state_candidates

        log(f"Found {len(type_candidates)} piece type candidates.")
        log("Top 20 candidates (showing small integer value changes):")

        count = 0
        for addr in sorted(list(type_candidates)):
            old, new = drop_diffs[addr]
            # Good candidates change from one small int to another (0-10)
            if 0 <= old <= 10 and 0 <= new <= 10 and old != new:
                log(f"  - 0x{addr:08X}: {old} -> {new}")
                count += 1
                if count >= 20:
                    break

    log("\\n>>> Shutting down...")
    try:
        core.stop()
    except Exception:
        pass
    input_server.stop()
    log("Done.")

if __name__ == "__main__":
    main()
